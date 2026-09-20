use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use quinn::{ClientConfig, Endpoint, ServerConfig};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use super::inbox::Inbox;
use super::protocol::{self, MeshMessage, PROTOCOL_VERSION};
use super::MeshTransport;
use crate::config::MeshSecurityMode;
use crate::error::{MeshError, Result};
use crate::peer::NodeId;

/// QUIC-based transport using quinn + rustls.
pub struct QuicTransport {
    local_id: NodeId,
    endpoint: Mutex<Option<Endpoint>>,
    /// Connected peers: NodeId -> QUIC connection.
    connections: DashMap<Vec<u8>, quinn::Connection>,
    /// Inbound message channel.
    inbox: Arc<Inbox>,
    /// Security mode for TLS certificate verification.
    security_mode: MeshSecurityMode,
    /// Optional path to persist TOFU certificate pins across restarts.
    tofu_pins_path: Option<PathBuf>,
}

impl QuicTransport {
    pub fn new(
        local_id: NodeId,
        security_mode: MeshSecurityMode,
        tofu_pins_path: Option<PathBuf>,
    ) -> Self {
        #[cfg(feature = "insecure-dev")]
        if security_mode == MeshSecurityMode::Insecure {
            warn!("⚠️  Mesh security mode is INSECURE. Certificate verification is disabled. Do NOT use in production!");
        }
        Self {
            local_id,
            endpoint: Mutex::new(None),
            connections: DashMap::new(),
            inbox: Arc::new(Inbox::new()),
            security_mode,
            tofu_pins_path,
        }
    }

    fn node_key(id: &NodeId) -> Vec<u8> {
        id.0.to_bytes().to_vec()
    }

    /// TLS server name used when dialling `addr`.
    ///
    /// Every connection used to be made with the literal name
    /// "hivebear-mesh", and `TofuVerifier` keys its pin map on the server
    /// name — so the map only ever held one entry and the *second* distinct
    /// peer a process dialled was rejected as a fingerprint mismatch
    /// ("possible MITM"). A mesh where each node can hold one peer is not a
    /// mesh. Deriving the name from the peer address makes the pinning
    /// per-peer, which is what TOFU means everywhere else (ssh pins per host).
    ///
    /// The address is sanitised into a single DNS-safe label. The custom
    /// verifier ignores certificate names entirely, so this only has to be
    /// unique and parseable, not resolvable.
    fn peer_server_name(addr: SocketAddr) -> String {
        let label: String = addr
            .to_string()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        format!("{label}.hivebear-mesh")
    }

    /// Make sure a rustls `CryptoProvider` is the process default.
    ///
    /// `TofuVerifier::verify_tls1{2,3}_signature` calls
    /// `CryptoProvider::get_default().expect(...)`. Nothing in this workspace
    /// installed one — it happened to work only because
    /// `ClientConfig::builder()` installs the crate-feature default as a side
    /// effect. Doing it explicitly means that `expect` cannot abort a peer
    /// connection if that side effect ever goes away.
    fn ensure_crypto_provider() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            if rustls::crypto::CryptoProvider::get_default().is_none() {
                let _ = rustls::crypto::ring::default_provider().install_default();
            }
        });
    }

    /// Generate self-signed TLS certificate with the configured security mode.
    fn generate_self_signed_config(
        security_mode: MeshSecurityMode,
        tofu_pins_path: Option<PathBuf>,
    ) -> Result<(ServerConfig, ClientConfig)> {
        Self::ensure_crypto_provider();

        let cert = rcgen::generate_simple_self_signed(vec!["hivebear-mesh".into()])
            .map_err(|e| MeshError::Transport(format!("cert generation: {e}")))?;

        let cert_der = rustls::pki_types::CertificateDer::from(cert.cert);
        let key_der = rustls::pki_types::PrivateKeyDer::try_from(cert.key_pair.serialize_der())
            .map_err(|e| MeshError::Transport(format!("key conversion: {e}")))?;

        let server_config = ServerConfig::with_single_cert(vec![cert_der.clone()], key_der)
            .map_err(|e| MeshError::Transport(format!("server config: {e}")))?;

        let verifier: Arc<dyn rustls::client::danger::ServerCertVerifier> = match security_mode {
            #[cfg(feature = "insecure-dev")]
            MeshSecurityMode::Insecure => {
                warn!("⚠️  Using INSECURE certificate verification — all certificates are accepted without validation");
                Arc::new(InsecureVerification)
            }
            MeshSecurityMode::Pinned => {
                info!("Using TOFU (Trust On First Use) certificate pinning");
                Arc::new(TofuVerifier::new(tofu_pins_path))
            }
        };

        let client_crypto = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth();

        let quic_client_config =
            quinn::crypto::rustls::QuicClientConfig::try_from(Arc::new(client_crypto))
                .map_err(|e| MeshError::Transport(format!("QUIC client config: {e}")))?;
        let client_config = ClientConfig::new(Arc::new(quic_client_config));

        Ok((server_config, client_config))
    }

    /// Spawn a task that reads messages from a QUIC connection.
    fn spawn_reader(&self, conn: quinn::Connection, peer_id: NodeId, inbox: Arc<Inbox>) {
        tokio::spawn(async move {
            loop {
                match conn.accept_uni().await {
                    Ok(mut recv) => {
                        let data = match recv.read_to_end(16 * 1024 * 1024).await {
                            Ok(data) => data,
                            Err(e) => {
                                warn!("Failed to read from peer {peer_id}: {e}");
                                break;
                            }
                        };
                        match protocol::decode(&data) {
                            Ok(msg) => {
                                if !inbox.deliver(peer_id.clone(), msg) {
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to decode message from {peer_id}: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        debug!("Connection to {peer_id} closed: {e}");
                        break;
                    }
                }
            }
        });
    }
}

#[async_trait]
impl MeshTransport for QuicTransport {
    async fn send(&self, peer: &NodeId, msg: MeshMessage) -> Result<()> {
        let key = Self::node_key(peer);
        let conn = self
            .connections
            .get(&key)
            .ok_or_else(|| MeshError::PeerDisconnected(peer.to_string()))?;

        let data = protocol::encode(&msg)?;
        let mut send = conn
            .open_uni()
            .await
            .map_err(|e| MeshError::Transport(format!("open stream: {e}")))?;
        send.write_all(&data)
            .await
            .map_err(|e| MeshError::Transport(format!("write: {e}")))?;
        send.finish()
            .map_err(|e| MeshError::Transport(format!("finish: {e}")))?;
        Ok(())
    }

    async fn recv(&self) -> Result<(NodeId, MeshMessage)> {
        self.inbox.recv().await
    }

    async fn connect(&self, addr: SocketAddr) -> Result<NodeId> {
        let endpoint = self.endpoint.lock().await;
        let endpoint = endpoint
            .as_ref()
            .ok_or_else(|| MeshError::Transport("Not listening".into()))?;

        info!("Connecting to peer at {addr}");
        let server_name = Self::peer_server_name(addr);
        let conn = endpoint
            .connect(addr, &server_name)
            .map_err(|e| MeshError::Transport(format!("connect: {e}")))?
            .await
            .map_err(|e| MeshError::Transport(format!("handshake: {e}")))?;

        // Exchange Hello messages
        let hello = MeshMessage::Hello {
            node_id: self.local_id.clone(),
            hardware: hivebear_core::profile(),
            protocol_version: PROTOCOL_VERSION,
        };
        let data = protocol::encode(&hello)?;
        let mut send = conn
            .open_uni()
            .await
            .map_err(|e| MeshError::Transport(format!("open stream: {e}")))?;
        send.write_all(&data)
            .await
            .map_err(|e| MeshError::Transport(format!("write: {e}")))?;
        send.finish()
            .map_err(|e| MeshError::Transport(format!("finish: {e}")))?;

        // Read peer's Hello response
        let mut recv = conn
            .accept_uni()
            .await
            .map_err(|e| MeshError::Transport(format!("accept: {e}")))?;
        let resp_data = recv
            .read_to_end(64 * 1024)
            .await
            .map_err(|e| MeshError::Transport(format!("read: {e}")))?;
        let resp = protocol::decode(&resp_data)?;

        let peer_id = match resp {
            MeshMessage::HelloAck {
                node_id, accepted, ..
            } => {
                if !accepted {
                    return Err(MeshError::Transport("Peer rejected connection".into()));
                }
                node_id
            }
            MeshMessage::Hello { node_id, .. } => node_id,
            _ => return Err(MeshError::Protocol("Expected Hello/HelloAck".into())),
        };

        let key = Self::node_key(&peer_id);
        self.connections.insert(key, conn.clone());
        self.spawn_reader(conn, peer_id.clone(), Arc::clone(&self.inbox));

        info!("Connected to peer {peer_id}");
        Ok(peer_id)
    }

    async fn disconnect(&self, peer: &NodeId) -> Result<()> {
        let key = Self::node_key(peer);
        if let Some((_, conn)) = self.connections.remove(&key) {
            conn.close(0u32.into(), b"disconnect");
        }
        Ok(())
    }

    async fn listen(&self, addr: SocketAddr) -> Result<()> {
        let (server_config, client_config) =
            Self::generate_self_signed_config(self.security_mode, self.tofu_pins_path.clone())?;

        let mut endpoint = Endpoint::server(server_config, addr)
            .map_err(|e| MeshError::Transport(format!("bind: {e}")))?;
        endpoint.set_default_client_config(client_config);

        info!("Listening on {addr}");

        let inbox = Arc::clone(&self.inbox);
        let connections = self.connections.clone();
        let local_id = self.local_id.clone();

        // Spawn acceptor task
        let endpoint_clone = endpoint.clone();
        tokio::spawn(async move {
            while let Some(incoming) = endpoint_clone.accept().await {
                let inbox = Arc::clone(&inbox);
                let connections = connections.clone();
                let local_id = local_id.clone();

                tokio::spawn(async move {
                    match incoming.await {
                        Ok(conn) => {
                            debug!("Accepted connection from {}", conn.remote_address());

                            // Read Hello from the peer
                            match conn.accept_uni().await {
                                Ok(mut recv) => {
                                    match recv.read_to_end(64 * 1024).await {
                                        Ok(data) => match protocol::decode(&data) {
                                            Ok(MeshMessage::Hello {
                                                node_id,
                                                protocol_version,
                                                ..
                                            }) => {
                                                if protocol_version != PROTOCOL_VERSION {
                                                    warn!(
                                                        "Protocol version mismatch: {} vs {}",
                                                        protocol_version, PROTOCOL_VERSION
                                                    );
                                                }

                                                // Send HelloAck
                                                let ack = MeshMessage::HelloAck {
                                                    node_id: local_id,
                                                    accepted: true,
                                                };
                                                if let Ok(data) = protocol::encode(&ack) {
                                                    if let Ok(mut send) = conn.open_uni().await {
                                                        let _ = send.write_all(&data).await;
                                                        let _ = send.finish();
                                                    }
                                                }

                                                let key = node_id.0.to_bytes().to_vec();
                                                connections.insert(key, conn.clone());

                                                // Start reading messages
                                                let peer_id = node_id;
                                                tokio::spawn(async move {
                                                    while let Ok(mut recv) = conn.accept_uni().await
                                                    {
                                                        match recv
                                                            .read_to_end(16 * 1024 * 1024)
                                                            .await
                                                        {
                                                            Ok(data) => {
                                                                match protocol::decode(&data) {
                                                                    Ok(msg) => {
                                                                        if !inbox.deliver(
                                                                            peer_id.clone(),
                                                                            msg,
                                                                        ) {
                                                                            break;
                                                                        }
                                                                    }
                                                                    Err(e) => {
                                                                        warn!("Decode error: {e}");
                                                                    }
                                                                }
                                                            }
                                                            Err(e) => {
                                                                debug!("Read error: {e}");
                                                                break;
                                                            }
                                                        }
                                                    }
                                                });
                                            }
                                            _ => {
                                                warn!("Expected Hello message from peer");
                                            }
                                        },
                                        Err(e) => {
                                            warn!("Failed to read Hello: {e}");
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to accept Hello stream: {e}");
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to accept connection: {e}");
                        }
                    }
                });
            }
        });

        *self.endpoint.lock().await = Some(endpoint);
        Ok(())
    }

    fn is_connected(&self, peer: &NodeId) -> bool {
        let key = Self::node_key(peer);
        self.connections.contains_key(&key)
    }

    fn peer_count(&self) -> usize {
        self.connections.len()
    }

    fn subscribe_session(&self, session_id: uuid::Uuid) -> super::inbox::SessionReceiver {
        self.inbox.subscribe(session_id)
    }

    fn unsubscribe_session(&self, session_id: &uuid::Uuid) {
        self.inbox.unsubscribe(session_id);
    }
}

// ---------------------------------------------------------------------------
// Certificate verifiers
// ---------------------------------------------------------------------------

/// Trust-On-First-Use (TOFU) certificate verifier.
///
/// On first connection to a server name, the certificate is accepted and its
/// SHA-256 fingerprint is stored. Subsequent connections to the same server
/// name must present a certificate with a matching fingerprint, otherwise the
/// connection is rejected (possible MITM attack).
#[derive(Debug)]
struct TofuVerifier {
    /// per-peer server name -> SHA-256 fingerprint of the pinned DER cert.
    /// See `QuicTransport::peer_server_name` for why this is per-peer.
    pinned: DashMap<String, Vec<u8>>,
    /// Optional path for persisting pins to disk across restarts.
    storage_path: Option<PathBuf>,
}

impl TofuVerifier {
    fn new(storage_path: Option<PathBuf>) -> Self {
        let pinned = DashMap::new();

        // Load existing pins from disk if a storage path is provided.
        if let Some(ref path) = storage_path {
            if let Some(loaded) = Self::load_pins(path) {
                for (name, fp_hex) in loaded {
                    match hex::decode(&fp_hex) {
                        Ok(fp) => {
                            pinned.insert(name, fp);
                        }
                        Err(e) => {
                            warn!(
                                "TOFU: Skipping pin for '{}': invalid hex fingerprint: {e}",
                                name
                            );
                        }
                    }
                }
                info!(
                    "TOFU: Loaded {} pinned certificate(s) from {}",
                    pinned.len(),
                    path.display()
                );
            }
        }

        Self {
            pinned,
            storage_path,
        }
    }

    /// Compute the SHA-256 fingerprint of a DER-encoded certificate.
    fn fingerprint(cert_der: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(cert_der);
        hasher.finalize().to_vec()
    }

    /// Load pins from a JSON file on disk. Returns `None` if the file doesn't
    /// exist or cannot be parsed (logs a warning in the latter case).
    fn load_pins(path: &PathBuf) -> Option<HashMap<String, String>> {
        match std::fs::read_to_string(path) {
            Ok(contents) => match serde_json::from_str::<HashMap<String, String>>(&contents) {
                Ok(map) => Some(map),
                Err(e) => {
                    warn!(
                        "TOFU: Corrupt pin file at {}, starting fresh: {e}",
                        path.display()
                    );
                    None
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                debug!("TOFU: No existing pin file at {}", path.display());
                None
            }
            Err(e) => {
                warn!(
                    "TOFU: Failed to read pin file at {}, starting fresh: {e}",
                    path.display()
                );
                None
            }
        }
    }

    /// Persist the current pin map to disk as JSON.
    fn save_pins(&self) {
        let Some(ref path) = self.storage_path else {
            return;
        };

        // Build a HashMap<String, String> with hex-encoded fingerprints.
        let map: HashMap<String, String> = self
            .pinned
            .iter()
            .map(|entry| (entry.key().clone(), hex::encode(entry.value())))
            .collect();

        // Ensure the parent directory exists.
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                warn!("TOFU: Failed to create directory {}: {e}", parent.display());
                return;
            }
        }

        match serde_json::to_string_pretty(&map) {
            Ok(json) => {
                if let Err(e) = std::fs::write(path, json) {
                    warn!("TOFU: Failed to write pin file to {}: {e}", path.display());
                } else {
                    debug!("TOFU: Persisted {} pin(s) to {}", map.len(), path.display());
                }
            }
            Err(e) => {
                warn!("TOFU: Failed to serialize pins: {e}");
            }
        }
    }
}

impl rustls::client::danger::ServerCertVerifier for TofuVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        let name = server_name.to_str().to_string();
        let fp = Self::fingerprint(end_entity.as_ref());

        if let Some(existing) = self.pinned.get(&name) {
            if *existing != fp {
                let expected_hex = hex::encode(existing.value());
                let got_hex = hex::encode(&fp);
                warn!(
                    "TOFU: Certificate fingerprint mismatch for '{name}'. \
                     Expected {expected_hex}, got {got_hex}. Possible MITM attack!"
                );
                return Err(rustls::Error::General(format!(
                    "TOFU certificate mismatch for '{name}': pinned fingerprint does not match"
                )));
            }
            debug!("TOFU: Certificate for '{name}' matches pinned fingerprint");
        } else {
            let fp_hex = hex::encode(&fp);
            info!("TOFU: Pinning certificate for '{name}' (SHA-256: {fp_hex})");
            self.pinned.insert(name, fp);
            self.save_pins();
        }

        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &rustls::crypto::CryptoProvider::get_default()
                .expect("no default CryptoProvider installed")
                .signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &rustls::crypto::CryptoProvider::get_default()
                .expect("no default CryptoProvider installed")
                .signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

/// Insecure certificate verifier that accepts ANY certificate without validation.
///
/// WARNING: This is vulnerable to man-in-the-middle attacks. Only use for
/// development and testing. Never use in production.
///
/// Only available with the `insecure-dev` feature flag.
#[cfg(feature = "insecure-dev")]
#[derive(Debug)]
struct InsecureVerification;

#[cfg(feature = "insecure-dev")]
impl rustls::client::danger::ServerCertVerifier for InsecureVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::NodeIdentity;

    /// Grab a port the OS just told us is free. Racy in principle, fine here.
    fn free_addr() -> SocketAddr {
        let sock = std::net::UdpSocket::bind("127.0.0.1:0").expect("bind probe socket");
        let addr = sock.local_addr().expect("probe local_addr");
        drop(sock);
        addr
    }

    fn transport() -> QuicTransport {
        QuicTransport::new(
            NodeIdentity::generate().node_id,
            MeshSecurityMode::default(),
            None,
        )
    }

    #[test]
    fn peer_server_name_is_unique_and_dns_safe() {
        let a = QuicTransport::peer_server_name("127.0.0.1:7878".parse().unwrap());
        let b = QuicTransport::peer_server_name("127.0.0.1:7879".parse().unwrap());
        let c = QuicTransport::peer_server_name("10.0.0.5:7878".parse().unwrap());

        assert_ne!(a, b, "different ports must pin separately");
        assert_ne!(a, c, "different hosts must pin separately");
        assert!(a.ends_with(".hivebear-mesh"));
        assert!(
            a.chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '.'),
            "server name must stay DNS-safe, got {a}"
        );
    }

    /// The regression this guards: every connection used to be dialled with
    /// the fixed name "hivebear-mesh", and TOFU pins are keyed on that name.
    /// The first peer's self-signed certificate got pinned under it, so the
    /// second peer — presenting its own, different certificate — was rejected
    /// as "possible MITM". One peer per process is not a mesh.
    #[tokio::test]
    async fn connects_to_two_distinct_peers() {
        let addr_a = free_addr();
        let addr_b = free_addr();

        let server_a = transport();
        let server_b = transport();
        server_a.listen(addr_a).await.expect("server A listen");
        server_b.listen(addr_b).await.expect("server B listen");

        // The client must listen too: `connect` requires an endpoint.
        let client = transport();
        client.listen(free_addr()).await.expect("client listen");

        let peer_a = client
            .connect(addr_a)
            .await
            .expect("first peer should connect");
        let peer_b = client
            .connect(addr_b)
            .await
            .expect("second peer should connect — this is the TOFU regression");

        assert_ne!(
            peer_a.0.to_bytes(),
            peer_b.0.to_bytes(),
            "the two servers should report distinct node ids"
        );
        assert_eq!(
            client.peer_count(),
            2,
            "client should hold both connections"
        );
    }
}
