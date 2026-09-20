//! TURN client (RFC 5766) for peers behind symmetric NATs.
//!
//! Hole punching fails on a NAT that picks a different external port per
//! destination, because there is no single address for the far side to punch
//! toward. Those peers have to have their traffic relayed.
//!
//! What lives here is the control plane: authenticate, allocate a relayed
//! address, keep it alive, and bind a channel per peer. Carrying QUIC over
//! the allocation is [`super::turn_socket`].
//!
//! The previous `nat::relay` was not TURN at all — it POSTed JSON to port
//! 3478 and did no allocation, permission or channel-bind exchange, so no
//! standard server would have answered it.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use hmac::{Hmac, Mac};
use sha1::Sha1;
use tracing::{debug, info};

use crate::error::{MeshError, Result};

const MAGIC_COOKIE: u32 = 0x2112_A442;

// Message classes, encoded into the method field (RFC 5389 §6).
const CLASS_REQUEST: u16 = 0x0000;
const CLASS_SUCCESS: u16 = 0x0100;
const CLASS_ERROR: u16 = 0x0110;

// Methods.
const METHOD_ALLOCATE: u16 = 0x0003;
const METHOD_REFRESH: u16 = 0x0004;
const METHOD_CHANNEL_BIND: u16 = 0x0009;

// Attributes.
const ATTR_USERNAME: u16 = 0x0006;
const ATTR_MESSAGE_INTEGRITY: u16 = 0x0008;
const ATTR_ERROR_CODE: u16 = 0x0009;
const ATTR_CHANNEL_NUMBER: u16 = 0x000C;
const ATTR_LIFETIME: u16 = 0x000D;
const ATTR_XOR_PEER_ADDRESS: u16 = 0x0012;
const ATTR_REALM: u16 = 0x0014;
const ATTR_NONCE: u16 = 0x0015;
const ATTR_XOR_RELAYED_ADDRESS: u16 = 0x0016;
const ATTR_REQUESTED_TRANSPORT: u16 = 0x0019;

/// Channel numbers are restricted to this range (RFC 5766 §11).
pub const CHANNEL_MIN: u16 = 0x4000;
pub const CHANNEL_MAX: u16 = 0x7FFF;

/// Requested transport value for UDP.
const TRANSPORT_UDP: u8 = 17;

/// Credentials for a TURN allocation.
///
/// Issued by the coordinator, which holds the server's shared secret and
/// derives time-limited pairs from it. The client deliberately never sees
/// that secret: anything holding it can mint unlimited credentials.
#[derive(Debug, Clone)]
pub struct TurnCredentials {
    pub username: String,
    pub password: String,
}

/// A successful allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allocation {
    /// Address the TURN server relays to us. This is what peers dial.
    pub relayed_addr: SocketAddr,
    /// Seconds until the allocation expires unless refreshed.
    pub lifetime_secs: u32,
}

/// A parsed STUN/TURN message.
#[derive(Debug, Clone)]
pub struct Message {
    pub method: u16,
    pub class: u16,
    pub txn_id: [u8; 12],
    pub attributes: Vec<(u16, Vec<u8>)>,
}

impl Message {
    pub fn attr(&self, kind: u16) -> Option<&[u8]> {
        self.attributes
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, v)| v.as_slice())
    }

    /// Error code, if this is an error response.
    pub fn error_code(&self) -> Option<u16> {
        let v = self.attr(ATTR_ERROR_CODE)?;
        if v.len() < 4 {
            return None;
        }
        // Bytes 2-3 are class and number (RFC 5389 §15.6).
        Some(u16::from(v[2] & 0x07) * 100 + u16::from(v[3]))
    }

    pub fn attr_str(&self, kind: u16) -> Option<String> {
        self.attr(kind)
            .map(|v| String::from_utf8_lossy(v).into_owned())
    }
}

/// Build a STUN/TURN message, optionally signed with MESSAGE-INTEGRITY.
pub struct MessageBuilder {
    method: u16,
    class: u16,
    txn_id: [u8; 12],
    attributes: Vec<(u16, Vec<u8>)>,
}

impl MessageBuilder {
    pub fn new(method: u16, class: u16) -> Self {
        Self {
            method,
            class,
            txn_id: rand::random(),
            attributes: Vec::new(),
        }
    }

    pub fn with_txn_id(mut self, txn_id: [u8; 12]) -> Self {
        self.txn_id = txn_id;
        self
    }

    pub fn attr(mut self, kind: u16, value: Vec<u8>) -> Self {
        self.attributes.push((kind, value));
        self
    }

    pub fn txn_id(&self) -> [u8; 12] {
        self.txn_id
    }

    /// Serialize, appending MESSAGE-INTEGRITY when credentials are supplied.
    ///
    /// The integrity key is `MD5(username:realm:password)` per RFC 5389
    /// §15.4, and the HMAC covers the header with its length field already
    /// counting the attribute that is about to be appended — a detail that
    /// silently breaks authentication if missed.
    pub fn build(&self, creds: Option<(&TurnCredentials, &str)>) -> Vec<u8> {
        let mut body = Vec::new();
        for (kind, value) in &self.attributes {
            push_attr(&mut body, *kind, value);
        }

        let Some((creds, realm)) = creds else {
            return self.frame(&body);
        };

        // Length must already include the 24-byte MESSAGE-INTEGRITY TLV.
        let mut framed = self.frame_with_extra(&body, 24);
        let key = long_term_key(&creds.username, realm, &creds.password);
        let mut mac = Hmac::<Sha1>::new_from_slice(&key).expect("HMAC accepts any key length");
        mac.update(&framed);
        let digest = mac.finalize().into_bytes();

        push_attr(&mut body, ATTR_MESSAGE_INTEGRITY, &digest);
        framed = self.frame(&body);
        framed
    }

    fn frame(&self, body: &[u8]) -> Vec<u8> {
        self.frame_with_extra(body, 0)
    }

    fn frame_with_extra(&self, body: &[u8], extra_len: u16) -> Vec<u8> {
        let mut out = Vec::with_capacity(20 + body.len());
        out.extend_from_slice(&(self.method | self.class).to_be_bytes());
        out.extend_from_slice(&(body.len() as u16 + extra_len).to_be_bytes());
        out.extend_from_slice(&MAGIC_COOKIE.to_be_bytes());
        out.extend_from_slice(&self.txn_id);
        out.extend_from_slice(body);
        out
    }
}

fn push_attr(buf: &mut Vec<u8>, kind: u16, value: &[u8]) {
    buf.extend_from_slice(&kind.to_be_bytes());
    buf.extend_from_slice(&(value.len() as u16).to_be_bytes());
    buf.extend_from_slice(value);
    // Attributes are padded to a 4-byte boundary; the padding is not counted
    // in the length.
    while !buf.len().is_multiple_of(4) {
        buf.push(0);
    }
}

/// `MD5(username:realm:password)` — the long-term credential key.
fn long_term_key(username: &str, realm: &str, password: &str) -> Vec<u8> {
    // RFC 5389 fixes MD5 here. It is a protocol constant, not a security
    // choice we get to make; the HMAC around it is what provides integrity.
    md5_bytes(format!("{username}:{realm}:{password}").as_bytes())
}

/// Minimal MD5, needed only for the fixed key derivation above.
fn md5_bytes(data: &[u8]) -> Vec<u8> {
    // Implemented inline to avoid taking a dependency on an md5 crate for
    // one fixed protocol constant.
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64)
        .map(|i| ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32)
        .collect();

    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());

    let (mut a0, mut b0, mut c0, mut d0) = (
        0x6745_2301u32,
        0xefcd_ab89u32,
        0x98ba_dcfeu32,
        0x1032_5476u32,
    );

    for chunk in msg.chunks(64) {
        let m: Vec<u32> = chunk
            .chunks(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f2 = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f2.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut out = Vec::with_capacity(16);
    for v in [a0, b0, c0, d0] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Parse a STUN/TURN message.
pub fn decode(buf: &[u8]) -> Option<Message> {
    if buf.len() < 20 {
        return None;
    }
    let type_field = u16::from_be_bytes([buf[0], buf[1]]);
    // Top two bits must be zero for STUN; anything else is ChannelData.
    if type_field & 0xC000 != 0 {
        return None;
    }
    if u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) != MAGIC_COOKIE {
        return None;
    }

    let class = type_field & 0x0110;
    let method = type_field & !0x0110;
    let mut txn_id = [0u8; 12];
    txn_id.copy_from_slice(&buf[8..20]);

    let msg_len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
    let end = std::cmp::min(20 + msg_len, buf.len());
    let mut pos = 20;
    let mut attributes = Vec::new();

    while pos + 4 <= end {
        let kind = u16::from_be_bytes([buf[pos], buf[pos + 1]]);
        let len = u16::from_be_bytes([buf[pos + 2], buf[pos + 3]]) as usize;
        let start = pos + 4;
        if start + len > end {
            break;
        }
        attributes.push((kind, buf[start..start + len].to_vec()));
        pos = start + len.div_ceil(4) * 4;
    }

    Some(Message {
        method,
        class,
        txn_id,
        attributes,
    })
}

/// Decode an XOR-MAPPED/RELAYED/PEER-ADDRESS attribute value.
pub fn decode_xor_addr(value: &[u8], txn_id: &[u8; 12]) -> Option<SocketAddr> {
    if value.len() < 8 {
        return None;
    }
    let port = u16::from_be_bytes([value[2], value[3]]) ^ (MAGIC_COOKIE >> 16) as u16;
    match value[1] {
        0x01 => {
            let raw = u32::from_be_bytes([value[4], value[5], value[6], value[7]]) ^ MAGIC_COOKIE;
            Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::from(raw)), port))
        }
        0x02 => {
            if value.len() < 20 {
                return None;
            }
            let mut key = [0u8; 16];
            key[..4].copy_from_slice(&MAGIC_COOKIE.to_be_bytes());
            key[4..].copy_from_slice(txn_id);
            let mut octets = [0u8; 16];
            for i in 0..16 {
                octets[i] = value[4 + i] ^ key[i];
            }
            Some(SocketAddr::new(
                IpAddr::V6(std::net::Ipv6Addr::from(octets)),
                port,
            ))
        }
        _ => None,
    }
}

/// Encode a socket address as an XOR-*-ADDRESS attribute value.
pub fn encode_xor_addr(addr: SocketAddr, txn_id: &[u8; 12]) -> Vec<u8> {
    let mut out = vec![0u8, 0, 0, 0];
    let port = addr.port() ^ (MAGIC_COOKIE >> 16) as u16;
    out[2..4].copy_from_slice(&port.to_be_bytes());
    match addr.ip() {
        IpAddr::V4(v4) => {
            out[1] = 0x01;
            let raw = u32::from(v4) ^ MAGIC_COOKIE;
            out.extend_from_slice(&raw.to_be_bytes());
        }
        IpAddr::V6(v6) => {
            out[1] = 0x02;
            let mut key = [0u8; 16];
            key[..4].copy_from_slice(&MAGIC_COOKIE.to_be_bytes());
            key[4..].copy_from_slice(txn_id);
            let octets = v6.octets();
            for i in 0..16 {
                out.push(octets[i] ^ key[i]);
            }
        }
    }
    out
}

/// Wrap a datagram in a ChannelData header (RFC 5766 §11.4).
pub fn encode_channel_data(channel: u16, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&channel.to_be_bytes());
    out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// Read a ChannelData header, returning `(channel, payload)`.
///
/// Returns `None` for anything that is not ChannelData — including STUN
/// messages, whose first two bits are zero.
pub fn decode_channel_data(buf: &[u8]) -> Option<(u16, &[u8])> {
    if buf.len() < 4 {
        return None;
    }
    let channel = u16::from_be_bytes([buf[0], buf[1]]);
    if !(CHANNEL_MIN..=CHANNEL_MAX).contains(&channel) {
        return None;
    }
    let len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
    if buf.len() < 4 + len {
        return None;
    }
    Some((channel, &buf[4..4 + len]))
}

/// Performs the TURN control exchanges over a caller-supplied socket.
pub struct TurnClient {
    server: SocketAddr,
    creds: TurnCredentials,
    realm: Option<String>,
    nonce: Option<Vec<u8>>,
}

impl TurnClient {
    pub fn new(server: SocketAddr, creds: TurnCredentials) -> Self {
        Self {
            server,
            creds,
            realm: None,
            nonce: None,
        }
    }

    pub fn server(&self) -> SocketAddr {
        self.server
    }

    /// Allocate a relayed address.
    ///
    /// TURN always rejects the first Allocate with 401 and supplies the realm
    /// and nonce, so this sends unauthenticated, learns them, and retries
    /// signed. Anything else is a protocol error.
    pub async fn allocate(&mut self, socket: &tokio::net::UdpSocket) -> Result<Allocation> {
        let unsigned = MessageBuilder::new(METHOD_ALLOCATE, CLASS_REQUEST)
            .attr(ATTR_REQUESTED_TRANSPORT, vec![TRANSPORT_UDP, 0, 0, 0]);
        let first = self.exchange(socket, &unsigned, None).await?;

        let response = match first.class {
            CLASS_SUCCESS => first,
            CLASS_ERROR if first.error_code() == Some(401) => {
                self.realm = first.attr_str(ATTR_REALM);
                self.nonce = first.attr(ATTR_NONCE).map(|v| v.to_vec());
                let realm = self.realm.clone().ok_or_else(|| {
                    MeshError::Relay("TURN server sent 401 without a realm".into())
                })?;
                let signed = MessageBuilder::new(METHOD_ALLOCATE, CLASS_REQUEST)
                    .attr(ATTR_REQUESTED_TRANSPORT, vec![TRANSPORT_UDP, 0, 0, 0])
                    .attr(ATTR_USERNAME, self.creds.username.clone().into_bytes())
                    .attr(ATTR_REALM, realm.clone().into_bytes())
                    .attr(ATTR_NONCE, self.nonce.clone().unwrap_or_default());
                self.exchange(socket, &signed, Some(&realm)).await?
            }
            CLASS_ERROR => {
                return Err(MeshError::Relay(format!(
                    "TURN Allocate rejected: {:?}",
                    first.error_code()
                )))
            }
            other => return Err(MeshError::Relay(format!("unexpected class {other:#x}"))),
        };

        if response.class != CLASS_SUCCESS {
            return Err(MeshError::Relay(format!(
                "TURN Allocate failed: {:?}",
                response.error_code()
            )));
        }

        let relayed = response
            .attr(ATTR_XOR_RELAYED_ADDRESS)
            .and_then(|v| decode_xor_addr(v, &response.txn_id))
            .ok_or_else(|| {
                MeshError::Relay("Allocate succeeded without a relayed address".into())
            })?;

        let lifetime_secs = response
            .attr(ATTR_LIFETIME)
            .filter(|v| v.len() >= 4)
            .map(|v| u32::from_be_bytes([v[0], v[1], v[2], v[3]]))
            .unwrap_or(600);

        info!("TURN allocation at {relayed} (lifetime {lifetime_secs}s)");
        Ok(Allocation {
            relayed_addr: relayed,
            lifetime_secs,
        })
    }

    /// Extend the allocation. Without this it expires and every relayed
    /// connection drops.
    pub async fn refresh(&mut self, socket: &tokio::net::UdpSocket, lifetime: u32) -> Result<()> {
        let realm = self
            .realm
            .clone()
            .ok_or_else(|| MeshError::Relay("refresh before allocate".into()))?;
        let msg = MessageBuilder::new(METHOD_REFRESH, CLASS_REQUEST)
            .attr(ATTR_LIFETIME, lifetime.to_be_bytes().to_vec())
            .attr(ATTR_USERNAME, self.creds.username.clone().into_bytes())
            .attr(ATTR_REALM, realm.clone().into_bytes())
            .attr(ATTR_NONCE, self.nonce.clone().unwrap_or_default());
        let resp = self.exchange(socket, &msg, Some(&realm)).await?;
        if resp.class == CLASS_SUCCESS {
            Ok(())
        } else {
            Err(MeshError::Relay(format!(
                "TURN Refresh failed: {:?}",
                resp.error_code()
            )))
        }
    }

    /// Bind `channel` to `peer` so datagrams can use the 4-byte ChannelData
    /// header instead of a full Send indication per packet.
    pub async fn bind_channel(
        &mut self,
        socket: &tokio::net::UdpSocket,
        channel: u16,
        peer: SocketAddr,
    ) -> Result<()> {
        if !(CHANNEL_MIN..=CHANNEL_MAX).contains(&channel) {
            return Err(MeshError::Relay(format!(
                "channel {channel:#x} is outside the permitted range"
            )));
        }
        let realm = self
            .realm
            .clone()
            .ok_or_else(|| MeshError::Relay("bind_channel before allocate".into()))?;

        let builder = MessageBuilder::new(METHOD_CHANNEL_BIND, CLASS_REQUEST);
        let txn = builder.txn_id();
        let msg = builder
            .attr(ATTR_CHANNEL_NUMBER, {
                let mut v = channel.to_be_bytes().to_vec();
                v.extend_from_slice(&[0, 0]);
                v
            })
            .attr(ATTR_XOR_PEER_ADDRESS, encode_xor_addr(peer, &txn))
            .attr(ATTR_USERNAME, self.creds.username.clone().into_bytes())
            .attr(ATTR_REALM, realm.clone().into_bytes())
            .attr(ATTR_NONCE, self.nonce.clone().unwrap_or_default());

        let resp = self.exchange(socket, &msg, Some(&realm)).await?;
        if resp.class == CLASS_SUCCESS {
            debug!("Bound channel {channel:#x} to {peer}");
            Ok(())
        } else {
            Err(MeshError::Relay(format!(
                "TURN ChannelBind failed: {:?}",
                resp.error_code()
            )))
        }
    }

    /// Send one request and wait for the matching response.
    async fn exchange(
        &self,
        socket: &tokio::net::UdpSocket,
        builder: &MessageBuilder,
        realm: Option<&str>,
    ) -> Result<Message> {
        let creds = realm.map(|r| (&self.creds, r));
        let bytes = builder.build(creds);
        let txn = builder.txn_id();

        socket
            .send_to(&bytes, self.server)
            .await
            .map_err(|e| MeshError::Relay(format!("TURN send failed: {e}")))?;

        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let mut buf = [0u8; 2048];
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(MeshError::Relay("TURN request timed out".into()));
            }
            let (n, from) = tokio::time::timeout(remaining, socket.recv_from(&mut buf))
                .await
                .map_err(|_| MeshError::Relay("TURN request timed out".into()))?
                .map_err(|e| MeshError::Relay(format!("TURN recv failed: {e}")))?;

            if from != self.server {
                continue;
            }
            // The socket may also be carrying relayed traffic; only a STUN
            // reply with our transaction id answers this request.
            match decode(&buf[..n]) {
                Some(msg) if msg.txn_id == txn => return Ok(msg),
                _ => continue,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 1321 test vectors — the key derivation is worthless if this is
    /// wrong, and a wrong HMAC key fails authentication with no clue why.
    #[test]
    fn md5_matches_the_published_vectors() {
        let cases = [
            ("", "d41d8cd98f00b204e9800998ecf8427e"),
            ("a", "0cc175b9c0f1b6a831c399e269772661"),
            ("abc", "900150983cd24fb0d6963f7d28e17f72"),
            ("message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (
                "abcdefghijklmnopqrstuvwxyz",
                "c3fcd3d76192e4007dfb496cca67e13b",
            ),
            (
                "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(
                hex::encode(md5_bytes(input.as_bytes())),
                expected,
                "MD5({input:?})"
            );
        }
    }

    #[test]
    fn xor_addresses_round_trip() {
        let txn: [u8; 12] = [1; 12];
        for addr in [
            "203.0.113.9:7878".parse().unwrap(),
            "127.0.0.1:3478".parse().unwrap(),
            "[2001:db8::1]:9000".parse::<SocketAddr>().unwrap(),
        ] {
            let encoded = encode_xor_addr(addr, &txn);
            assert_eq!(decode_xor_addr(&encoded, &txn), Some(addr), "{addr}");
        }
    }

    #[test]
    fn channel_data_round_trips_and_rejects_stun() {
        let framed = encode_channel_data(0x4001, b"hello quic");
        assert_eq!(
            decode_channel_data(&framed),
            Some((0x4001, &b"hello quic"[..]))
        );

        // A STUN message starts with two zero bits, so it must never be
        // mistaken for ChannelData — that is the whole demultiplexing rule
        // on a TURN-carrying socket.
        let stun = MessageBuilder::new(METHOD_ALLOCATE, CLASS_REQUEST).build(None);
        assert_eq!(decode_channel_data(&stun), None);
        assert!(decode(&stun).is_some());

        // Channel numbers outside 0x4000..=0x7FFF are not ChannelData.
        let bogus = encode_channel_data(0x1234, b"x");
        assert_eq!(decode_channel_data(&bogus), None);
    }

    #[test]
    fn message_integrity_length_counts_the_attribute_itself() {
        let creds = TurnCredentials {
            username: "user".into(),
            password: "pass".into(),
        };
        let bytes = MessageBuilder::new(METHOD_ALLOCATE, CLASS_REQUEST)
            .attr(ATTR_REQUESTED_TRANSPORT, vec![TRANSPORT_UDP, 0, 0, 0])
            .build(Some((&creds, "realm")));

        let declared = u16::from_be_bytes([bytes[2], bytes[3]]) as usize;
        assert_eq!(
            declared,
            bytes.len() - 20,
            "the header length must cover every attribute including MESSAGE-INTEGRITY"
        );

        let parsed = decode(&bytes).expect("own message should parse");
        assert!(parsed.attr(ATTR_MESSAGE_INTEGRITY).is_some());
    }

    /// A stand-in for coturn that checks MESSAGE-INTEGRITY exactly as a real
    /// server does, so a mis-signed request fails here too.
    async fn spawn_turn_server(username: &str, password: &str, realm: &str) -> SocketAddr {
        let sock = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let addr = sock.local_addr().unwrap();
        let (username, password, realm) = (
            username.to_string(),
            password.to_string(),
            realm.to_string(),
        );

        tokio::spawn(async move {
            let mut buf = [0u8; 2048];
            loop {
                let Ok((n, from)) = sock.recv_from(&mut buf).await else {
                    return;
                };
                let Some(msg) = decode(&buf[..n]) else {
                    continue;
                };

                let reply = if msg.attr(ATTR_MESSAGE_INTEGRITY).is_none() {
                    // Unauthenticated: demand credentials, as TURN requires.
                    MessageBuilder::new(msg.method, CLASS_ERROR)
                        .with_txn_id(msg.txn_id)
                        .attr(ATTR_ERROR_CODE, vec![0, 0, 4, 1])
                        .attr(ATTR_REALM, realm.clone().into_bytes())
                        .attr(ATTR_NONCE, b"nonce-1".to_vec())
                        .build(None)
                } else if !integrity_ok(&buf[..n], &username, &realm, &password) {
                    MessageBuilder::new(msg.method, CLASS_ERROR)
                        .with_txn_id(msg.txn_id)
                        .attr(ATTR_ERROR_CODE, vec![0, 0, 4, 1])
                        .build(None)
                } else {
                    let builder =
                        MessageBuilder::new(msg.method, CLASS_SUCCESS).with_txn_id(msg.txn_id);
                    match msg.method {
                        METHOD_ALLOCATE => builder
                            .attr(
                                ATTR_XOR_RELAYED_ADDRESS,
                                encode_xor_addr("203.0.113.50:50000".parse().unwrap(), &msg.txn_id),
                            )
                            .attr(ATTR_LIFETIME, 600u32.to_be_bytes().to_vec())
                            .build(None),
                        _ => builder.build(None),
                    }
                };

                let _ = sock.send_to(&reply, from).await;
            }
        });

        addr
    }

    /// Recompute MESSAGE-INTEGRITY over the received bytes, the way a server
    /// must: the HMAC covers everything up to (not including) the attribute,
    /// with the header length adjusted to include it.
    fn integrity_ok(buf: &[u8], username: &str, realm: &str, password: &str) -> bool {
        let Some(msg) = decode(buf) else { return false };
        let Some(received) = msg.attr(ATTR_MESSAGE_INTEGRITY) else {
            return false;
        };
        let Some(pos) = find_integrity_offset(buf) else {
            return false;
        };

        let mut covered = buf[..pos].to_vec();
        let len = (pos + 24 - 20) as u16;
        covered[2..4].copy_from_slice(&len.to_be_bytes());

        let key = long_term_key(username, realm, password);
        let mut mac = Hmac::<Sha1>::new_from_slice(&key).unwrap();
        mac.update(&covered);
        mac.verify_slice(received).is_ok()
    }

    /// Byte offset of the MESSAGE-INTEGRITY TLV header.
    fn find_integrity_offset(buf: &[u8]) -> Option<usize> {
        let msg_len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
        let end = std::cmp::min(20 + msg_len, buf.len());
        let mut pos = 20;
        while pos + 4 <= end {
            let kind = u16::from_be_bytes([buf[pos], buf[pos + 1]]);
            let len = u16::from_be_bytes([buf[pos + 2], buf[pos + 3]]) as usize;
            if kind == ATTR_MESSAGE_INTEGRITY {
                return Some(pos);
            }
            pos = pos + 4 + len.div_ceil(4) * 4;
        }
        None
    }

    #[tokio::test]
    async fn allocates_through_the_401_challenge() {
        let server = spawn_turn_server("alice", "s3cret", "hivebear.com").await;
        let sock = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();

        let mut client = TurnClient::new(
            server,
            TurnCredentials {
                username: "alice".into(),
                password: "s3cret".into(),
            },
        );

        let allocation = client.allocate(&sock).await.expect("allocation");
        assert_eq!(
            allocation.relayed_addr,
            "203.0.113.50:50000".parse::<SocketAddr>().unwrap()
        );
        assert_eq!(allocation.lifetime_secs, 600);

        // Refresh and ChannelBind reuse the learned realm and nonce.
        client.refresh(&sock, 600).await.expect("refresh");
        client
            .bind_channel(&sock, 0x4001, "198.51.100.7:7878".parse().unwrap())
            .await
            .expect("channel bind");
    }

    #[tokio::test]
    async fn the_wrong_password_is_rejected() {
        let server = spawn_turn_server("alice", "s3cret", "hivebear.com").await;
        let sock = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();

        let mut client = TurnClient::new(
            server,
            TurnCredentials {
                username: "alice".into(),
                password: "wrong".into(),
            },
        );

        assert!(
            client.allocate(&sock).await.is_err(),
            "a bad credential must not yield an allocation"
        );
    }

    #[tokio::test]
    async fn a_channel_outside_the_valid_range_is_refused_locally() {
        let server = spawn_turn_server("alice", "s3cret", "hivebear.com").await;
        let sock = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let mut client = TurnClient::new(
            server,
            TurnCredentials {
                username: "alice".into(),
                password: "s3cret".into(),
            },
        );
        client.allocate(&sock).await.unwrap();

        assert!(client
            .bind_channel(&sock, 0x0001, "198.51.100.7:7878".parse().unwrap())
            .await
            .is_err());
    }
}
