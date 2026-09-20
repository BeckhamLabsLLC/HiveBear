use std::net::SocketAddr;

use crate::error::{MeshError, Result};

/// Discover the external (NAT-mapped) address of a *temporary* socket.
///
/// Almost never what you want: a NAT maps each local port separately, so the
/// address this returns belongs to the throwaway socket, not to whatever port
/// your peers will actually dial. Use [`discover_external_addr_for`] with the
/// socket you are going to listen on.
///
/// Kept for callers that only need a coarse "what does the outside world see
/// as my IP" answer, where the port does not matter.
pub async fn discover_external_addr(stun_server: &str) -> Result<SocketAddr> {
    let socket = tokio::net::UdpSocket::bind("0.0.0.0:0")
        .await
        .map_err(|e| MeshError::NatTraversal(format!("Failed to bind UDP socket: {e}")))?;
    discover_external_addr_for(&socket, stun_server).await
}

/// Discover the external (NAT-mapped) address *of `socket`*.
///
/// The mapping a NAT creates is per source port, so STUN has to run on the
/// very socket peers will connect to. This used to bind its own ephemeral
/// socket and advertise that mapping as the node's `external_addr`, which is
/// only correct on an endpoint-independent (full-cone) NAT — on anything
/// stricter the advertised port was one nothing was listening on, so
/// hole punching could not work.
pub async fn discover_external_addr_for(
    socket: &tokio::net::UdpSocket,
    stun_server: &str,
) -> Result<SocketAddr> {
    let server_addr: SocketAddr = resolve_stun_addr(stun_server).await?;

    // Build a minimal STUN Binding Request
    // RFC 5389: type=0x0001, length=0, magic cookie=0x2112A442, + 12 byte txn id
    let txn_id: [u8; 12] = rand::random();
    let mut request = Vec::with_capacity(20);
    request.extend_from_slice(&0x0001u16.to_be_bytes()); // Message type: Binding Request
    request.extend_from_slice(&0x0000u16.to_be_bytes()); // Message length: 0
    request.extend_from_slice(&0x2112A442u32.to_be_bytes()); // Magic cookie
    request.extend_from_slice(&txn_id); // Transaction ID

    socket
        .send_to(&request, server_addr)
        .await
        .map_err(|e| MeshError::NatTraversal(format!("Failed to send STUN request: {e}")))?;

    // Read the response. Because this may be the live QUIC socket, other
    // datagrams can arrive here too — keep reading until one matches our
    // transaction id rather than failing on the first unrelated packet.
    let mut buf = [0u8; 512];
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    let n = loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(MeshError::NatTraversal("STUN request timed out".into()));
        }
        let (n, from) = tokio::time::timeout(remaining, socket.recv_from(&mut buf))
            .await
            .map_err(|_| MeshError::NatTraversal("STUN request timed out".into()))?
            .map_err(|e| MeshError::NatTraversal(format!("Failed to read STUN response: {e}")))?;

        if from != server_addr || n < 20 {
            continue;
        }
        let cookie = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
        if cookie != 0x2112A442 || buf[8..20] != txn_id {
            continue;
        }
        break n;
    };

    // Parse attributes looking for XOR-MAPPED-ADDRESS (0x0020) or MAPPED-ADDRESS (0x0001)
    let msg_len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
    let attrs_end = std::cmp::min(20 + msg_len, n);
    let mut pos = 20;

    while pos + 4 <= attrs_end {
        let attr_type = u16::from_be_bytes([buf[pos], buf[pos + 1]]);
        let attr_len = u16::from_be_bytes([buf[pos + 2], buf[pos + 3]]) as usize;
        let attr_start = pos + 4;

        if attr_type == 0x0020 && attr_len >= 8 {
            // XOR-MAPPED-ADDRESS
            let family = buf[attr_start + 1];
            if family == 0x01 {
                // IPv4
                let xor_port =
                    u16::from_be_bytes([buf[attr_start + 2], buf[attr_start + 3]]) ^ 0x2112;
                let xor_ip = u32::from_be_bytes([
                    buf[attr_start + 4],
                    buf[attr_start + 5],
                    buf[attr_start + 6],
                    buf[attr_start + 7],
                ]) ^ 0x2112A442;
                let ip = std::net::Ipv4Addr::from(xor_ip);
                return Ok(SocketAddr::new(std::net::IpAddr::V4(ip), xor_port));
            }
        } else if attr_type == 0x0001 && attr_len >= 8 {
            // MAPPED-ADDRESS (fallback)
            let family = buf[attr_start + 1];
            if family == 0x01 {
                let port = u16::from_be_bytes([buf[attr_start + 2], buf[attr_start + 3]]);
                let ip = std::net::Ipv4Addr::new(
                    buf[attr_start + 4],
                    buf[attr_start + 5],
                    buf[attr_start + 6],
                    buf[attr_start + 7],
                );
                return Ok(SocketAddr::new(std::net::IpAddr::V4(ip), port));
            }
        }

        // Move to next attribute (padded to 4 bytes)
        pos = attr_start + ((attr_len + 3) & !3);
    }

    Err(MeshError::NatTraversal(
        "No MAPPED-ADDRESS found in STUN response".into(),
    ))
}

/// Resolve a STUN server address (host:port string) to a SocketAddr.
async fn resolve_stun_addr(server: &str) -> Result<SocketAddr> {
    // Try parsing as SocketAddr directly
    if let Ok(addr) = server.parse::<SocketAddr>() {
        return Ok(addr);
    }

    // DNS resolution
    let addrs: Vec<SocketAddr> = tokio::net::lookup_host(server)
        .await
        .map_err(|e| {
            MeshError::NatTraversal(format!("Failed to resolve STUN server '{server}': {e}"))
        })?
        .collect();

    addrs.into_iter().next().ok_or_else(|| {
        MeshError::NatTraversal(format!("No addresses found for STUN server '{server}'"))
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_stun_request_format() {
        // Verify the STUN binding request is 20 bytes
        let txn_id: [u8; 12] = [0; 12];
        let mut request = Vec::with_capacity(20);
        request.extend_from_slice(&0x0001u16.to_be_bytes());
        request.extend_from_slice(&0x0000u16.to_be_bytes());
        request.extend_from_slice(&0x2112A442u32.to_be_bytes());
        request.extend_from_slice(&txn_id);
        assert_eq!(request.len(), 20);
        // Magic cookie at offset 4
        assert_eq!(&request[4..8], &[0x21, 0x12, 0xA4, 0x42]);
    }
}

#[cfg(test)]
mod socket_tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    /// Minimal STUN server that reflects the source address it saw, which is
    /// exactly what a real one does. Returns its own address.
    async fn spawn_reflector() -> SocketAddr {
        let sock = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let addr = sock.local_addr().unwrap();

        tokio::spawn(async move {
            let mut buf = [0u8; 512];
            loop {
                let Ok((n, from)) = sock.recv_from(&mut buf).await else {
                    return;
                };
                if n < 20 {
                    continue;
                }
                let txn = &buf[8..20];

                let SocketAddr::V4(v4) = from else { continue };
                let xor_port = v4.port() ^ 0x2112;
                let xor_ip = u32::from(*v4.ip()) ^ 0x2112_A442;

                let mut resp = Vec::with_capacity(32);
                resp.extend_from_slice(&0x0101u16.to_be_bytes()); // Binding Success
                resp.extend_from_slice(&12u16.to_be_bytes()); // attribute bytes
                resp.extend_from_slice(&0x2112_A442u32.to_be_bytes());
                resp.extend_from_slice(txn);
                resp.extend_from_slice(&0x0020u16.to_be_bytes()); // XOR-MAPPED-ADDRESS
                resp.extend_from_slice(&8u16.to_be_bytes());
                resp.push(0); // reserved
                resp.push(0x01); // IPv4
                resp.extend_from_slice(&xor_port.to_be_bytes());
                resp.extend_from_slice(&xor_ip.to_be_bytes());

                let _ = sock.send_to(&resp, from).await;
            }
        });

        addr
    }

    /// The regression: STUN bound its own ephemeral socket, so the mapping it
    /// reported belonged to that socket rather than to the port peers would
    /// dial. On any NAT stricter than full-cone the advertised port had
    /// nothing listening on it.
    #[tokio::test]
    async fn reports_the_mapping_of_the_socket_it_was_given() {
        let server = spawn_reflector().await;

        let mesh_socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let mesh_port = mesh_socket.local_addr().unwrap().port();

        let discovered = discover_external_addr_for(&mesh_socket, &server.to_string())
            .await
            .expect("STUN should succeed against the local reflector");

        assert_eq!(
            discovered.port(),
            mesh_port,
            "must report the port of the socket peers will dial, not a throwaway one"
        );
        assert_eq!(discovered.ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
    }

    #[tokio::test]
    async fn ignores_unrelated_datagrams_on_a_shared_socket() {
        let server = spawn_reflector().await;

        let mesh_socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let mesh_port = mesh_socket.local_addr().unwrap().port();

        // Something else writes to the mesh socket mid-handshake — realistic
        // once STUN runs on the live QUIC socket.
        let noise = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let target = mesh_socket.local_addr().unwrap();
        tokio::spawn(async move {
            for _ in 0..5 {
                let _ = noise.send_to(b"not a stun response", target).await;
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        });

        let discovered = discover_external_addr_for(&mesh_socket, &server.to_string())
            .await
            .expect("unrelated traffic must not break the handshake");
        assert_eq!(discovered.port(), mesh_port);
    }

    #[tokio::test]
    async fn times_out_against_a_silent_server() {
        // Nothing is listening here.
        let dead = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let dead_addr = dead.local_addr().unwrap();
        drop(dead);

        let sock = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(8),
            discover_external_addr_for(&sock, &dead_addr.to_string()),
        )
        .await
        .expect("should give up on its own, not hang");
        assert!(result.is_err());
    }
}
