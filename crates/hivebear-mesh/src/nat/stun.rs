use std::net::SocketAddr;

use crate::error::{MeshError, Result};

/// Discover our external (NAT-mapped) address using a STUN server.
///
/// This sends a STUN Binding Request over UDP and parses the
/// XOR-MAPPED-ADDRESS from the response.
pub async fn discover_external_addr(stun_server: &str) -> Result<SocketAddr> {
    let server_addr: SocketAddr = resolve_stun_addr(stun_server).await?;

    // Bind a temporary UDP socket
    let socket = tokio::net::UdpSocket::bind("0.0.0.0:0")
        .await
        .map_err(|e| MeshError::NatTraversal(format!("Failed to bind UDP socket: {e}")))?;

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

    // Read response with timeout
    let mut buf = [0u8; 512];
    let n = tokio::time::timeout(std::time::Duration::from_secs(5), socket.recv(&mut buf))
        .await
        .map_err(|_| MeshError::NatTraversal("STUN request timed out".into()))?
        .map_err(|e| MeshError::NatTraversal(format!("Failed to read STUN response: {e}")))?;

    if n < 20 {
        return Err(MeshError::NatTraversal("STUN response too short".into()));
    }

    // Verify magic cookie
    let cookie = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
    if cookie != 0x2112A442 {
        return Err(MeshError::NatTraversal("Invalid STUN magic cookie".into()));
    }

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
