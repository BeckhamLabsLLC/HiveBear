//! A `quinn::AsyncUdpSocket` that carries QUIC over a TURN allocation.
//!
//! [`super::turn`] negotiates the allocation; this is what actually moves
//! packets through it. Outbound datagrams addressed to a peer we have a
//! bound channel for are wrapped in a 4-byte ChannelData header and sent to
//! the TURN server; inbound ChannelData is unwrapped and re-attributed to
//! the peer, so quinn sees an ordinary datagram from an ordinary address and
//! needs to know nothing about relaying.
//!
//! Peers we can reach directly keep going direct — only relayed peers pay
//! the indirection.

use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use dashmap::DashMap;
use quinn::udp::{RecvMeta, Transmit};
use quinn::{AsyncUdpSocket, UdpPoller};
use tracing::{debug, trace};

use super::turn::{decode_channel_data, encode_channel_data, CHANNEL_MIN};

/// Wraps the socket quinn would otherwise use directly.
#[derive(Debug)]
pub struct TurnSocket {
    inner: Arc<dyn AsyncUdpSocket>,
    /// Where relayed traffic is sent.
    server: SocketAddr,
    /// The address peers dial to reach us.
    relayed_addr: SocketAddr,
    peer_to_channel: DashMap<SocketAddr, u16>,
    channel_to_peer: DashMap<u16, SocketAddr>,
    next_channel: std::sync::atomic::AtomicU16,
}

impl TurnSocket {
    pub fn new(
        inner: Arc<dyn AsyncUdpSocket>,
        server: SocketAddr,
        relayed_addr: SocketAddr,
    ) -> Self {
        Self {
            inner,
            server,
            relayed_addr,
            peer_to_channel: DashMap::new(),
            channel_to_peer: DashMap::new(),
            next_channel: std::sync::atomic::AtomicU16::new(CHANNEL_MIN),
        }
    }

    /// Allocate the next channel number for `peer`.
    ///
    /// The caller is responsible for completing the ChannelBind exchange
    /// with the server; until it does, the server will drop anything sent on
    /// this channel. Returns the existing number if one is already assigned.
    pub fn assign_channel(&self, peer: SocketAddr) -> u16 {
        if let Some(existing) = self.peer_to_channel.get(&peer) {
            return *existing;
        }
        let channel = self
            .next_channel
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.peer_to_channel.insert(peer, channel);
        self.channel_to_peer.insert(channel, peer);
        debug!("Channel {channel:#x} assigned to relayed peer {peer}");
        channel
    }

    /// Stop relaying to `peer`.
    pub fn forget(&self, peer: &SocketAddr) {
        if let Some((_, channel)) = self.peer_to_channel.remove(peer) {
            self.channel_to_peer.remove(&channel);
        }
    }

    /// Whether `peer` is reached through the relay.
    pub fn is_relayed(&self, peer: &SocketAddr) -> bool {
        self.peer_to_channel.contains_key(peer)
    }

    /// The address to advertise to other peers.
    pub fn relayed_addr(&self) -> SocketAddr {
        self.relayed_addr
    }
}

impl AsyncUdpSocket for TurnSocket {
    fn create_io_poller(self: Arc<Self>) -> Pin<Box<dyn UdpPoller>> {
        // Writability is a property of the underlying socket either way.
        Arc::clone(&self.inner).create_io_poller()
    }

    fn try_send(&self, transmit: &Transmit) -> io::Result<()> {
        let Some(channel) = self.peer_to_channel.get(&transmit.destination).map(|c| *c) else {
            // Not a relayed peer — send it straight out.
            return self.inner.try_send(transmit);
        };

        let framed = encode_channel_data(channel, transmit.contents);
        trace!(
            "Relaying {} bytes to {} via channel {channel:#x}",
            transmit.contents.len(),
            transmit.destination
        );

        self.inner.try_send(&Transmit {
            destination: self.server,
            ecn: transmit.ecn,
            contents: &framed,
            // Never segment a relayed transmit: each ChannelData header
            // describes exactly one datagram, so a GSO batch would be
            // unwrappable at the far end.
            segment_size: None,
            src_ip: transmit.src_ip,
        })
    }

    fn poll_recv(
        &self,
        cx: &mut Context,
        bufs: &mut [io::IoSliceMut<'_>],
        meta: &mut [RecvMeta],
    ) -> Poll<io::Result<usize>> {
        let count = match self.inner.poll_recv(cx, bufs, meta) {
            Poll::Ready(Ok(n)) => n,
            other => return other,
        };

        for i in 0..count {
            if meta[i].addr != self.server {
                continue;
            }
            let len = meta[i].len;
            let Some((channel, payload)) = decode_channel_data(&bufs[i][..len]) else {
                // A STUN message from the server (a Refresh reply, say).
                // Leave it alone; the TURN client reads those separately.
                continue;
            };
            let Some(peer) = self.channel_to_peer.get(&channel).map(|p| *p) else {
                debug!("Dropping ChannelData for unknown channel {channel:#x}");
                meta[i].len = 0;
                continue;
            };

            // Strip the 4-byte header in place and re-attribute the datagram
            // to the peer, so quinn sees it as having come from them.
            let payload_len = payload.len();
            bufs[i].copy_within(4..4 + payload_len, 0);
            meta[i].len = payload_len;
            meta[i].stride = payload_len;
            meta[i].addr = peer;
        }

        Poll::Ready(Ok(count))
    }

    fn local_addr(&self) -> io::Result<SocketAddr> {
        // The relayed address, not the local one: this is what quinn should
        // believe it is reachable at, and what peers will dial.
        Ok(self.relayed_addr)
    }

    fn max_transmit_segments(&self) -> usize {
        // One ChannelData header per datagram, so no GSO batching.
        1
    }

    fn max_receive_segments(&self) -> usize {
        // Likewise no GRO: coalesced datagrams cannot be unwrapped
        // individually from a single buffer.
        1
    }

    fn may_fragment(&self) -> bool {
        self.inner.may_fragment()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Records what was sent and replays canned inbound datagrams.
    #[derive(Debug, Default)]
    struct FakeSocket {
        sent: Mutex<Vec<(SocketAddr, Vec<u8>)>>,
        inbound: Mutex<Vec<(SocketAddr, Vec<u8>)>>,
    }

    #[derive(Debug)]
    struct NullPoller;
    impl UdpPoller for NullPoller {
        fn poll_writable(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    impl AsyncUdpSocket for FakeSocket {
        fn create_io_poller(self: Arc<Self>) -> Pin<Box<dyn UdpPoller>> {
            Box::pin(NullPoller)
        }
        fn try_send(&self, transmit: &Transmit) -> io::Result<()> {
            self.sent
                .lock()
                .unwrap()
                .push((transmit.destination, transmit.contents.to_vec()));
            Ok(())
        }
        fn poll_recv(
            &self,
            _cx: &mut Context,
            bufs: &mut [io::IoSliceMut<'_>],
            meta: &mut [RecvMeta],
        ) -> Poll<io::Result<usize>> {
            let mut queued = self.inbound.lock().unwrap();
            if queued.is_empty() {
                return Poll::Pending;
            }
            let (from, data) = queued.remove(0);
            bufs[0][..data.len()].copy_from_slice(&data);
            meta[0] = RecvMeta {
                addr: from,
                len: data.len(),
                stride: data.len(),
                ecn: None,
                dst_ip: None,
            };
            Poll::Ready(Ok(1))
        }
        fn local_addr(&self) -> io::Result<SocketAddr> {
            Ok("127.0.0.1:1".parse().unwrap())
        }
    }

    fn server() -> SocketAddr {
        "203.0.113.1:3478".parse().unwrap()
    }
    fn relayed() -> SocketAddr {
        "203.0.113.1:50000".parse().unwrap()
    }
    fn peer() -> SocketAddr {
        "198.51.100.7:7878".parse().unwrap()
    }
    fn direct() -> SocketAddr {
        "192.0.2.5:7878".parse().unwrap()
    }

    #[test]
    fn relayed_peers_go_through_the_server_and_direct_peers_do_not() {
        let inner = Arc::new(FakeSocket::default());
        let sock = TurnSocket::new(inner.clone(), server(), relayed());
        let channel = sock.assign_channel(peer());

        sock.try_send(&Transmit {
            destination: peer(),
            ecn: None,
            contents: b"to-relayed",
            segment_size: None,
            src_ip: None,
        })
        .unwrap();

        sock.try_send(&Transmit {
            destination: direct(),
            ecn: None,
            contents: b"to-direct",
            segment_size: None,
            src_ip: None,
        })
        .unwrap();

        let sent = inner.sent.lock().unwrap();
        assert_eq!(sent.len(), 2);

        // Relayed: addressed to the TURN server, ChannelData-wrapped.
        assert_eq!(sent[0].0, server());
        assert_eq!(
            decode_channel_data(&sent[0].1),
            Some((channel, &b"to-relayed"[..]))
        );

        // Direct: untouched.
        assert_eq!(sent[1].0, direct());
        assert_eq!(sent[1].1, b"to-direct");
    }

    #[test]
    fn inbound_channel_data_is_unwrapped_and_attributed_to_the_peer() {
        let inner = Arc::new(FakeSocket::default());
        let sock = TurnSocket::new(inner.clone(), server(), relayed());
        let channel = sock.assign_channel(peer());

        inner
            .inbound
            .lock()
            .unwrap()
            .push((server(), encode_channel_data(channel, b"from-peer")));

        let mut storage = [0u8; 256];
        let mut bufs = [io::IoSliceMut::new(&mut storage)];
        let mut meta = [RecvMeta::default()];
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);

        let n = match sock.poll_recv(&mut cx, &mut bufs, &mut meta) {
            Poll::Ready(Ok(n)) => n,
            other => panic!("expected a datagram, got {other:?}"),
        };

        assert_eq!(n, 1);
        assert_eq!(
            meta[0].addr,
            peer(),
            "quinn must see the peer's address, not the relay's"
        );
        assert_eq!(&bufs[0][..meta[0].len], b"from-peer");
    }

    #[test]
    fn direct_inbound_traffic_is_left_alone() {
        let inner = Arc::new(FakeSocket::default());
        let sock = TurnSocket::new(inner.clone(), server(), relayed());

        inner
            .inbound
            .lock()
            .unwrap()
            .push((direct(), b"plain datagram".to_vec()));

        let mut storage = [0u8; 256];
        let mut bufs = [io::IoSliceMut::new(&mut storage)];
        let mut meta = [RecvMeta::default()];
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);

        assert!(matches!(
            sock.poll_recv(&mut cx, &mut bufs, &mut meta),
            Poll::Ready(Ok(1))
        ));
        assert_eq!(meta[0].addr, direct());
        assert_eq!(&bufs[0][..meta[0].len], b"plain datagram");
    }

    #[test]
    fn local_addr_is_the_relayed_address() {
        let sock = TurnSocket::new(Arc::new(FakeSocket::default()), server(), relayed());
        assert_eq!(
            sock.local_addr().unwrap(),
            relayed(),
            "peers dial the relayed address, so that is what quinn must advertise"
        );
    }

    #[test]
    fn channels_are_stable_and_removable() {
        let sock = TurnSocket::new(Arc::new(FakeSocket::default()), server(), relayed());
        let first = sock.assign_channel(peer());
        assert_eq!(sock.assign_channel(peer()), first, "must be idempotent");
        assert!(first >= CHANNEL_MIN);

        let other = sock.assign_channel(direct());
        assert_ne!(other, first);

        assert!(sock.is_relayed(&peer()));
        sock.forget(&peer());
        assert!(!sock.is_relayed(&peer()));
    }
}
