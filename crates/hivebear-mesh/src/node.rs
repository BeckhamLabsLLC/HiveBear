use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use tokio::sync::Notify;
use tracing::{debug, info, warn};

use crate::config::MeshTier;
use crate::discovery::PeerDiscovery;
use crate::error::Result;
use crate::identity::NodeIdentity;
use crate::nat;
use crate::peer::{NodeId, PeerInfo, PeerState};
use crate::transport::protocol::MeshMessage;
use crate::transport::MeshTransport;
use crate::trust::ReputationManager;

/// Core lifecycle manager for a mesh node.
///
/// Ties together transport, discovery, peer state, and reputation.
pub struct MeshNode {
    pub local_id: NodeId,
    pub signing_key: ed25519_dalek::SigningKey,
    pub transport: Arc<dyn MeshTransport>,
    pub discovery: Arc<dyn PeerDiscovery>,
    pub peers: DashMap<Vec<u8>, (PeerInfo, PeerState)>,
    pub reputation: tokio::sync::Mutex<ReputationManager>,
    pub tier: MeshTier,
    /// Our STUN-discovered external address (populated on start).
    pub external_addr: tokio::sync::RwLock<Option<SocketAddr>>,
    /// Address we are listening on, once started.
    listen_addr: tokio::sync::RwLock<Option<SocketAddr>>,
    /// The PeerInfo we registered with, kept so registration can be retried.
    local_info: tokio::sync::RwLock<Option<PeerInfo>>,
    /// Reputation below which a peer is not worth connecting to.
    ///
    /// `MeshConfig::min_reputation` is validated on save and displayed in
    /// Settings, but nothing read it — only the hard ban threshold applied,
    /// so the setting did nothing at any value.
    min_reputation: f64,
    /// STUN servers for NAT detection.
    pub stun_servers: Vec<String>,
    /// Relay servers for symmetric NAT fallback.
    pub relay_servers: Vec<String>,
    running: std::sync::atomic::AtomicBool,
    /// Whether the coordination server has acknowledged us. Distinct from
    /// `running`: the local node can be listening happily while the
    /// coordinator is unreachable, and the UI must not conflate the two.
    registered: std::sync::atomic::AtomicBool,
    shutdown: Arc<Notify>,
}

/// How many peer connection attempts to run at once during a discovery sweep.
const MAX_CONCURRENT_CONNECTS: usize = 8;

/// Upper bound on one discovery sweep, so it cannot monopolise the
/// maintenance loop no matter how many peers are unreachable.
const DISCOVERY_SWEEP_BUDGET: Duration = Duration::from_secs(45);

impl MeshNode {
    pub fn new(
        transport: Arc<dyn MeshTransport>,
        discovery: Arc<dyn PeerDiscovery>,
        tier: MeshTier,
        reputation_path: Option<std::path::PathBuf>,
    ) -> Self {
        let (local_id, signing_key) = NodeId::generate();
        Self {
            local_id,
            signing_key,
            transport,
            discovery,
            peers: DashMap::new(),
            reputation: tokio::sync::Mutex::new(ReputationManager::new(reputation_path)),
            tier,
            external_addr: tokio::sync::RwLock::new(None),
            listen_addr: tokio::sync::RwLock::new(None),
            local_info: tokio::sync::RwLock::new(None),
            min_reputation: 0.0,
            stun_servers: vec!["stun.l.google.com:19302".into()],
            relay_servers: vec!["relay.hivebear.com:3478".into()],
            running: std::sync::atomic::AtomicBool::new(false),
            registered: std::sync::atomic::AtomicBool::new(false),
            shutdown: Arc::new(Notify::new()),
        }
    }

    /// Create a node with a pre-existing persistent identity.
    pub fn with_identity(
        identity: NodeIdentity,
        transport: Arc<dyn MeshTransport>,
        discovery: Arc<dyn PeerDiscovery>,
        tier: MeshTier,
        reputation_path: Option<std::path::PathBuf>,
    ) -> Self {
        Self {
            local_id: identity.node_id,
            signing_key: identity.signing_key,
            transport,
            discovery,
            peers: DashMap::new(),
            reputation: tokio::sync::Mutex::new(ReputationManager::new(reputation_path)),
            tier,
            external_addr: tokio::sync::RwLock::new(None),
            listen_addr: tokio::sync::RwLock::new(None),
            local_info: tokio::sync::RwLock::new(None),
            min_reputation: 0.0,
            stun_servers: vec!["stun.l.google.com:19302".into()],
            relay_servers: vec!["relay.hivebear.com:3478".into()],
            running: std::sync::atomic::AtomicBool::new(false),
            registered: std::sync::atomic::AtomicBool::new(false),
            shutdown: Arc::new(Notify::new()),
        }
    }

    /// Start the mesh node: listen for connections and register with discovery.
    ///
    /// Automatically discovers external address via STUN before registering,
    /// so the coordinator knows how other peers can reach us.
    /// Override the NAT helper servers.
    ///
    /// `MeshConfig::stun_servers` and `relay_servers` are parsed, persisted
    /// and shown in Settings, but both constructors hardcoded their own
    /// values and never consulted the config — so changing either setting did
    /// nothing. Callers should pass the configured lists through here.
    /// Refuse peers whose reputation is below `min_reputation`.
    pub fn with_min_reputation(mut self, min_reputation: f64) -> Self {
        self.min_reputation = min_reputation.clamp(0.0, 1.0);
        self
    }

    /// Re-send our registration after losing contact.
    ///
    /// Cheap and idempotent: the coordinator upserts by node id.
    async fn try_reregister(&self) {
        let info = match self.local_info.read().await.clone() {
            Some(i) => i,
            None => return,
        };
        match self.discovery.register(&info).await {
            Ok(()) => {
                info!("Re-registered with the coordination server");
                self.registered
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
            Err(e) => debug!("Re-registration failed, will retry: {e}"),
        }
    }

    /// Record the outcome of a verification challenge against a peer.
    ///
    /// Nothing called into ReputationManager before this, so scores never
    /// moved off the neutral 0.5 and `is_banned` could never become true no
    /// matter how a peer behaved.
    pub async fn record_verification(&self, peer: &NodeId, passed: bool) {
        let mut rep = self.reputation.lock().await;
        rep.record_verification(peer, passed);
        let score = rep.score(peer);
        if passed {
            debug!("Verification passed for {peer}; score now {score:.2}");
        } else {
            warn!("Verification FAILED for {peer}; score now {score:.2}");
        }
        if rep.is_banned(peer) {
            warn!("Peer {peer} is now below the ban threshold and will be skipped");
        }
    }

    pub fn with_nat_servers(
        mut self,
        stun_servers: Vec<String>,
        relay_servers: Vec<String>,
    ) -> Self {
        if !stun_servers.is_empty() {
            self.stun_servers = stun_servers;
        }
        if !relay_servers.is_empty() {
            self.relay_servers = relay_servers;
        }
        self
    }

    pub async fn start(&self, listen_addr: SocketAddr, mut local_info: PeerInfo) -> Result<()> {
        info!("Starting mesh node {} on {}", self.local_id, listen_addr);

        // Listen first: the transport probes STUN on the socket it is about
        // to serve on. Doing it here, on a throwaway socket, reported a NAT
        // mapping for a port nothing was listening on — useless for hole
        // punching on anything stricter than a full-cone NAT.
        self.transport.listen(listen_addr).await?;
        *self.listen_addr.write().await = Some(listen_addr);

        if let Some(ext_addr) = self.transport.discovered_external_addr().await {
            info!("External address for {listen_addr}: {ext_addr}");
            *self.external_addr.write().await = Some(ext_addr);
            local_info.external_addr = Some(ext_addr);
        } else {
            debug!("No external address discovered; advertising {listen_addr} only");
        }

        // Listening is what makes us runnable; registration is what makes us
        // reachable by strangers. Treat them separately so a coordinator
        // outage degrades to local-only instead of failing startup — and so
        // nothing can claim we are on the hive when we are not.
        *self.local_info.write().await = Some(local_info.clone());
        let registration = self.discovery.register(&local_info).await;
        let registered = registration.is_ok();
        self.registered
            .store(registered, std::sync::atomic::Ordering::Relaxed);
        self.running
            .store(true, std::sync::atomic::Ordering::Relaxed);

        match registration {
            Ok(()) => info!("Mesh node {} is running and registered", self.local_id),
            Err(e) => warn!(
                "Mesh node {} is running but NOT registered with the coordination \
                 server ({e}); it will retry on the next heartbeat. Peers cannot \
                 discover this node until then.",
                self.local_id
            ),
        }
        Ok(())
    }

    /// Start the background maintenance loop.
    ///
    /// This spawns a task that periodically:
    /// - Sends heartbeats to the coordination server
    /// - Pings connected peers and disconnects stale ones
    /// - Discovers and connects to new peers
    pub fn start_maintenance(self: &Arc<Self>) {
        let node = Arc::clone(self);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut heartbeat_interval = tokio::time::interval(Duration::from_secs(60));
            let mut health_interval = tokio::time::interval(Duration::from_secs(30));
            let mut discovery_interval = tokio::time::interval(Duration::from_secs(120));
            // Punch requests are time-sensitive: the far side is dialling
            // *now*, so a slow poll means the window has closed by the time
            // we answer.
            let mut signal_interval = tokio::time::interval(Duration::from_secs(3));

            // Don't fire immediately for health/discovery
            health_interval.tick().await;
            discovery_interval.tick().await;

            loop {
                tokio::select! {
                    _ = shutdown.notified() => {
                        debug!("Maintenance loop shutting down");
                        break;
                    }
                    _ = heartbeat_interval.tick() => {
                        if !node.is_running() { break; }
                        match node.discovery.heartbeat().await {
                            Ok(()) => {
                                if !node.is_registered() {
                                    info!("Re-established contact with the coordination server");
                                }
                                node.registered.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                            Err(e) => {
                                if node.is_registered() {
                                    warn!("Lost contact with the coordination server: {e}");
                                }
                                node.registered.store(false, std::sync::atomic::Ordering::Relaxed);
                                // register() ran exactly once, at startup. A
                                // coordinator restart, or a registration that
                                // failed on the way up, therefore left the
                                // node heartbeating forever against a server
                                // that had never heard of it — invisible to
                                // every peer, with no way back.
                                node.try_reregister().await;
                            }
                        }
                    }
                    _ = health_interval.tick() => {
                        if !node.is_running() { break; }
                        node.check_peer_health().await;
                    }
                    _ = discovery_interval.tick() => {
                        if !node.is_running() { break; }
                        node.discover_and_connect_peers().await;
                    }
                    _ = signal_interval.tick() => {
                        if !node.is_running() { break; }
                        node.handle_pending_signals().await;
                    }
                }
            }
        });
    }

    /// Ping all connected peers and disconnect those that are unresponsive.
    async fn check_peer_health(&self) {
        let peer_ids: Vec<NodeId> = self
            .peers
            .iter()
            .filter(|entry| {
                matches!(
                    entry.value().1,
                    PeerState::Connected | PeerState::Active { .. }
                )
            })
            .map(|entry| entry.value().0.node_id.clone())
            .collect();

        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        for peer_id in peer_ids {
            if let Err(e) = self
                .transport
                .send(&peer_id, MeshMessage::Ping { timestamp_ms })
                .await
            {
                debug!("Peer {} unreachable during health check: {e}", peer_id);
                // Mark as disconnected
                let key = peer_id.0.to_bytes().to_vec();
                if let Some(mut entry) = self.peers.get_mut(&key) {
                    entry.value_mut().1 = PeerState::Disconnected;
                }
                let _ = self.transport.disconnect(&peer_id).await;
            }
        }
    }

    /// Discover new peers from the coordination server and connect to them.
    ///
    /// Uses a tiered connection strategy for NAT traversal:
    /// 1. Direct connect to the peer's local address
    /// 2. Try the peer's STUN-discovered external address
    /// 3. Attempt hole-punch (simultaneous QUIC handshake)
    /// 4. Fall back to relay server
    async fn discover_and_connect_peers(&self) {
        let peers = match self.discovery.find_peers("", 0).await {
            Ok(p) => p,
            Err(e) => {
                debug!("Peer discovery failed (non-fatal): {e}");
                return;
            }
        };

        let mut candidates = Vec::new();
        for peer_info in peers {
            // Skip self
            if peer_info.node_id == self.local_id {
                continue;
            }

            // Skip already-connected peers
            let key = peer_info.node_id.0.to_bytes().to_vec();
            if self.peers.contains_key(&key) {
                continue;
            }

            // Check reputation
            {
                let rep = self.reputation.lock().await;
                if rep.is_banned(&peer_info.node_id) {
                    debug!("Skipping banned peer {}", peer_info.node_id);
                    continue;
                }
                let score = rep.score(&peer_info.node_id);
                if score < self.min_reputation {
                    debug!(
                        "Skipping {}: reputation {score:.2} is below the configured minimum {:.2}",
                        peer_info.node_id, self.min_reputation
                    );
                    continue;
                }
            }

            candidates.push(peer_info);
        }

        if candidates.is_empty() {
            return;
        }

        // Connect concurrently, bounded.
        //
        // This used to be a sequential loop, and each unreachable peer costs
        // seconds of timeouts. It runs inside one arm of the maintenance
        // `select!`, so with a couple of dozen peers the arm could occupy the
        // loop for minutes — during which heartbeats (60s) and health checks
        // (30s) never fired and the coordinator dropped us for being silent.
        // The whole sweep is also capped, so a pathological peer list cannot
        // stall the next tick.
        use futures::stream::StreamExt;
        let attempts = futures::stream::iter(candidates.into_iter().map(|peer_info| async move {
            let result = self.connect_with_nat_traversal(&peer_info).await;
            (peer_info, result)
        }))
        .buffer_unordered(MAX_CONCURRENT_CONNECTS);

        let sweep = tokio::time::timeout(
            DISCOVERY_SWEEP_BUDGET,
            attempts.for_each(|(peer_info, result)| async move {
                match result {
                    Ok(connected_id) => {
                        info!(
                            "Auto-connected to peer {} at {}",
                            connected_id, peer_info.addr
                        );
                        let key = connected_id.0.to_bytes().to_vec();
                        self.peers.insert(key, (peer_info, PeerState::Connected));
                    }
                    Err(e) => {
                        debug!(
                            "All connection strategies failed for peer {}: {e}",
                            peer_info.node_id
                        );
                    }
                }
            }),
        )
        .await;

        if sweep.is_err() {
            debug!(
                "Peer connection sweep hit its {:?} budget; remaining peers wait for the next tick",
                DISCOVERY_SWEEP_BUDGET
            );
        }

        if self.peer_count() > 0 {
            debug!("Connected to {} mesh peers", self.peer_count());
        }
    }

    /// Answer signalling messages relayed by the coordinator.
    ///
    /// Currently one kind: a peer asking us to dial it so both sides punch
    /// simultaneously. Our outbound packet opens our NAT for theirs; without
    /// it, their punch attempts hit a closed mapping and hole punching
    /// cannot work at all.
    async fn handle_pending_signals(&self) {
        let signals = match self.discovery.poll_signals().await {
            Ok(s) => s,
            Err(e) => {
                debug!("Signal poll failed (non-fatal): {e}");
                return;
            }
        };

        for signal in signals {
            let Some(target) = nat::holepunch::punch_target_from_signal(&signal) else {
                continue;
            };

            debug!("Punch request received; dialling {target} back");
            // Best effort and deliberately short: the point is to emit
            // packets toward them, not to wait for a result. If the
            // connection lands, all the better.
            let transport = Arc::clone(&self.transport);
            tokio::spawn(async move {
                match tokio::time::timeout(Duration::from_secs(3), transport.connect(target)).await
                {
                    Ok(Ok(id)) => info!("Punch-back to {target} connected as {id}"),
                    Ok(Err(e)) => debug!("Punch-back to {target} failed: {e}"),
                    Err(_) => debug!("Punch-back to {target} timed out"),
                }
            });
        }
    }

    /// Attempt to connect to a peer using a tiered NAT traversal strategy.
    ///
    /// Tries each method in order, falling through to the next on failure:
    /// 1. Direct connection to advertised address
    /// 2. Connection to STUN-discovered external address
    /// 3. Hole-punch via simultaneous connect
    /// 4. Relay server fallback
    async fn connect_with_nat_traversal(&self, peer: &PeerInfo) -> Result<NodeId> {
        // Strategy 1: Direct connect (works on LAN or when no NAT)
        let direct_timeout = Duration::from_secs(3);
        if let Ok(Ok(id)) =
            tokio::time::timeout(direct_timeout, self.transport.connect(peer.addr)).await
        {
            debug!("Direct connect succeeded to {}", peer.node_id);
            return Ok(id);
        }

        // Strategy 2: Try STUN-discovered external address
        if let Some(ext_addr) = peer.external_addr {
            debug!("Trying external address {} for {}", ext_addr, peer.node_id);
            if let Ok(Ok(id)) =
                tokio::time::timeout(direct_timeout, self.transport.connect(ext_addr)).await
            {
                debug!("External address connect succeeded to {}", peer.node_id);
                return Ok(id);
            }
        }

        // Strategy 3: Hole-punch — both sides dial at once.
        //
        // A punch only works if the peer is dialling us at the same time;
        // its outbound packet is what opens its NAT for ours. Previously
        // nothing told the peer to start, so this was just a slower retry of
        // the direct connect that had already failed. Ask the coordinator to
        // relay a punch-request first, then dial repeatedly while the peer
        // (on receiving it) dials back.
        let our_ext = *self.external_addr.read().await;
        if our_ext.is_some() || peer.external_addr.is_some() {
            let target = peer.external_addr.unwrap_or(peer.addr);

            // Tell the peer where to dial us. Prefer the NAT-mapped address;
            // fall back to whatever we are listening on.
            let reachable_at = match our_ext {
                Some(a) => Some(a.to_string()),
                None => self.listen_addr().await.map(|a| a.to_string()),
            };
            if let Err(e) = self
                .discovery
                .send_signal(
                    &peer.node_id.to_hex(),
                    nat::holepunch::PUNCH_REQUEST,
                    reachable_at.as_deref(),
                )
                .await
            {
                debug!("Could not signal {} to punch: {e}", peer.node_id);
            }

            match nat::holepunch::attempt_holepunch(
                self.transport.as_ref(),
                peer,
                target,
                Duration::from_secs(5),
            )
            .await
            {
                Ok(id) => {
                    debug!("Hole-punch succeeded to {}", peer.node_id);
                    return Ok(id);
                }
                Err(e) => {
                    debug!("Hole-punch failed for {}: {e}", peer.node_id);
                }
            }
        }

        // Strategy 4: relay.
        //
        // Only a node behind a symmetric NAT needs an allocation; everyone
        // else just dials the relayed address it advertises, which
        // strategies 1 and 2 already do. What this node must do is authorise
        // the peer, or its relay drops their return traffic.
        let target = peer.external_addr.unwrap_or(peer.addr);
        match self.transport.relay_to(target).await {
            Ok(()) => {
                if let Ok(id) =
                    tokio::time::timeout(Duration::from_secs(5), self.transport.connect(target))
                        .await
                        .unwrap_or_else(|_| {
                            Err(crate::error::MeshError::Transport(
                                "relay connect timed out".into(),
                            ))
                        })
                {
                    info!("Connected to {} via the relay", peer.node_id);
                    return Ok(id);
                }
            }
            Err(e) => debug!("Relay unavailable for {}: {e}", peer.node_id),
        }

        // Note on the old strategy 4.
        //
        // nat::relay is not a TURN client: it POSTs JSON to port 3478 — the
        // TURN port — and performs no allocation, permission or channel-bind
        // exchange. There is also no server behind it; relay.hivebear.com
        // does not resolve. Attempting it cost roughly 15s per unreachable
        // peer (is_available 5s + allocate 10s), sequentially, inside the
        // maintenance loop, which is a large part of why heartbeats and
        // health checks were starved.
        //
        // Symmetric NATs need a real TURN client *and* a deployed TURN
        // server. Until both exist, failing fast is the honest outcome.

        Err(crate::error::MeshError::Transport(format!(
            "All connection strategies exhausted for peer {}",
            peer.node_id
        )))
    }

    /// Start the mesh node in the background without blocking the caller.
    ///
    /// Spawns a task that listens, registers with discovery, and starts
    /// maintenance. Returns immediately so the CLI startup path is never
    /// blocked by network issues. Errors from the background task itself are
    /// logged, not propagated.
    ///
    /// Requires an entered Tokio runtime. `tokio::spawn` panics when there is
    /// none, and on Android that panic unwound across the FFI boundary and
    /// aborted the whole app with SIGABRT, right after the first frame
    /// rendered. The caller guarded this with `if let Err(..)`, which a panic
    /// does not trigger. Return an error instead so a caller on a thread
    /// without a reactor gets a bad mesh rather than a dead process.
    pub fn start_background(
        self: &Arc<Self>,
        listen_addr: SocketAddr,
        local_info: PeerInfo,
    ) -> Result<()> {
        let handle = tokio::runtime::Handle::try_current().map_err(|_| {
            crate::error::MeshError::Transport(
                "start_background must be called from a Tokio runtime context".to_string(),
            )
        })?;
        let node = Arc::clone(self);
        handle.spawn(async move {
            match node.start(listen_addr, local_info).await {
                Ok(()) => {
                    node.start_maintenance();
                    info!("Mesh node {} running in background", node.local_id);
                }
                Err(e) => {
                    warn!("Background mesh start failed (non-fatal): {e}");
                }
            }
        });
        Ok(())
    }

    /// Stop the mesh node gracefully.
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping mesh node {}", self.local_id);
        self.running
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.shutdown.notify_waiters();

        // Disconnect all peers
        let peer_ids: Vec<NodeId> = self
            .peers
            .iter()
            .map(|entry| entry.value().0.node_id.clone())
            .collect();

        for peer_id in peer_ids {
            if let Err(e) = self.transport.disconnect(&peer_id).await {
                warn!("Failed to disconnect from {peer_id}: {e}");
            }
        }
        self.peers.clear();

        self.discovery.deregister().await?;

        info!("Mesh node {} stopped", self.local_id);
        Ok(())
    }

    /// Check if the node is running (listening locally).
    ///
    /// This does NOT mean the coordination server knows about us — use
    /// [`Self::is_registered`] for that.
    pub fn is_running(&self) -> bool {
        self.running.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Address this node is listening on, once started.
    pub async fn listen_addr(&self) -> Option<SocketAddr> {
        *self.listen_addr.read().await
    }

    /// Whether the coordination server has acknowledged this node.
    ///
    /// Anything that tells a user they are "connected to the hive" must read
    /// this, not `is_running`.
    pub fn is_registered(&self) -> bool {
        self.registered.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if the node has any connected peers.
    pub fn is_connected(&self) -> bool {
        self.transport.peer_count() > 0
    }

    /// Number of connected peers.
    pub fn peer_count(&self) -> usize {
        self.transport.peer_count()
    }

    /// Connect to a peer and add them to our peer state.
    pub async fn connect_to_peer(&self, addr: SocketAddr) -> Result<NodeId> {
        let peer_id = self.transport.connect(addr).await?;
        Ok(peer_id)
    }

    /// Get a summary of all known peers and their states.
    pub fn peer_summary(&self) -> Vec<(NodeId, PeerState, f64)> {
        self.peers
            .iter()
            .map(|entry| {
                let (info, state) = entry.value();
                (info.node_id.clone(), state.clone(), info.reputation_score)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::MockDiscovery;
    use crate::transport::mock::{MockRegistry, MockTransport};

    fn make_node() -> MeshNode {
        let (_id, _) = NodeId::generate();
        let registry = MockRegistry::new();
        let (id2, _) = NodeId::generate();
        let transport = Arc::new(MockTransport::new(id2, registry));
        let discovery = Arc::new(MockDiscovery::new());

        MeshNode::new(transport, discovery, MeshTier::Free, None)
    }

    #[test]
    fn test_node_creation() {
        let node = make_node();
        assert!(!node.is_running());
        assert!(!node.is_connected());
        assert_eq!(node.peer_count(), 0);
    }

    #[test]
    fn test_node_with_identity() {
        let identity = NodeIdentity::generate();
        let expected_id = identity.node_id.to_hex();

        let (id, _) = NodeId::generate();
        let registry = MockRegistry::new();
        let transport = Arc::new(MockTransport::new(id, registry));
        let discovery = Arc::new(MockDiscovery::new());

        let node = MeshNode::with_identity(identity, transport, discovery, MeshTier::Free, None);
        assert_eq!(node.local_id.to_hex(), expected_id);
    }
}
