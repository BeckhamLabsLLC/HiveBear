//! Inbound message routing shared by the transports.
//!
//! Every transport used to funnel inbound messages into one
//! `UnboundedReceiver` guarded by a mutex, and `recv()` handed the next
//! message to whichever task happened to win the lock. In practice several
//! tasks call `recv()` at once — the worker daemon, a pipeline initiator, the
//! trust verifier, and the CLI's inline loop — so they stole each other's
//! messages, and the loops that `continue` on a non-matching message dropped
//! them on the floor. That is unfixable by making those loops smarter: the
//! demultiplexing has to happen where messages arrive.
//!
//! `Inbox` routes by [`MeshMessage::session_id`]. A task that owns a session
//! subscribes to it and gets its own queue; everything else (handshakes,
//! pings, swarm traffic, and any session nobody claimed) stays on the general
//! queue that `recv()` drains.

use dashmap::DashMap;
use tokio::sync::{mpsc, Mutex};
use tracing::debug;
use uuid::Uuid;

use super::protocol::MeshMessage;
use crate::error::{MeshError, Result};
use crate::peer::NodeId;

/// Receiving end of a per-session queue.
pub type SessionReceiver = mpsc::UnboundedReceiver<(NodeId, MeshMessage)>;

type Sender = mpsc::UnboundedSender<(NodeId, MeshMessage)>;

/// Routes inbound messages to per-session queues, with a general fallback.
pub struct Inbox {
    general_tx: Sender,
    general_rx: Mutex<mpsc::UnboundedReceiver<(NodeId, MeshMessage)>>,
    sessions: DashMap<Uuid, Sender>,
}

impl Inbox {
    pub fn new() -> Self {
        let (general_tx, general_rx) = mpsc::unbounded_channel();
        Self {
            general_tx,
            general_rx: Mutex::new(general_rx),
            sessions: DashMap::new(),
        }
    }

    /// Hand an inbound message to whoever should receive it.
    ///
    /// Returns false only if the destination queue is gone, which means the
    /// owning task has exited.
    pub fn deliver(&self, from: NodeId, msg: MeshMessage) -> bool {
        if let Some(session_id) = msg.session_id() {
            if let Some(tx) = self.sessions.get(&session_id) {
                if tx.send((from, msg)).is_ok() {
                    return true;
                }
                // Subscriber went away without unsubscribing; fall through to
                // the general queue rather than losing the message silently.
                drop(tx);
                self.sessions.remove(&session_id);
                debug!("Session {session_id} subscriber dropped; routing to general inbox");
                return false;
            }
        }
        self.general_tx.send((from, msg)).is_ok()
    }

    /// Next message not claimed by a session subscriber.
    pub async fn recv(&self) -> Result<(NodeId, MeshMessage)> {
        let mut rx = self.general_rx.lock().await;
        rx.recv()
            .await
            .ok_or_else(|| MeshError::Transport("All senders dropped".into()))
    }

    /// Claim `session_id`; its messages go to the returned queue instead of
    /// the general one. Re-subscribing replaces the previous claim.
    pub fn subscribe(&self, session_id: Uuid) -> SessionReceiver {
        let (tx, rx) = mpsc::unbounded_channel();
        self.sessions.insert(session_id, tx);
        rx
    }

    /// Release `session_id` back to the general queue.
    pub fn unsubscribe(&self, session_id: &Uuid) {
        self.sessions.remove(session_id);
    }

    /// Number of currently claimed sessions. Test/diagnostic helper.
    pub fn subscription_count(&self) -> usize {
        self.sessions.len()
    }
}

impl Default for Inbox {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Inbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inbox")
            .field("subscriptions", &self.sessions.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::protocol::MeshMessage;

    fn peer() -> NodeId {
        NodeId::generate().0
    }

    fn token(session_id: Uuid, text: &str) -> MeshMessage {
        MeshMessage::InferenceToken {
            session_id,
            text: text.to_string(),
            token_id: 0,
            is_done: false,
        }
    }

    fn ping() -> MeshMessage {
        MeshMessage::Ping { timestamp_ms: 1 }
    }

    /// The regression: with one shared queue, whichever task won the mutex
    /// took the next message regardless of who it was for, so a pipeline
    /// initiator and a worker daemon running concurrently stole each other's
    /// traffic and the non-matching ones were dropped on the floor.
    #[tokio::test]
    async fn session_messages_do_not_land_in_the_general_queue() {
        let inbox = Inbox::new();
        let from = peer();
        let session = Uuid::new_v4();

        let mut session_rx = inbox.subscribe(session);

        inbox.deliver(from.clone(), token(session, "mine"));
        inbox.deliver(from.clone(), ping());

        let (_, got) = session_rx.try_recv().expect("session queue should have it");
        match got {
            MeshMessage::InferenceToken { text, .. } => assert_eq!(text, "mine"),
            other => panic!("unexpected message on session queue: {other:?}"),
        }
        assert!(
            session_rx.try_recv().is_err(),
            "the Ping is not session traffic and must not be routed here"
        );

        // The general queue got the Ping, and only the Ping.
        let (_, general) = inbox.recv().await.expect("general queue");
        assert!(matches!(general, MeshMessage::Ping { .. }));
    }

    #[tokio::test]
    async fn two_sessions_stay_separate() {
        let inbox = Inbox::new();
        let from = peer();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();

        let mut rx_a = inbox.subscribe(a);
        let mut rx_b = inbox.subscribe(b);

        inbox.deliver(from.clone(), token(b, "for-b"));
        inbox.deliver(from.clone(), token(a, "for-a"));

        let (_, msg_a) = rx_a.try_recv().expect("session A");
        let (_, msg_b) = rx_b.try_recv().expect("session B");

        let text = |m: MeshMessage| match m {
            MeshMessage::InferenceToken { text, .. } => text,
            other => panic!("unexpected: {other:?}"),
        };
        assert_eq!(text(msg_a), "for-a");
        assert_eq!(text(msg_b), "for-b");
        assert_eq!(inbox.subscription_count(), 2);
    }

    #[tokio::test]
    async fn unsubscribing_returns_traffic_to_the_general_queue() {
        let inbox = Inbox::new();
        let from = peer();
        let session = Uuid::new_v4();

        let rx = inbox.subscribe(session);
        drop(rx);
        inbox.unsubscribe(&session);
        assert_eq!(inbox.subscription_count(), 0);

        inbox.deliver(from, token(session, "orphan"));
        let (_, msg) = inbox.recv().await.expect("general queue");
        assert!(matches!(msg, MeshMessage::InferenceToken { .. }));
    }
}
