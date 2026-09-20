use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info, warn};

use crate::node::MeshNode;
use crate::pipeline::initiator::PipelineInitiator;
use crate::protocol::MeshPipelineHandler;
use crate::scheduler::swarm_scheduler::SwarmAwareScheduler;
use crate::scheduler::LayerScheduler;
use crate::swarm::router::SwarmRouter;
use hivebear_core::types::{InferenceEngine, ModelFormat};
use hivebear_inference::engine::{InferenceBackend, TokenStream};
use hivebear_inference::error::{InferenceError, Result};
#[allow(unused_imports)]
use hivebear_inference::types::*;

/// Mesh backend that implements `InferenceBackend` for distributed inference.
///
/// When the Orchestrator selects this backend, inference is distributed
/// across mesh peers rather than running locally.
pub struct MeshBackend {
    node: Arc<MeshNode>,
    scheduler: Arc<dyn LayerScheduler>,
    router: Arc<SwarmRouter>,
    /// Pipelines set up by `load_model`, keyed by the model path the handle
    /// carries, so `unload` can release them on the peers. Without this,
    /// every mesh load left a model resident on every peer forever.
    active: dashmap::DashMap<std::path::PathBuf, ActiveSession>,
    /// Local stage handler, when this node can serve one.
    ///
    /// Required for layer splitting: the initiator owns layers `0..k`, so it
    /// holds the token embeddings (which `load_partial` provides only when
    /// the range starts at 0) and the tokenizer needed to turn a sampled id
    /// back into text. Without a handler the backend can only replicate.
    pipeline_handler: Option<Arc<dyn MeshPipelineHandler>>,
}

/// How many leading layers the initiator keeps for itself.
///
/// It must own at least one, because only a stage starting at layer 0 gets
/// the token embeddings, and it must leave at least one for the peers, or
/// nothing is actually distributed. Returns `None` when neither is possible.
fn local_stage_end(total_layers: u32, peer_count: usize) -> Option<u32> {
    if total_layers < 2 || peer_count == 0 {
        return None;
    }
    let participants = peer_count as u32 + 1;
    Some(
        total_layers
            .div_ceil(participants)
            .clamp(1, total_layers - 1),
    )
}

/// Move peer stages past the local one and make the last stage reach the end.
///
/// The scheduler always plans from layer zero, so without the shift every
/// peer would be told to serve layers the initiator is already serving. And
/// the final stage has to finish at `total_layers`: the output projection is
/// only loaded for a range ending there, so otherwise no peer can produce
/// logits and generation blocks forever.
fn shift_and_seal(
    assignments: &mut [crate::scheduler::plan::LayerAssignment],
    local_end: u32,
    total_layers: u32,
) {
    for assignment in assignments.iter_mut() {
        assignment.layer_range.start += local_end;
        assignment.layer_range.end += local_end;
    }
    if let Some(last) = assignments.last_mut() {
        last.layer_range.end = total_layers;
    }
}

/// How a loaded model is being served.
#[derive(Clone)]
struct ActiveSession {
    initiator: Arc<PipelineInitiator>,
    mode: ServingMode,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ServingMode {
    /// One peer runs the whole model.
    Replicated,
    /// Layers split: this node runs `0..local_end`, peers run the rest.
    Split,
}

impl MeshBackend {
    pub fn new(node: Arc<MeshNode>) -> Self {
        Self {
            node,
            scheduler: Arc::new(SwarmAwareScheduler::new()),
            router: Arc::new(SwarmRouter::new()),
            active: dashmap::DashMap::new(),
            pipeline_handler: None,
        }
    }

    /// Give this backend a local stage handler, enabling layer splitting.
    pub fn with_pipeline_handler(mut self, handler: Arc<dyn MeshPipelineHandler>) -> Self {
        self.pipeline_handler = Some(handler);
        self
    }

    pub fn with_scheduler(node: Arc<MeshNode>, scheduler: Arc<dyn LayerScheduler>) -> Self {
        Self {
            node,
            scheduler,
            router: Arc::new(SwarmRouter::new()),
            active: dashmap::DashMap::new(),
            pipeline_handler: None,
        }
    }

    /// Set up a layer-split session: this node takes `0..k`, peers take the
    /// rest.
    ///
    /// The initiator has to own a stage starting at layer 0 — that is the
    /// only way `load_partial` hands it the token embeddings, and `embed()`
    /// fails on any other stage. It keeps that share small: the point of the
    /// mesh is to run a model this machine could not hold alone.
    async fn try_split(
        &self,
        path: &Path,
        handler: Arc<dyn MeshPipelineHandler>,
        total_layers: u32,
        peers: &[crate::peer::PeerInfo],
    ) -> std::result::Result<ActiveSession, String> {
        if total_layers < 2 {
            return Err(format!("{total_layers} layers is too few to split"));
        }
        if peers.is_empty() {
            return Err("no peers to take the remaining layers".into());
        }

        // Share the model across this node plus the peers, then hand the
        // local node the first slice. Always leave at least one layer for
        // the peers, or there is nothing distributed about it.
        let local_end = local_stage_end(total_layers, peers.len())
            .ok_or_else(|| format!("{total_layers} layers cannot be split across these peers"))?;

        let source = path.display().to_string();
        handler
            .load_layers(&source, 0..local_end, total_layers)
            .await
            .map_err(|e| format!("local stage 0..{local_end} would not load: {e}"))?;

        // Plan the remainder, then shift the scheduler's ranges up past the
        // local stage. The scheduler always plans from zero, so without the
        // shift every peer would be told to serve layers this node is
        // already serving — and nobody would hold the output head.
        let remaining = total_layers - local_end;
        let model_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let mut plan = self
            .scheduler
            .plan(&source, remaining, model_size, peers)
            .await
            .map_err(|e| format!("scheduling the remaining {remaining} layers failed: {e}"))?;

        if plan.assignments.is_empty() {
            return Err("scheduler produced no assignments".into());
        }
        shift_and_seal(&mut plan.assignments, local_end, total_layers);
        plan.total_layers = total_layers;

        info!(
            "Layer split for {}: local 0..{local_end}, {} peer stage(s) covering {local_end}..{total_layers}",
            path.display(),
            plan.assignments.len()
        );

        let initiator = Arc::new(PipelineInitiator::new(
            self.node.transport.clone(),
            plan,
            self.node.local_id.clone(),
        ));
        initiator
            .setup(&source)
            .await
            .map_err(|e| format!("peers would not take their stages: {e}"))?;

        Ok(ActiveSession {
            initiator,
            mode: ServingMode::Split,
        })
    }

    /// Get a reference to the swarm router for registration/management.
    pub fn router(&self) -> &Arc<SwarmRouter> {
        &self.router
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl InferenceBackend for MeshBackend {
    fn engine_id(&self) -> InferenceEngine {
        InferenceEngine::Mesh
    }

    fn name(&self) -> &str {
        "P2P Mesh"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        // Start with GGUF as it's the most common format
        &[ModelFormat::Gguf]
    }

    fn is_available(&self) -> bool {
        self.node.is_running() && self.node.peer_count() > 0
    }

    fn supports_grammar(&self) -> bool {
        // Grammar support depends on the underlying backend used by workers.
        // Conservative: report false until we can query worker capabilities.
        false
    }

    async fn load_model(&self, path: &Path, _config: &LoadConfig) -> Result<ModelHandle> {
        info!("Loading model via mesh: {}", path.display());

        // Discover available peers
        let peers = self
            .node
            .discovery
            .find_peers("", 0)
            .await
            .map_err(|e| InferenceError::LoadError(format!("Discovery failed: {e}")))?;

        if peers.is_empty() {
            return Err(InferenceError::LoadError("No mesh peers available".into()));
        }

        // Prefer a real layer split when we can do one.
        //
        // Needs three things: a local stage handler (the initiator must hold
        // the token embeddings and tokenizer, which `load_partial` only
        // provides for a range starting at 0), a genuine layer count from
        // the model's own metadata, and at least one peer to take the rest.
        let block_count = hivebear_core::gguf::read_block_count(path);
        if let (Some(handler), Some(total_layers)) = (&self.pipeline_handler, block_count) {
            match self
                .try_split(path, handler.clone(), total_layers, &peers)
                .await
            {
                Ok(session) => {
                    self.active.insert(path.to_path_buf(), session);
                    return Ok(ModelHandle::new(path.to_path_buf(), InferenceEngine::Mesh));
                }
                Err(e) => {
                    // Falling back is better than failing: replication still
                    // runs the model, just on one peer.
                    warn!("Layer split unavailable ({e}); falling back to replication");
                }
            }
        } else if block_count.is_none() {
            debug!(
                "No block_count in {}; cannot size a layer split",
                path.display()
            );
        }

        // Plan for replication: one peer serves the whole model.
        //
        // This used to fabricate a 32-layer split (`total_layers = 32 //
        // Default; would read from model file`) and call `initiator.setup()`,
        // which sends AssignLayers to every peer and makes them load those
        // layer ranges. Generation then issued a full-model InferenceRequest
        // instead, so none of those partial loads was ever used — peers were
        // holding layer ranges for nothing, and `unload` did not release them.
        //
        // Real layer splitting needs two things this backend does not have
        // yet: an actual layer count from the model's metadata, and the
        // auto-regressive pipeline token loop (PipelineInitiator::stream_tokens,
        // whose dtype contract is still broken and which has no callers).
        // Until both land, planning for one stage is the honest description
        // of what happens.
        let total_layers = 1;
        let model_size = std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(4 * 1024 * 1024 * 1024);

        let plan = self
            .scheduler
            .plan(
                &path.display().to_string(),
                total_layers,
                model_size,
                &peers,
            )
            .await
            .map_err(|e| InferenceError::LoadError(format!("Scheduling failed: {e}")))?;

        info!(
            "Mesh plan: {} peer(s), est. {:.1} tok/s",
            plan.peer_count(),
            plan.estimated_throughput_tok_s
        );

        // Deliberately no setup() call: replication needs no layer
        // assignment, and issuing one only pins resources on peers that the
        // inference path will not touch.
        let initiator = PipelineInitiator::new(
            self.node.transport.clone(),
            plan,
            self.node.local_id.clone(),
        );
        self.active.insert(
            path.to_path_buf(),
            ActiveSession {
                initiator: Arc::new(initiator),
                mode: ServingMode::Replicated,
            },
        );

        Ok(ModelHandle::new(path.to_path_buf(), InferenceEngine::Mesh))
    }

    async fn generate(
        &self,
        handle: &ModelHandle,
        req: &GenerateRequest,
    ) -> Result<GenerateResponse> {
        // For non-streaming: collect all tokens from the stream
        let mut stream = self.stream(handle, req);
        let mut text = String::new();

        use futures::StreamExt;
        while let Some(result) = stream.next().await {
            match result {
                Ok(token) => text.push_str(&token.text),
                Err(e) => return Err(e),
            }
        }

        Ok(GenerateResponse::Text(text))
    }

    fn stream(&self, handle: &ModelHandle, req: &GenerateRequest) -> TokenStream {
        let max_tokens = req.max_tokens;
        let temperature = req.sampling.temperature;
        let top_p = req.sampling.top_p;
        let transport = self.node.transport.clone();
        let scheduler = self.scheduler.clone();
        let node = self.node.clone();
        let model_id = handle.model_path.display().to_string();
        let messages_json = serde_json::to_string(&req.messages).unwrap_or_else(|_| "[]".into());

        let (tx, rx) = tokio::sync::mpsc::channel(32);

        // Reuse the pipeline load_model already negotiated, if there is one.
        // stream() used to throw it away and re-plan from scratch with
        // total_layers = 1 and a hardcoded 4 GB model size — so the peers
        // load_model had assigned layers to were left holding a model nobody
        // subsequently used, and the scheduler's ranking was recomputed
        // against invented numbers.
        let existing = self
            .active
            .get(&handle.model_path)
            .map(|e| e.value().clone());

        if let Some(session) = existing {
            let initiator = session.initiator;
            info!(
                "Reusing {:?} mesh session {} for {}",
                session.mode,
                initiator.session_id(),
                handle.model_path.display()
            );

            match session.mode {
                ServingMode::Split => {
                    // Layer-split generation: the initiator embeds locally,
                    // runs its own stage, and hands the activation down the
                    // pipeline. Needs the tokenizer and embeddings, which is
                    // exactly what the local stage holds.
                    let Some(handler) = self.pipeline_handler.clone() else {
                        let _ = tx.try_send(Err(InferenceError::GenerationError(
                            "split session without a local stage handler".into(),
                        )));
                        return Box::pin(ReceiverStream::new(rx));
                    };
                    let messages_json = messages_json.clone();
                    tokio::spawn(async move {
                        let prompt_tokens = match handler.tokenize(&messages_json).await {
                            Ok(t) => t,
                            Err(e) => {
                                let _ = tx
                                    .send(Err(InferenceError::GenerationError(format!(
                                        "Could not tokenize the prompt: {e}"
                                    ))))
                                    .await;
                                return;
                            }
                        };
                        let mut token_rx = initiator.stream_tokens(
                            prompt_tokens,
                            max_tokens,
                            temperature,
                            top_p,
                            handler,
                        );
                        while let Some(result) = token_rx.recv().await {
                            let mapped = result.map_err(|e| {
                                InferenceError::GenerationError(format!("Mesh error: {e}"))
                            });
                            if tx.send(mapped).await.is_err() {
                                break;
                            }
                        }
                    });
                }
                ServingMode::Replicated => {
                    let model_id = model_id.clone();
                    let messages_json = messages_json.clone();
                    tokio::spawn(async move {
                        let mut token_rx = initiator.stream_tokens_replicated(
                            model_id,
                            messages_json,
                            max_tokens,
                            temperature,
                            top_p,
                        );
                        while let Some(result) = token_rx.recv().await {
                            let mapped = result.map_err(|e| {
                                InferenceError::GenerationError(format!("Mesh error: {e}"))
                            });
                            if tx.send(mapped).await.is_err() {
                                break;
                            }
                        }
                    });
                }
            }
            return Box::pin(ReceiverStream::new(rx));
        }

        tokio::spawn(async move {
            // Discover peers for this inference
            let peers = match node.discovery.find_peers("", 0).await {
                Ok(p) if !p.is_empty() => p,
                Ok(_) => {
                    let _ = tx
                        .send(Err(InferenceError::GenerationError(
                            "No mesh peers available for distributed inference".into(),
                        )))
                        .await;
                    return;
                }
                Err(e) => {
                    let _ = tx
                        .send(Err(InferenceError::GenerationError(format!(
                            "Discovery failed: {e}"
                        ))))
                        .await;
                    return;
                }
            };

            // No pre-negotiated pipeline: fall back to replication, where a
            // single peer serves the whole model. total_layers = 1 is correct
            // for that (the "pipeline" is one stage); the size is a scheduling
            // hint only, and is read from disk when the path is local.
            let total_layers = 1;
            let model_size = std::fs::metadata(&model_id)
                .map(|m| m.len())
                .unwrap_or(4 * 1024 * 1024 * 1024);
            let plan = match scheduler
                .plan(&model_id, total_layers, model_size, &peers)
                .await
            {
                Ok(p) => p,
                Err(e) => {
                    let _ = tx
                        .send(Err(InferenceError::GenerationError(format!(
                            "Scheduling failed: {e}"
                        ))))
                        .await;
                    return;
                }
            };

            info!("Mesh replication: routing to {} peer(s)", plan.peer_count());

            // Use full-model replication with trust verification
            let mut initiator =
                PipelineInitiator::new(transport.clone(), plan, node.local_id.clone());

            // Wire trust verification from the node's config
            let verifier = Arc::new(crate::trust::TrustVerifier::new(
                transport.clone(),
                0.1, // Default 10% verification rate
            ));
            let reputation = Arc::new(tokio::sync::Mutex::new(
                crate::trust::ReputationManager::new(None),
            ));
            initiator = initiator.with_trust(verifier, reputation);

            let initiator = Arc::new(initiator);
            let mut token_rx = initiator.stream_tokens_replicated(
                model_id,
                messages_json,
                max_tokens,
                temperature,
                top_p,
            );

            while let Some(result) = token_rx.recv().await {
                let mapped =
                    result.map_err(|e| InferenceError::GenerationError(format!("Mesh error: {e}")));
                if tx.send(mapped).await.is_err() {
                    break;
                }
            }
        });

        Box::pin(ReceiverStream::new(rx))
    }

    async fn unload(&self, handle: &ModelHandle) -> Result<()> {
        // This used to be a no-op with a comment claiming sessions were
        // tracked "at a higher level". They were not, so every mesh load
        // leaked a loaded model on every peer for the rest of their uptime.
        match self.active.remove(&handle.model_path) {
            Some((path, session)) => {
                info!(
                    "Releasing mesh session {} for {}",
                    session.initiator.session_id(),
                    path.display()
                );
                session.initiator.teardown().await;
                if session.mode == ServingMode::Split {
                    if let Some(handler) = &self.pipeline_handler {
                        // Release the local stage too, or this node keeps its
                        // slice of every model it has ever initiated.
                        if let Err(e) = handler.unload_layers().await {
                            debug!("Local stage would not unload: {e}");
                        }
                    }
                }
            }
            None => {
                debug!(
                    "No active mesh pipeline for {}; nothing to release",
                    handle.model_path.display()
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peer::NodeId;
    use crate::scheduler::plan::LayerAssignment;

    fn assignment(range: std::ops::Range<u32>) -> LayerAssignment {
        LayerAssignment {
            peer_id: NodeId::generate().0,
            layer_range: range,
            estimated_compute_ms: 1.0,
            estimated_transfer_ms: 1.0,
        }
    }

    #[test]
    fn the_initiator_keeps_a_stage_but_never_all_of_them() {
        // One peer: split roughly in half.
        assert_eq!(local_stage_end(32, 1), Some(16));
        // Three peers: the initiator takes a quarter.
        assert_eq!(local_stage_end(32, 3), Some(8));
        // Never the whole model, however lopsided the arithmetic gets.
        assert_eq!(local_stage_end(2, 1), Some(1));
        assert_eq!(local_stage_end(3, 1), Some(2));
        for peers in 1..8 {
            for total in 2..200u32 {
                let end = local_stage_end(total, peers).unwrap();
                assert!(end >= 1, "the initiator needs layer 0 for the embeddings");
                assert!(
                    end < total,
                    "leaving nothing for peers is not a distributed split"
                );
            }
        }
    }

    #[test]
    fn splitting_is_refused_when_it_would_be_meaningless() {
        assert_eq!(local_stage_end(1, 4), None, "a one-layer model");
        assert_eq!(local_stage_end(0, 4), None);
        assert_eq!(local_stage_end(32, 0), None, "nobody to split with");
    }

    /// The regression this guards: the scheduler always plans from layer
    /// zero, so unshifted assignments tell every peer to serve layers the
    /// initiator already holds — and nobody ends at the last layer, which is
    /// the only stage that loads the output projection. Generation would
    /// then block forever waiting for logits nobody could produce.
    #[test]
    fn peer_stages_follow_the_local_one_and_reach_the_last_layer() {
        let total = 32;
        let local_end = local_stage_end(total, 2).unwrap(); // 11
                                                            // What a scheduler planning 21 remaining layers might return.
        let mut assignments = vec![assignment(0..11), assignment(11..21)];

        shift_and_seal(&mut assignments, local_end, total);

        assert_eq!(assignments[0].layer_range, local_end..(local_end + 11));
        assert_eq!(
            assignments.last().unwrap().layer_range.end,
            total,
            "the final stage must own the output projection"
        );

        // Contiguous, gapless cover of local_end..total.
        let mut cursor = local_end;
        for a in &assignments {
            assert_eq!(a.layer_range.start, cursor, "gap or overlap between stages");
            cursor = a.layer_range.end;
        }
        assert_eq!(cursor, total);
    }

    #[test]
    fn a_single_peer_stage_is_sealed_to_the_end() {
        let mut assignments = vec![assignment(0..4)];
        shift_and_seal(&mut assignments, 8, 32);
        assert_eq!(assignments[0].layer_range, 8..32);
    }
}
