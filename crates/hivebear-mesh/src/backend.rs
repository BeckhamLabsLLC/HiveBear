use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info};

use crate::node::MeshNode;
use crate::pipeline::initiator::PipelineInitiator;
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
    active: dashmap::DashMap<std::path::PathBuf, Arc<PipelineInitiator>>,
}

impl MeshBackend {
    pub fn new(node: Arc<MeshNode>) -> Self {
        Self {
            node,
            scheduler: Arc::new(SwarmAwareScheduler::new()),
            router: Arc::new(SwarmRouter::new()),
            active: dashmap::DashMap::new(),
        }
    }

    pub fn with_scheduler(node: Arc<MeshNode>, scheduler: Arc<dyn LayerScheduler>) -> Self {
        Self {
            node,
            scheduler,
            router: Arc::new(SwarmRouter::new()),
            active: dashmap::DashMap::new(),
        }
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
        self.active.insert(path.to_path_buf(), Arc::new(initiator));

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
            .map(|e| Arc::clone(e.value()));

        if let Some(initiator) = existing {
            info!(
                "Reusing mesh session {} for {}",
                initiator.session_id(),
                handle.model_path.display()
            );
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
                    let mapped = result
                        .map_err(|e| InferenceError::GenerationError(format!("Mesh error: {e}")));
                    if tx.send(mapped).await.is_err() {
                        break;
                    }
                }
            });
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
            Some((path, initiator)) => {
                info!(
                    "Releasing mesh session {} for {}",
                    initiator.session_id(),
                    path.display()
                );
                initiator.teardown().await;
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
