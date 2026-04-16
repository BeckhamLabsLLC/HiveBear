import { invoke as rawInvoke } from "@tauri-apps/api/core";
import { notify } from "../components/Toast";
import type {
  BenchmarkResult, ChatMessage, Config, Conversation, HardwareProfile, InstalledInfo,
  LoadedModel, MeshConfig, MeshStatus, ModelInfo, ModelMetadata, ModelRecommendation,
  PersistedMessage, SearchResult, StorageReport,
} from "../types";

// ── Error normalization + toast surfacing ──────────────────────────
//
// Every Tauri command can reject with one of:
//   - a plain string (our CmdResult<T, String> convention)
//   - a structured { code, message, ... } object
//   - an Error instance (rare, usually from the IPC layer itself)
//
// We want a single place that turns all three into a user-friendly
// message, shows a toast unless the caller opted out, and re-throws
// so caller-side `.catch()` still works.

export class InvokeError extends Error {
  readonly command: string;
  readonly code: string | null;
  readonly raw: unknown;
  constructor(command: string, message: string, code: string | null, raw: unknown) {
    super(message);
    this.command = command;
    this.code = code;
    this.raw = raw;
  }
}

interface InvokeOptions {
  /** Suppress the automatic toast — caller handles the error UX itself. */
  silent?: boolean;
  /** Override the user-facing message prefix (default: command name). */
  label?: string;
}

function friendlyMessage(_command: string, code: string | null, message: string, label?: string): string {
  // Map a few common Rust error codes/patterns to something actionable.
  // Unknown codes fall through to the raw message.
  const lower = (code ?? "").toLowerCase();
  if (lower === "unauthenticated" || lower === "unauthorized") {
    return "You're signed out. Open the Account page to sign back in.";
  }
  if (lower === "network" || /timed out|connection refused|unreachable/i.test(message)) {
    return "Can't reach the coordinator. Check your connection and retry.";
  }
  if (/address.*in use|port.*in use|bind.*in use/i.test(message)) {
    return "Required port is in use by another process (try closing other Ollama / HiveBear instances).";
  }
  if (/not ?found|no such/i.test(message) && !label) {
    return message;
  }
  return label ? `${label}: ${message}` : message;
}

function normalize(command: string, err: unknown): InvokeError {
  if (err instanceof InvokeError) return err;
  if (typeof err === "string") return new InvokeError(command, err, null, err);
  if (err instanceof Error) return new InvokeError(command, err.message, null, err);
  if (err && typeof err === "object") {
    const anyErr = err as { code?: unknown; message?: unknown };
    const code = typeof anyErr.code === "string" ? anyErr.code : null;
    const message = typeof anyErr.message === "string" ? anyErr.message : JSON.stringify(err);
    return new InvokeError(command, message, code, err);
  }
  return new InvokeError(command, String(err), null, err);
}

function invoke<T>(command: string, args?: Record<string, unknown>, opts?: InvokeOptions): Promise<T> {
  return (rawInvoke<T>(command, args) as Promise<T>).catch((e) => {
    const ie = normalize(command, e);
    if (!opts?.silent) {
      notify(friendlyMessage(command, ie.code, ie.message, opts?.label), "error");
    }
    throw ie;
  });
}

// ── Profile ────────────────────────────────────────────────────────

export function getHardwareProfile(): Promise<HardwareProfile> {
  return invoke("get_hardware_profile", undefined, { label: "Hardware profile" });
}

export function getRecommendations(): Promise<ModelRecommendation[]> {
  return invoke("get_recommendations");
}

// ── Registry ───────────────────────────────────────────────────────

export function searchModels(query: string, limit?: number): Promise<SearchResult[]> {
  return invoke("search_models", { query, limit });
}

export function installModel(modelId: string, quant?: string): Promise<InstalledInfo> {
  return invoke("install_model", { modelId, quant });
}

export function listInstalled(): Promise<ModelMetadata[]> {
  return invoke("list_installed");
}

export function removeModel(modelId: string): Promise<number> {
  return invoke("remove_model", { modelId });
}

export function getStorageReport(): Promise<StorageReport> {
  return invoke("get_storage_report");
}

// ── Inference ──────────────────────────────────────────────────────

export function loadModel(modelPath: string, contextLength?: number, gpuLayers?: number): Promise<LoadedModel> {
  return invoke("load_model", { modelPath, contextLength, gpuLayers });
}

export function streamChat(handleId: number, messages: ChatMessage[], temperature?: number, maxTokens?: number): Promise<string> {
  return invoke("stream_chat", { handleId, messages, temperature, maxTokens });
}

export function unloadModel(handleId: number): Promise<void> {
  return invoke("unload_model", { handleId });
}

export function listLoadedModels(): Promise<ModelInfo[]> {
  return invoke("list_loaded_models");
}

// ── Benchmark ──────────────────────────────────────────────────────

export function runBenchmark(durationSecs?: number): Promise<BenchmarkResult | null> {
  return invoke("run_benchmark", { durationSecs });
}

// ── Config ─────────────────────────────────────────────────────────

export function getConfig(): Promise<Config> {
  return invoke("get_config");
}

export function saveConfig(newConfig: Config): Promise<void> {
  return invoke("save_config", { newConfig });
}

// ── Mesh ──────────────────────────────────────────────────────────

export function getMeshStatus(): Promise<MeshStatus> {
  return invoke("get_mesh_status", undefined, { silent: true });
}

export function getMeshConfig(): Promise<MeshConfig> {
  return invoke("get_mesh_config");
}

export function saveMeshConfig(meshConfig: MeshConfig): Promise<void> {
  return invoke("save_mesh_config", { meshConfig });
}

export interface MeshConnectionStatus {
  running: boolean;
  peer_count: number;
  node_id: string | null;
}

export function joinMesh(): Promise<MeshConnectionStatus> {
  return invoke("join_mesh");
}

export function leaveMesh(): Promise<void> {
  return invoke("leave_mesh");
}

export function getMeshConnectionStatus(): Promise<MeshConnectionStatus> {
  // Called on a timer — silence toasts to avoid spamming on transient errors.
  return invoke("get_mesh_connection_status", undefined, { silent: true });
}

// ── Chat Persistence ─────────────────────────────────────────────

export function listConversations(): Promise<Conversation[]> {
  return invoke("list_conversations");
}

export function createConversation(title: string, modelId: string): Promise<Conversation> {
  return invoke("create_conversation", { title, modelId });
}

export function getConversationMessages(conversationId: string): Promise<PersistedMessage[]> {
  return invoke("get_conversation_messages", { conversationId });
}

export function addMessage(conversationId: string, role: string, content: string): Promise<PersistedMessage> {
  return invoke("add_message", { conversationId, role, content });
}

export function deleteConversation(conversationId: string): Promise<void> {
  return invoke("delete_conversation", { conversationId });
}

export function renameConversation(conversationId: string, newTitle: string): Promise<void> {
  return invoke("rename_conversation", { conversationId, newTitle });
}

export function searchConversations(query: string): Promise<Conversation[]> {
  return invoke("search_conversations", { query });
}

// ── Secrets (OS Keychain) ───────────────────────────────────────────

export function setCloudApiKey(provider: string, key: string): Promise<void> {
  return invoke("set_cloud_api_key", { provider, key });
}

export function getCloudApiKeys(): Promise<Record<string, string>> {
  return invoke("get_cloud_api_keys");
}

export function deleteCloudApiKey(provider: string): Promise<void> {
  return invoke("delete_cloud_api_key", { provider });
}

export function migrateApiKeysToKeychain(): Promise<string[]> {
  return invoke("migrate_api_keys_to_keychain");
}

export function isKeychainAvailable(): Promise<boolean> {
  return invoke("is_keychain_available");
}

// ── Device (Mobile) ──────────────────────────────────────────────

export interface DeviceStatus {
  is_charging: boolean;
  battery_percent: number | null;
  is_wifi: boolean;
  is_mobile: boolean;
  thermal_state: string;
}

export interface MeshEligibility {
  eligible: boolean;
  reasons: string[];
}

export function getDeviceStatus(): Promise<DeviceStatus> {
  return invoke("get_device_status", undefined, { silent: true });
}

export function canContributeToMesh(): Promise<MeshEligibility> {
  return invoke("can_contribute_to_mesh");
}
