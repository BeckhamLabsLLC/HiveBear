import { invoke } from "@tauri-apps/api/core";
import type {
  BenchmarkResult, ChatMessage, Config, Conversation, HardwareProfile, InstalledInfo,
  LoadedModel, MeshConfig, MeshStatus, ModelInfo, ModelMetadata, ModelRecommendation,
  PersistedMessage, SearchResult, StorageReport,
} from "../types";

// ── Profile ────────────────────────────────────────────────────────

export function getHardwareProfile(): Promise<HardwareProfile> {
  return invoke("get_hardware_profile");
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
  return invoke("get_mesh_status");
}

export function getMeshConfig(): Promise<MeshConfig> {
  return invoke("get_mesh_config");
}

export function saveMeshConfig(meshConfig: MeshConfig): Promise<void> {
  return invoke("save_mesh_config", { meshConfig });
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
