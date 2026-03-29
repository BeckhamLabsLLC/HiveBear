// ── Core types (from hivebear-core) ─────────────────────────────────

export interface HardwareProfile {
  cpu: CpuInfo;
  memory: MemoryInfo;
  gpus: GpuInfo[];
  storage: StorageInfo;
  platform: PlatformInfo;
}

export interface CpuInfo {
  model_name: string;
  physical_cores: number;
  logical_cores: number;
  isa_extensions: string[];
  cache_size_bytes: number;
}

export interface MemoryInfo {
  total_bytes: number;
  available_bytes: number;
  estimated_bandwidth_gbps: number;
}

export interface GpuInfo {
  name: string;
  vram_bytes: number;
  compute_api: string;
  driver_version: string | null;
}

export interface StorageInfo {
  available_bytes: number;
  estimated_read_speed_mbps: number;
}

export interface PlatformInfo {
  os: string;
  arch: string;
  is_mobile: boolean;
  power_source: PowerSource;
}

export type PowerSource = "Ac" | { Battery: { charge_percent: number } } | "Unknown";

export interface ModelRecommendation {
  model_id: string;
  model_name: string;
  quantization: string;
  engine: string;
  estimated_tokens_per_sec: number;
  estimated_memory_usage_bytes: number;
  confidence: number;
  warnings: string[];
  score: number;
}

export interface BenchmarkResult {
  model_used: string;
  tokens_generated: number;
  total_duration_ms: number;
  tokens_per_sec: number;
  time_to_first_token_ms: number;
  peak_memory_bytes: number;
  cpu_utilization: number;
  gpu_utilization: number | null;
}

export interface Config {
  models_dir: string;
  max_memory_usage: number;
  min_tokens_per_sec: number;
  top_n_recommendations: number;
  share_benchmarks: boolean;
  default_context_length: number;
  // Mesh config (flattened for form state)
  mesh_enabled?: boolean;
  mesh_port?: number;
  mesh_coordination_server?: string;
  mesh_tier?: string;
  mesh_max_contribution?: number;
  mesh_verification_rate?: number;
  // Cloud provider config
  cloud_api_keys?: Record<string, string>;
  cloud_fallback?: boolean;
  cloud_default_provider?: string;
}

/** Static definition of a supported cloud provider. */
export interface CloudProviderInfo {
  prefix: string;
  display_name: string;
  env_var: string;
  requires_api_key: boolean;
}

// ── Registry types (from hivebear-registry) ─────────────────────────

export interface SearchResult {
  metadata: ModelMetadata;
  is_installed: boolean;
  compatibility_score: number | null;
}

export interface ModelMetadata {
  id: string;
  name: string;
  params_billions: number;
  formats: string[];
  context_length: number;
  quality_score: number;
  category: string;
  source: ModelSource;
  huggingface_id: string | null;
  installed: InstalledInfo | null;
  description: string | null;
  tags: string[];
  downloads_count: number | null;
  likes_count: number | null;
  last_modified: string | null;
}

export type ModelSource =
  | { HuggingFace: { repo_id: string; revision: string | null } }
  | { Ollama: { tag: string } }
  | { Local: { imported: boolean } };

export interface InstalledInfo {
  path: string;
  format: string;
  quantization: string | null;
  size_bytes: number;
  sha256: string | null;
  installed_at: string;
  last_used: string | null;
  filename: string;
}

export interface StorageReport {
  total_bytes: number;
  models: ModelStorageInfo[];
  partial_downloads: PartialDownloadInfo[];
  orphaned_files: string[];
}

export interface ModelStorageInfo {
  model_id: string;
  model_name: string;
  size_bytes: number;
  path: string;
  last_used: string | null;
}

export interface PartialDownloadInfo {
  model_id: string;
  filename: string;
  bytes_downloaded: number;
  path: string;
}

export interface DownloadProgress {
  bytes_downloaded: number;
  total_bytes: number | null;
  bytes_per_sec: number;
}

// ── Inference types (from hivebear-inference) ───────────────────────

export interface LoadedModel {
  handle_id: number;
  model_path: string;
  engine: string;
}

export interface ModelInfo {
  handle_id: number;
  model_path: string;
  engine: string;
  context_length: number;
  gpu_layers: number | null;
}

// Matches Rust: #[serde(tag = "role", content = "content")]
// User variant takes Vec<ContentPart>, not a plain string
export type ChatMessage =
  | { role: "system"; content: string }
  | { role: "user"; content: ContentPart[] }
  | { role: "assistant"; content: string }
  | { role: "tool"; content: { tool_call_id: string; content: string } };

export type ContentPart =
  | { type: "text"; text: string }
  | { type: "image"; data: string; media_type: string }
  | { type: "image_url"; url: string };

/** Create a text-only user message (convenience helper). */
export function userTextMessage(text: string): ChatMessage {
  return { role: "user", content: [{ type: "text", text }] };
}

/** Create a system message. */
export function systemMessage(text: string): ChatMessage {
  return { role: "system", content: text };
}

/** Create an assistant message. */
export function assistantMessage(text: string): ChatMessage {
  return { role: "assistant", content: text };
}

// ── Mesh types ────────────────────────────────────────────────────

export interface MeshConfig {
  enabled: boolean;
  port: number;
  coordination_server: string;
  tier: string;
  max_contribution_percent: number;
  min_reputation: number;
  verification_rate: number;
}

export interface MeshStatus {
  enabled: boolean;
  tier: string;
  port: number;
  coordination_server: string;
  max_contribution_percent: number;
  min_reputation: number;
  verification_rate: number;
}

// ── Persistence types (from hivebear-persistence) ─────────────────

export interface Conversation {
  id: string;
  title: string;
  model_id: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface PersistedMessage {
  id: string;
  conversation_id: string;
  role: "System" | "User" | "Assistant";
  content: string;
  tool_call_json: string | null;
  created_at: string;
  sequence_num: number;
}

// ── Errors ─────────────────────────────────────────────────────────

export interface CommandError {
  message: string;
  code: string;
}

// ── Account types ─────────────────────────────────────────────────

export interface AuthResult {
  auth_mode: "email" | "device";
  tier: string;
  jwt: string;
  license_token: string | null;
}

export interface AccountInfo {
  auth_mode: string;
  tier: string;
  status: string;
  user_id: string | null;
  email: string | null;
  display_name: string | null;
  pubkey: string | null;
  current_period_end: string | null;
  limits: {
    cloud_requests_per_day: number | null;
    max_swarm_size: number;
    priority_mesh: boolean;
    private_hives: boolean;
    max_api_keys: number;
  };
}

export interface UsageSummary {
  cloud_requests_today: number;
  cloud_requests_limit: number | null;
  mesh_seconds_contributed: number;
  mesh_seconds_consumed: number;
  model_downloads_this_month: number;
}

export interface ApiKeyInfo {
  id: string;
  key_prefix: string;
  label: string;
  created_at: string;
  last_used_at: string | null;
}

export interface NewApiKey {
  id: string;
  key: string;
  key_prefix: string;
  label: string;
}

// ── Helpers ────────────────────────────────────────────────────────

export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

export function formatToksPerSec(tps: number): string {
  return `${tps.toFixed(1)} tok/s`;
}
