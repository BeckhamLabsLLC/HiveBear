import { useCallback, useEffect, useState } from "react";
import { useConfig, useStorageReport } from "../hooks/useConfig";
import { formatBytes } from "../types";
import type { Config } from "../types";
import { Save, Loader, Eye, EyeOff, Cloud, Check } from "lucide-react";

export default function Settings() {
  const { config, loading, saving, error, save } = useConfig();
  const { report, loading: storageLoading } = useStorageReport();
  const [form, setForm] = useState<Config | null>(null);

  useEffect(() => { if (config) setForm({ ...config }); }, [config]);

  const handleSave = useCallback(() => { if (form) save(form); }, [form, save]);

  if (loading || !form) {
    return <div className="flex h-full items-center justify-center text-text-muted">Loading configuration...</div>;
  }

  return (
    <div className="space-y-8 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Settings</h1>
        <button onClick={handleSave} disabled={saving}
          className="flex items-center gap-2 rounded-lg bg-paw-500 px-4 py-2 text-sm font-medium text-white hover:bg-paw-600 disabled:opacity-50">
          {saving ? <Loader size={14} className="animate-spin" /> : <Save size={14} />}
          Save
        </button>
      </div>

      {error && <div className="rounded-lg border border-danger/30 bg-danger/10 px-4 py-2 text-sm text-danger">{error}</div>}

      <Section title="Inference">
        <Field label="Default Context Length" hint="Tokens per conversation turn">
          <input type="number" min={512} max={131072} step={512} value={form.default_context_length}
            onChange={(e) => setForm({ ...form, default_context_length: Number(e.target.value) })}
            className="w-32 rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500" />
        </Field>
        <Field label="Min Tokens/sec" hint="Models below this speed are excluded from recommendations">
          <input type="number" min={1} max={100} step={0.5} value={form.min_tokens_per_sec}
            onChange={(e) => setForm({ ...form, min_tokens_per_sec: Number(e.target.value) })}
            className="w-32 rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500" />
        </Field>
        <Field label="Max Memory Usage" hint="Fraction of available memory to use (0.0 - 1.0)">
          <input type="number" min={0.1} max={1.0} step={0.05} value={form.max_memory_usage}
            onChange={(e) => setForm({ ...form, max_memory_usage: Number(e.target.value) })}
            className="w-32 rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500" />
        </Field>
      </Section>

      <Section title="Model Discovery">
        <Field label="Top N Recommendations">
          <input type="number" min={1} max={50} value={form.top_n_recommendations}
            onChange={(e) => setForm({ ...form, top_n_recommendations: Number(e.target.value) })}
            className="w-32 rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500" />
        </Field>
        <Field label="Models Directory">
          <input type="text" value={form.models_dir}
            onChange={(e) => setForm({ ...form, models_dir: e.target.value })}
            className="w-full max-w-md rounded-lg border border-border bg-surface px-3 py-2 font-mono text-sm outline-none focus:border-paw-500" />
        </Field>
        <Field label="Share Benchmarks" hint="Anonymously contribute to community hardware database">
          <label className="relative inline-flex cursor-pointer items-center">
            <input type="checkbox" checked={form.share_benchmarks}
              onChange={(e) => setForm({ ...form, share_benchmarks: e.target.checked })} className="peer sr-only" />
            <div className="h-5 w-9 rounded-full bg-surface-overlay after:absolute after:left-0.5 after:top-0.5 after:h-4 after:w-4 after:rounded-full after:bg-text-muted after:transition-transform peer-checked:bg-paw-500 peer-checked:after:translate-x-4 peer-checked:after:bg-white" />
          </label>
        </Field>
      </Section>

      <Section title="P2P Mesh">
        <Field label="Enable Mesh" hint="Join the P2P network to share and use distributed compute">
          <label className="relative inline-flex cursor-pointer items-center">
            <input type="checkbox" checked={form.mesh_enabled ?? false}
              onChange={(e) => setForm({ ...form, mesh_enabled: e.target.checked })} className="peer sr-only" />
            <div className="h-5 w-9 rounded-full bg-surface-overlay after:absolute after:left-0.5 after:top-0.5 after:h-4 after:w-4 after:rounded-full after:bg-text-muted after:transition-transform peer-checked:bg-paw-500 peer-checked:after:translate-x-4 peer-checked:after:bg-white" />
          </label>
        </Field>
        <Field label="Port" hint="QUIC transport port (1024-65535)">
          <input type="number" min={1024} max={65535} value={form.mesh_port ?? 7878}
            onChange={(e) => setForm({ ...form, mesh_port: Number(e.target.value) })}
            className="w-32 rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500" />
        </Field>
        <Field label="Coordination Server" hint="URL of the peer discovery server">
          <input type="text" value={form.mesh_coordination_server ?? ""}
            onChange={(e) => setForm({ ...form, mesh_coordination_server: e.target.value })}
            className="w-full max-w-md rounded-lg border border-border bg-surface px-3 py-2 font-mono text-sm outline-none focus:border-paw-500" />
        </Field>
        <Field label="Tier" hint="Free: best-effort. Paid: priority routing.">
          <select value={form.mesh_tier ?? "free"}
            onChange={(e) => setForm({ ...form, mesh_tier: e.target.value })}
            className="rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500">
            <option value="free">Free</option>
            <option value="paid">Paid</option>
          </select>
        </Field>
        <Field label="Max Contribution" hint="Percentage of local resources to share (0-100%)">
          <div className="flex items-center gap-2">
            <input type="range" min={0} max={100} value={(form.mesh_max_contribution ?? 0.8) * 100}
              onChange={(e) => setForm({ ...form, mesh_max_contribution: Number(e.target.value) / 100 })}
              className="w-32" />
            <span className="w-10 text-right font-mono text-xs text-text-muted">
              {((form.mesh_max_contribution ?? 0.8) * 100).toFixed(0)}%
            </span>
          </div>
        </Field>
        <Field label="Verification Rate" hint="Fraction of tokens to spot-check for correctness">
          <div className="flex items-center gap-2">
            <input type="range" min={0} max={100} value={(form.mesh_verification_rate ?? 0.05) * 100}
              onChange={(e) => setForm({ ...form, mesh_verification_rate: Number(e.target.value) / 100 })}
              className="w-32" />
            <span className="w-10 text-right font-mono text-xs text-text-muted">
              {((form.mesh_verification_rate ?? 0.05) * 100).toFixed(0)}%
            </span>
          </div>
        </Field>
      </Section>

      <Section title="Cloud Providers">
        <p className="text-xs text-text-muted mb-2">
          Configure API keys to use cloud models (e.g. <code className="font-mono text-text-secondary">openai/gpt-4o</code>, <code className="font-mono text-text-secondary">groq/llama-3.1-70b</code>).
          Keys can also be set via environment variables.
        </p>
        <CloudProviderKeys
          apiKeys={form.cloud_api_keys ?? {}}
          onChange={(keys) => setForm({ ...form, cloud_api_keys: keys })}
        />
        <Field label="Cloud Fallback" hint="Fall back to cloud when local inference fails">
          <label className="relative inline-flex cursor-pointer items-center">
            <input type="checkbox" checked={form.cloud_fallback ?? false}
              onChange={(e) => setForm({ ...form, cloud_fallback: e.target.checked })} className="peer sr-only" />
            <div className="h-5 w-9 rounded-full bg-surface-overlay after:absolute after:left-0.5 after:top-0.5 after:h-4 after:w-4 after:rounded-full after:bg-text-muted after:transition-transform peer-checked:bg-paw-500 peer-checked:after:translate-x-4 peer-checked:after:bg-white" />
          </label>
        </Field>
      </Section>

      <Section title="Storage">
        {storageLoading ? (
          <p className="text-sm text-text-muted">Loading storage info...</p>
        ) : report ? (
          <div className="space-y-3">
            <p className="text-sm text-text-secondary">
              Total usage: <span className="font-mono">{formatBytes(report.total_bytes)}</span>
              {" · "}{report.models.length} model{report.models.length !== 1 && "s"}
              {report.partial_downloads.length > 0 && ` · ${report.partial_downloads.length} partial downloads`}
              {report.orphaned_files.length > 0 && ` · ${report.orphaned_files.length} orphaned files`}
            </p>
            {report.models.length > 0 && (
              <div className="space-y-1">
                {report.models.map((m) => (
                  <div key={m.model_id} className="flex items-center justify-between rounded-lg bg-surface-overlay px-3 py-2 text-xs">
                    <span className="truncate text-text-secondary">{m.model_name}</span>
                    <span className="shrink-0 font-mono text-text-muted">{formatBytes(m.size_bytes)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : null}
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="mb-4 text-base font-medium">{title}</h2>
      <div className="space-y-4">{children}</div>
    </section>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div>
        <p className="text-sm text-text-primary">{label}</p>
        {hint && <p className="text-xs text-text-muted">{hint}</p>}
      </div>
      {children}
    </div>
  );
}

// ── Cloud provider definitions (mirrors BUILTIN_PROVIDERS in Rust) ───

const CLOUD_PROVIDERS = [
  { prefix: "openai", display_name: "OpenAI", env_var: "OPENAI_API_KEY", group: "cloud" },
  { prefix: "anthropic", display_name: "Anthropic", env_var: "ANTHROPIC_API_KEY", group: "cloud" },
  { prefix: "gemini", display_name: "Google Gemini", env_var: "GEMINI_API_KEY", group: "cloud" },
  { prefix: "mistral", display_name: "Mistral", env_var: "MISTRAL_API_KEY", group: "cloud" },
  { prefix: "groq", display_name: "Groq", env_var: "GROQ_API_KEY", group: "cloud" },
  { prefix: "together", display_name: "Together AI", env_var: "TOGETHER_API_KEY", group: "cloud" },
  { prefix: "fireworks", display_name: "Fireworks AI", env_var: "FIREWORKS_API_KEY", group: "cloud" },
  { prefix: "deepseek", display_name: "DeepSeek", env_var: "DEEPSEEK_API_KEY", group: "cloud" },
  { prefix: "xai", display_name: "xAI (Grok)", env_var: "XAI_API_KEY", group: "cloud" },
  { prefix: "cerebras", display_name: "Cerebras", env_var: "CEREBRAS_API_KEY", group: "cloud" },
  { prefix: "sambanova", display_name: "SambaNova", env_var: "SAMBANOVA_API_KEY", group: "cloud" },
  { prefix: "perplexity", display_name: "Perplexity", env_var: "PERPLEXITY_API_KEY", group: "cloud" },
  { prefix: "cohere", display_name: "Cohere", env_var: "COHERE_API_KEY", group: "cloud" },
  { prefix: "ai21", display_name: "AI21 Labs", env_var: "AI21_API_KEY", group: "cloud" },
  { prefix: "openrouter", display_name: "OpenRouter", env_var: "OPENROUTER_API_KEY", group: "cloud" },
  { prefix: "nvidia", display_name: "NVIDIA NIM", env_var: "NVIDIA_API_KEY", group: "cloud" },
  { prefix: "azure", display_name: "Azure OpenAI", env_var: "AZURE_OPENAI_API_KEY", group: "cloud" },
  { prefix: "cloudflare", display_name: "Cloudflare Workers AI", env_var: "CLOUDFLARE_API_KEY", group: "cloud" },
  { prefix: "huggingface", display_name: "HuggingFace", env_var: "HF_API_KEY", group: "cloud" },
  { prefix: "replicate", display_name: "Replicate", env_var: "REPLICATE_API_KEY", group: "cloud" },
  { prefix: "lepton", display_name: "Lepton AI", env_var: "LEPTON_API_KEY", group: "cloud" },
  { prefix: "moonshot", display_name: "Moonshot AI", env_var: "MOONSHOT_API_KEY", group: "cloud" },
  { prefix: "dashscope", display_name: "Alibaba DashScope", env_var: "DASHSCOPE_API_KEY", group: "cloud" },
  { prefix: "zhipu", display_name: "Zhipu AI (GLM)", env_var: "ZHIPU_API_KEY", group: "cloud" },
  { prefix: "yi", display_name: "01.AI (Yi)", env_var: "YI_API_KEY", group: "cloud" },
  { prefix: "reka", display_name: "Reka AI", env_var: "REKA_API_KEY", group: "cloud" },
  { prefix: "baichuan", display_name: "Baichuan", env_var: "BAICHUAN_API_KEY", group: "cloud" },
  { prefix: "minimax", display_name: "Minimax", env_var: "MINIMAX_API_KEY", group: "cloud" },
  { prefix: "octoai", display_name: "OctoAI", env_var: "OCTOAI_API_KEY", group: "cloud" },
  { prefix: "anyscale", display_name: "Anyscale", env_var: "ANYSCALE_API_KEY", group: "cloud" },
] as const;

function CloudProviderKeys({
  apiKeys,
  onChange,
}: {
  apiKeys: Record<string, string>;
  onChange: (keys: Record<string, string>) => void;
}) {
  const [visibleKeys, setVisibleKeys] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");

  const toggleVisibility = (prefix: string) => {
    setVisibleKeys((prev) => {
      const next = new Set(prev);
      if (next.has(prefix)) next.delete(prefix);
      else next.add(prefix);
      return next;
    });
  };

  const setKey = (prefix: string, value: string) => {
    const next = { ...apiKeys };
    if (value) {
      next[prefix] = value;
    } else {
      delete next[prefix];
    }
    onChange(next);
  };

  const filtered = CLOUD_PROVIDERS.filter((p) =>
    !search || p.display_name.toLowerCase().includes(search.toLowerCase()) || p.prefix.includes(search.toLowerCase())
  );

  const configured = filtered.filter((p) => apiKeys[p.prefix]);
  const unconfigured = filtered.filter((p) => !apiKeys[p.prefix]);

  return (
    <div className="space-y-3">
      <input
        type="text"
        placeholder="Search providers..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500"
      />

      <div className="text-xs text-text-muted">
        <span className="font-medium text-paw-400">{Object.keys(apiKeys).length}</span> of {CLOUD_PROVIDERS.length} providers configured
      </div>

      {configured.length > 0 && (
        <div className="space-y-1">
          {configured.map((p) => (
            <ProviderKeyRow
              key={p.prefix}
              provider={p}
              value={apiKeys[p.prefix] ?? ""}
              visible={visibleKeys.has(p.prefix)}
              onToggleVisibility={() => toggleVisibility(p.prefix)}
              onChange={(v) => setKey(p.prefix, v)}
            />
          ))}
        </div>
      )}

      {unconfigured.length > 0 && (
        <div className="space-y-1">
          {unconfigured.map((p) => (
            <ProviderKeyRow
              key={p.prefix}
              provider={p}
              value=""
              visible={false}
              onToggleVisibility={() => toggleVisibility(p.prefix)}
              onChange={(v) => setKey(p.prefix, v)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ProviderKeyRow({
  provider,
  value,
  visible,
  onToggleVisibility,
  onChange,
}: {
  provider: (typeof CLOUD_PROVIDERS)[number];
  value: string;
  visible: boolean;
  onToggleVisibility: () => void;
  onChange: (value: string) => void;
}) {
  const hasKey = value.length > 0;
  return (
    <div className="flex items-center gap-2 rounded-lg bg-surface-overlay px-3 py-2">
      <div className="flex items-center gap-1.5 w-44 shrink-0">
        {hasKey ? (
          <Check size={12} className="text-green-400" />
        ) : (
          <Cloud size={12} className="text-text-muted" />
        )}
        <span className="text-xs font-medium text-text-primary truncate">{provider.display_name}</span>
      </div>
      <div className="flex-1 relative">
        <input
          type={visible ? "text" : "password"}
          placeholder={provider.env_var}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full rounded border border-border bg-surface px-2 py-1 font-mono text-xs outline-none focus:border-paw-500 pr-7"
        />
        {hasKey && (
          <button
            onClick={onToggleVisibility}
            className="absolute right-1.5 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary"
          >
            {visible ? <EyeOff size={12} /> : <Eye size={12} />}
          </button>
        )}
      </div>
      <span className="text-[10px] font-mono text-text-muted w-12 text-right shrink-0">{provider.prefix}/</span>
    </div>
  );
}
