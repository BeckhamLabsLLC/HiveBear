import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useConfig, useStorageReport } from "../hooks/useConfig";
import { formatBytes } from "../types";
import type { Config } from "../types";
import { Card, Button, Toggle, Surface, Badge } from "../components/ui";
import { Save, Loader, Eye, EyeOff, Cloud, Check, ChevronDown } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

const SECTIONS = ["Inference", "Discovery", "Mesh", "Cloud", "Storage"] as const;
type SectionId = typeof SECTIONS[number];

export default function Settings() {
  const { config, loading, saving, error, save } = useConfig();
  const { report, loading: storageLoading } = useStorageReport();
  const [form, setForm] = useState<Config | null>(null);
  const [activeSection, setActiveSection] = useState<SectionId>("Inference");
  const sectionRefs = useRef<Record<string, HTMLElement | null>>({});

  useEffect(() => { if (config) setForm({ ...config }); }, [config]);

  const isDirty = useMemo(() => {
    if (!config || !form) return false;
    return JSON.stringify(config) !== JSON.stringify(form);
  }, [config, form]);

  const handleSave = useCallback(() => { if (form) save(form); }, [form, save]);
  const handleDiscard = useCallback(() => { if (config) setForm({ ...config }); }, [config]);

  const scrollToSection = useCallback((id: SectionId) => {
    setActiveSection(id);
    sectionRefs.current[id]?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  if (loading || !form) {
    return (
      <Surface>
        <div className="flex h-full items-center justify-center text-text-muted text-sm">Loading configuration...</div>
      </Surface>
    );
  }

  return (
    <Surface>
      <div className="flex gap-8">
        {/* Left mini-nav (hidden on mobile) */}
        <nav className="sticky top-6 hidden w-32 shrink-0 flex-col gap-0.5 self-start sm:flex">
          {SECTIONS.map((s) => (
            <button
              key={s}
              onClick={() => scrollToSection(s)}
              className={[
                "rounded-[var(--radius-md)] px-3 py-1.5 text-left text-xs interactive-hover",
                activeSection === s
                  ? "bg-surface-overlay text-text-primary font-medium"
                  : "text-text-muted hover:text-text-secondary hover:bg-surface-raised",
              ].join(" ")}
            >
              {s}
            </button>
          ))}
        </nav>

        {/* Settings content */}
        <div className="min-w-0 flex-1 space-y-6 sm:space-y-8">
          <h1 className="text-xl font-semibold">Settings</h1>

          {error && <div className="rounded-[var(--radius-md)] border border-danger/30 bg-danger/10 px-4 py-2 text-sm text-danger">{error}</div>}

          {/* Inference */}
          <SettingsSection id="Inference" refs={sectionRefs}>
            <Field label="Default Context Length" hint="Tokens per conversation turn">
              <input type="number" min={512} max={131072} step={512} value={form.default_context_length}
                onChange={(e) => setForm({ ...form, default_context_length: Number(e.target.value) })}
                className="w-32 rounded-[var(--radius-md)] border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500 focus:ring-2 focus:ring-paw-500/20" />
            </Field>
            <Field label="Min Tokens/sec" hint="Models below this speed are excluded">
              <input type="number" min={1} max={100} step={0.5} value={form.min_tokens_per_sec}
                onChange={(e) => setForm({ ...form, min_tokens_per_sec: Number(e.target.value) })}
                className="w-32 rounded-[var(--radius-md)] border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500 focus:ring-2 focus:ring-paw-500/20" />
            </Field>
            <Field label="Max Memory Usage" hint="Fraction of available memory (0.0 - 1.0)">
              <div className="flex items-center gap-3">
                <input type="range" min={0.1} max={1.0} step={0.05} value={form.max_memory_usage}
                  onChange={(e) => setForm({ ...form, max_memory_usage: Number(e.target.value) })}
                  className="w-32" />
                <span className="w-10 text-right font-mono text-xs text-text-muted">
                  {(form.max_memory_usage * 100).toFixed(0)}%
                </span>
              </div>
            </Field>
          </SettingsSection>

          {/* Discovery */}
          <SettingsSection id="Discovery" refs={sectionRefs}>
            <Field label="Top N Recommendations">
              <input type="number" min={1} max={50} value={form.top_n_recommendations}
                onChange={(e) => setForm({ ...form, top_n_recommendations: Number(e.target.value) })}
                className="w-32 rounded-[var(--radius-md)] border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500 focus:ring-2 focus:ring-paw-500/20" />
            </Field>
            <Field label="Models Directory">
              <input type="text" value={form.models_dir}
                onChange={(e) => setForm({ ...form, models_dir: e.target.value })}
                className="w-full max-w-md rounded-[var(--radius-md)] border border-border bg-surface px-3 py-2 font-mono text-sm outline-none focus:border-paw-500 focus:ring-2 focus:ring-paw-500/20" />
            </Field>
            <Field label="Share Benchmarks" hint="Anonymously contribute to community database">
              <Toggle checked={form.share_benchmarks} onChange={(v) => setForm({ ...form, share_benchmarks: v })} />
            </Field>
          </SettingsSection>

          {/* Mesh */}
          <SettingsSection id="Mesh" refs={sectionRefs}>
            <Field label="Enable Mesh" hint="Join P2P network to share distributed compute">
              <Toggle checked={form.mesh_enabled ?? false} onChange={(v) => setForm({ ...form, mesh_enabled: v })} />
            </Field>
            <Field label="Port" hint="QUIC transport port (1024-65535)">
              <input type="number" min={1024} max={65535} value={form.mesh_port ?? 7878}
                onChange={(e) => setForm({ ...form, mesh_port: Number(e.target.value) })}
                className="w-32 rounded-[var(--radius-md)] border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500 focus:ring-2 focus:ring-paw-500/20" />
            </Field>
            <Field label="Coordination Server">
              <input type="text" value={form.mesh_coordination_server ?? ""}
                onChange={(e) => setForm({ ...form, mesh_coordination_server: e.target.value })}
                className="w-full max-w-md rounded-[var(--radius-md)] border border-border bg-surface px-3 py-2 font-mono text-sm outline-none focus:border-paw-500 focus:ring-2 focus:ring-paw-500/20" />
            </Field>
            <Field label="Max Contribution" hint="Resources to share (0-100%)">
              <div className="flex items-center gap-3">
                <input type="range" min={0} max={100} value={(form.mesh_max_contribution ?? 0.8) * 100}
                  onChange={(e) => setForm({ ...form, mesh_max_contribution: Number(e.target.value) / 100 })}
                  className="w-32" />
                <span className="w-10 text-right font-mono text-xs text-text-muted">
                  {((form.mesh_max_contribution ?? 0.8) * 100).toFixed(0)}%
                </span>
              </div>
            </Field>
            <Field label="Verification Rate" hint="Fraction of tokens to spot-check">
              <div className="flex items-center gap-3">
                <input type="range" min={0} max={100} value={(form.mesh_verification_rate ?? 0.05) * 100}
                  onChange={(e) => setForm({ ...form, mesh_verification_rate: Number(e.target.value) / 100 })}
                  className="w-32" />
                <span className="w-10 text-right font-mono text-xs text-text-muted">
                  {((form.mesh_verification_rate ?? 0.05) * 100).toFixed(0)}%
                </span>
              </div>
            </Field>
          </SettingsSection>

          {/* Cloud */}
          <SettingsSection id="Cloud" refs={sectionRefs}>
            <CloudProviderKeys
              apiKeys={form.cloud_api_keys ?? {}}
              onChange={(keys) => setForm({ ...form, cloud_api_keys: keys })}
            />
            <Field label="Cloud Fallback" hint="Fall back to cloud when local inference fails">
              <Toggle checked={form.cloud_fallback ?? false} onChange={(v) => setForm({ ...form, cloud_fallback: v })} />
            </Field>
          </SettingsSection>

          {/* Storage */}
          <SettingsSection id="Storage" refs={sectionRefs}>
            {storageLoading ? (
              <p className="text-sm text-text-muted">Loading storage info...</p>
            ) : report ? (
              <div className="space-y-3">
                <div className="flex items-center gap-3 text-sm text-text-secondary">
                  <span>Total: <span className="font-mono font-medium">{formatBytes(report.total_bytes)}</span></span>
                  <Badge>{report.models.length} model{report.models.length !== 1 && "s"}</Badge>
                  {report.partial_downloads.length > 0 && <Badge variant="warning">{report.partial_downloads.length} partial</Badge>}
                  {report.orphaned_files.length > 0 && <Badge variant="danger">{report.orphaned_files.length} orphaned</Badge>}
                </div>
                {report.models.length > 0 && (
                  <div className="space-y-1">
                    {report.models.map((m) => (
                      <div key={m.model_id} className="flex items-center justify-between rounded-[var(--radius-md)] bg-surface-overlay px-3 py-2 text-xs">
                        <span className="truncate text-text-secondary">{m.model_name}</span>
                        <span className="shrink-0 font-mono text-text-muted">{formatBytes(m.size_bytes)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : null}
          </SettingsSection>
        </div>
      </div>

      {/* Floating save bar */}
      <AnimatePresence>
        {isDirty && (
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 20, opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 400 }}
            className="fixed bottom-[calc(56px+env(safe-area-inset-bottom)+12px)] sm:bottom-6 left-1/2 z-50 flex -translate-x-1/2 items-center gap-3 rounded-[var(--radius-xl)] border border-border bg-surface-raised px-4 py-2.5 sm:px-6 sm:py-3 shadow-[var(--shadow-floating)]"
          >
            <span className="text-sm text-text-secondary">Unsaved changes</span>
            <Button variant="ghost" size="sm" onClick={handleDiscard}>Discard</Button>
            <Button size="sm" onClick={handleSave} disabled={saving}>
              {saving ? <Loader size={14} className="animate-spin" /> : <Save size={14} />}
              Save
            </Button>
          </motion.div>
        )}
      </AnimatePresence>
    </Surface>
  );
}

function SettingsSection({ id, children, refs }: {
  id: SectionId;
  children: React.ReactNode;
  refs: React.MutableRefObject<Record<string, HTMLElement | null>>;
}) {
  return (
    <Card ref={(el) => { refs.current[id] = el; }}>
      <h2 className="mb-4 text-sm font-semibold text-text-primary">{id}</h2>
      <div className="space-y-4">{children}</div>
    </Card>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
      <div className="shrink-0">
        <p className="text-sm text-text-primary">{label}</p>
        {hint && <p className="text-xs text-text-muted max-w-[280px]">{hint}</p>}
      </div>
      {children}
    </div>
  );
}

// ── Cloud Providers ─────────────────────────────────────────────────

const POPULAR_PROVIDERS = ["openai", "anthropic", "gemini", "mistral", "groq"];

const CLOUD_PROVIDERS = [
  { prefix: "openai", display_name: "OpenAI", env_var: "OPENAI_API_KEY" },
  { prefix: "anthropic", display_name: "Anthropic", env_var: "ANTHROPIC_API_KEY" },
  { prefix: "gemini", display_name: "Google Gemini", env_var: "GEMINI_API_KEY" },
  { prefix: "mistral", display_name: "Mistral", env_var: "MISTRAL_API_KEY" },
  { prefix: "groq", display_name: "Groq", env_var: "GROQ_API_KEY" },
  { prefix: "together", display_name: "Together AI", env_var: "TOGETHER_API_KEY" },
  { prefix: "fireworks", display_name: "Fireworks AI", env_var: "FIREWORKS_API_KEY" },
  { prefix: "deepseek", display_name: "DeepSeek", env_var: "DEEPSEEK_API_KEY" },
  { prefix: "xai", display_name: "xAI (Grok)", env_var: "XAI_API_KEY" },
  { prefix: "cerebras", display_name: "Cerebras", env_var: "CEREBRAS_API_KEY" },
  { prefix: "sambanova", display_name: "SambaNova", env_var: "SAMBANOVA_API_KEY" },
  { prefix: "perplexity", display_name: "Perplexity", env_var: "PERPLEXITY_API_KEY" },
  { prefix: "cohere", display_name: "Cohere", env_var: "COHERE_API_KEY" },
  { prefix: "ai21", display_name: "AI21 Labs", env_var: "AI21_API_KEY" },
  { prefix: "openrouter", display_name: "OpenRouter", env_var: "OPENROUTER_API_KEY" },
  { prefix: "nvidia", display_name: "NVIDIA NIM", env_var: "NVIDIA_API_KEY" },
  { prefix: "azure", display_name: "Azure OpenAI", env_var: "AZURE_OPENAI_API_KEY" },
  { prefix: "cloudflare", display_name: "Cloudflare Workers AI", env_var: "CLOUDFLARE_API_KEY" },
  { prefix: "huggingface", display_name: "HuggingFace", env_var: "HF_API_KEY" },
  { prefix: "replicate", display_name: "Replicate", env_var: "REPLICATE_API_KEY" },
  { prefix: "lepton", display_name: "Lepton AI", env_var: "LEPTON_API_KEY" },
  { prefix: "moonshot", display_name: "Moonshot AI", env_var: "MOONSHOT_API_KEY" },
  { prefix: "dashscope", display_name: "Alibaba DashScope", env_var: "DASHSCOPE_API_KEY" },
  { prefix: "zhipu", display_name: "Zhipu AI (GLM)", env_var: "ZHIPU_API_KEY" },
  { prefix: "yi", display_name: "01.AI (Yi)", env_var: "YI_API_KEY" },
  { prefix: "reka", display_name: "Reka AI", env_var: "REKA_API_KEY" },
  { prefix: "baichuan", display_name: "Baichuan", env_var: "BAICHUAN_API_KEY" },
  { prefix: "minimax", display_name: "Minimax", env_var: "MINIMAX_API_KEY" },
  { prefix: "octoai", display_name: "OctoAI", env_var: "OCTOAI_API_KEY" },
  { prefix: "anyscale", display_name: "Anyscale", env_var: "ANYSCALE_API_KEY" },
] as const;

function CloudProviderKeys({
  apiKeys,
  onChange,
}: {
  apiKeys: Record<string, string>;
  onChange: (keys: Record<string, string>) => void;
}) {
  const [visibleKeys, setVisibleKeys] = useState<Set<string>>(new Set());
  const [showAll, setShowAll] = useState(false);

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
    if (value) next[prefix] = value;
    else delete next[prefix];
    onChange(next);
  };

  const configured = CLOUD_PROVIDERS.filter((p) => apiKeys[p.prefix]);
  const popular = CLOUD_PROVIDERS.filter((p) => POPULAR_PROVIDERS.includes(p.prefix) && !apiKeys[p.prefix]);
  const rest = CLOUD_PROVIDERS.filter((p) => !POPULAR_PROVIDERS.includes(p.prefix) && !apiKeys[p.prefix]);

  const configuredCount = Object.keys(apiKeys).length;

  return (
    <div className="space-y-3">
      <div className="text-xs text-text-muted">
        <span className="font-medium text-paw-400">{configuredCount}</span> of {CLOUD_PROVIDERS.length} providers configured
      </div>

      {/* Configured (always visible) */}
      {configured.length > 0 && (
        <div className="space-y-1">
          {configured.map((p) => (
            <ProviderKeyRow key={p.prefix} provider={p} value={apiKeys[p.prefix] ?? ""} visible={visibleKeys.has(p.prefix)}
              onToggleVisibility={() => toggleVisibility(p.prefix)} onChange={(v) => setKey(p.prefix, v)} />
          ))}
        </div>
      )}

      {/* Popular (visible by default) */}
      {popular.length > 0 && (
        <div className="space-y-1">
          <p className="text-[11px] font-medium uppercase tracking-wider text-text-muted pt-2">Add Provider</p>
          {popular.map((p) => (
            <ProviderKeyRow key={p.prefix} provider={p} value="" visible={false}
              onToggleVisibility={() => toggleVisibility(p.prefix)} onChange={(v) => setKey(p.prefix, v)} />
          ))}
        </div>
      )}

      {/* Rest (hidden by default) */}
      {rest.length > 0 && (
        <div>
          <button
            onClick={() => setShowAll(!showAll)}
            className="interactive-hover flex items-center gap-1.5 rounded-[var(--radius-md)] px-2 py-1.5 text-xs text-text-muted hover:text-text-secondary"
          >
            <ChevronDown size={12} className={`transition-transform ${showAll ? "rotate-180" : ""}`} />
            {showAll ? "Hide" : "Show"} {rest.length} more providers
          </button>
          {showAll && (
            <div className="mt-1 space-y-1 animate-[fade-in]">
              {rest.map((p) => (
                <ProviderKeyRow key={p.prefix} provider={p} value={apiKeys[p.prefix] ?? ""} visible={visibleKeys.has(p.prefix)}
                  onToggleVisibility={() => toggleVisibility(p.prefix)} onChange={(v) => setKey(p.prefix, v)} />
              ))}
            </div>
          )}
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
  provider: { prefix: string; display_name: string; env_var: string };
  value: string;
  visible: boolean;
  onToggleVisibility: () => void;
  onChange: (value: string) => void;
}) {
  const hasKey = value.length > 0;
  return (
    <div className="flex items-center gap-2 rounded-[var(--radius-md)] bg-surface-overlay px-3 py-2">
      <div className="flex items-center gap-1.5 w-40 shrink-0">
        {hasKey ? (
          <Check size={12} className="text-success" />
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
          className="w-full rounded-[var(--radius-sm)] border border-border bg-surface px-2 py-1 font-mono text-xs outline-none focus:border-paw-500 focus:ring-2 focus:ring-paw-500/20 pr-7"
        />
        {hasKey && (
          <button onClick={onToggleVisibility}
            className="absolute right-1.5 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary">
            {visible ? <EyeOff size={12} /> : <Eye size={12} />}
          </button>
        )}
      </div>
      <span className="text-[10px] font-mono text-text-muted w-12 text-right shrink-0">{provider.prefix}/</span>
    </div>
  );
}
