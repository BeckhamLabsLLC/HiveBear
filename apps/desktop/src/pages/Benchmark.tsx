import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { open as openExternal } from "@tauri-apps/plugin-shell";
import { useBenchmark, useModelBenchmark } from "../hooks/useBenchmark";
import { useInstalledModels } from "../hooks/useRegistry";
import { formatBytes, formatToksPerSec } from "../types";
import type { BenchmarkResult } from "../types";
import { Play, Loader, Gauge, Share2, Check, ExternalLink, Download, UserPlus } from "lucide-react";
import { Card, Button, Surface, EmptyState, Badge } from "../components/ui";

const LEADERBOARD_URL = "https://hivebear.com/benchmarks";
const CLAIM_URL = "https://hivebear.com/register?next=/benchmarks";

export default function Benchmark() {
  const navigate = useNavigate();
  const { models: installed, loading: installedLoading } = useInstalledModels();
  const real = useModelBenchmark();
  const estimate = useBenchmark();
  const [modelId, setModelId] = useState<string>("");
  const [duration, setDuration] = useState(30);

  useEffect(() => {
    if (!modelId && installed.length > 0) setModelId(installed[0].id);
  }, [installed, modelId]);

  const hasModels = installed.length > 0;
  const busy = real.running || estimate.running;
  const shown: BenchmarkResult | null = real.bench?.result ?? (hasModels ? null : estimate.result);

  return (
    <Surface>
      <div className="space-y-6">
        <div>
          <h1 className="text-xl font-semibold">Benchmark</h1>
          <p className="mt-1 text-sm text-text-secondary">
            Measure how fast a model really runs on this machine, then put it on the community leaderboard.
          </p>
        </div>

        {installedLoading ? (
          <p className="text-xs text-text-muted">Looking for installed models…</p>
        ) : hasModels ? (
          /* Real benchmark of an installed model */
          <Card padding="lg">
            <div className="flex flex-wrap items-end gap-4">
              <div className="min-w-48 flex-1">
                <label htmlFor="bench-model" className="mb-1 block text-xs text-text-muted">Model</label>
                <select
                  id="bench-model"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  disabled={busy}
                  className="w-full rounded-[var(--radius-md)] border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500 focus:ring-2 focus:ring-paw-500/20 disabled:opacity-50"
                >
                  {installed.map((m) => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                </select>
              </div>
              <Button onClick={() => real.run(modelId)} disabled={busy || !modelId}>
                {real.running
                  ? <><Loader size={14} className="animate-spin" />Running…</>
                  : <><Play size={14} />Run benchmark</>}
              </Button>
            </div>
            {real.running && (
              <p className="mt-3 flex items-center gap-2 text-xs text-text-muted">
                <Loader size={12} className="animate-spin" aria-hidden />
                Loading the model and generating tokens. This usually takes 30–60 seconds.
              </p>
            )}
          </Card>
        ) : (
          /* No model: only a clearly labelled estimate is possible */
          <Card padding="lg">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold">Quick estimate (no model installed)</h2>
              <Badge variant="default">Estimate</Badge>
            </div>
            <p className="mt-1 text-xs text-text-muted">
              A CPU math test that guesses speed. It is not a real model run, so it can't go on the
              leaderboard. Install a model for a real benchmark.
            </p>
            <div className="mt-4 flex flex-wrap items-end gap-4">
              <div>
                <label htmlFor="bench-duration" className="mb-1 block text-xs text-text-muted">Duration (seconds)</label>
                <input id="bench-duration" type="number" min={5} max={120} value={duration}
                  onChange={(e) => setDuration(Number(e.target.value))} disabled={busy}
                  className="w-24 rounded-[var(--radius-md)] border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500 focus:ring-2 focus:ring-paw-500/20 disabled:opacity-50" />
              </div>
              <Button variant="secondary" onClick={() => estimate.run(duration)} disabled={busy}>
                {estimate.running ? <><Loader size={14} className="animate-spin" />Estimating…</> : <><Gauge size={14} />Run quick estimate</>}
              </Button>
              <Button onClick={() => navigate("/models")}>
                <Download size={14} />Install a model
              </Button>
            </div>
          </Card>
        )}

        {(real.error || estimate.error) && (
          <div className="rounded-[var(--radius-md)] border border-danger/30 bg-danger/10 px-4 py-2 text-sm text-danger">
            {real.error ?? estimate.error}
          </div>
        )}

        {real.bench && (
          <Card padding="lg">
            <div className="mb-4 flex flex-wrap items-center gap-2">
              <h2 className="text-sm font-semibold">Results</h2>
              <Badge variant="accent">{real.bench.quantization}</Badge>
              <Badge variant="default">{real.bench.engine}</Badge>
            </div>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <Stat label="Generation" value={formatToksPerSec(real.bench.result.tokens_per_sec)} highlight />
              <Stat label="Time to First Token" value={`${real.bench.result.time_to_first_token_ms} ms`} />
              <Stat
                label="Prompt Processing"
                value={real.bench.result.prompt_eval_tokens_per_sec != null
                  ? formatToksPerSec(real.bench.result.prompt_eval_tokens_per_sec) : "N/A"}
              />
              <Stat
                label="Peak Memory"
                value={real.bench.result.peak_memory_bytes > 0 ? formatBytes(real.bench.result.peak_memory_bytes) : "N/A"}
              />
              <Stat label="Tokens Generated" value={String(real.bench.result.tokens_generated)} />
              <Stat label="Total Duration" value={`${(real.bench.result.total_duration_ms / 1000).toFixed(1)}s`} />
              <Stat label="Model" value={real.bench.model_id} />
            </div>

            <div className="mt-5 border-t border-border pt-4">
              {real.shared ? (
                <div className="space-y-3">
                  <p className="flex items-center gap-2 text-sm text-success">
                    <Check size={14} aria-hidden /> Shared to the community leaderboard.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <Button size="sm" onClick={() => void openExternal(LEADERBOARD_URL)}>
                      <ExternalLink size={12} />See the leaderboard
                    </Button>
                    <Button size="sm" variant="secondary" onClick={() => void openExternal(CLAIM_URL)}>
                      <UserPlus size={12} />Claim your machine with a free account
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-wrap items-center gap-3">
                  <Button onClick={() => void real.share()} disabled={real.sharing}>
                    {real.sharing ? <Loader size={14} className="animate-spin" /> : <Share2 size={14} />}
                    Share to leaderboard
                  </Button>
                  <p className="text-xs text-text-muted">
                    Anonymous: model, speed and a hardware class (GPU type, RAM bucket). No name, no files, no prompts.
                  </p>
                </div>
              )}
            </div>
          </Card>
        )}

        {!hasModels && estimate.result && (
          <Card padding="lg">
            <div className="mb-4 flex items-center gap-2">
              <h2 className="text-sm font-semibold">Estimate</h2>
              <Badge variant="default">Not a real model run</Badge>
            </div>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <Stat label="Estimated tok/s (7B Q4)" value={formatToksPerSec(estimate.result.tokens_per_sec)} highlight />
              <Stat label="Total Duration" value={`${(estimate.result.total_duration_ms / 1000).toFixed(1)}s`} />
              <Stat label="CPU Utilization" value={`${estimate.result.cpu_utilization.toFixed(0)}% of one core`} />
            </div>
          </Card>
        )}

        {/* Reference benchmarks */}
        <Card>
          <h2 className="mb-1 text-sm font-semibold">Reference Benchmarks</h2>
          <p className="mb-4 text-xs text-text-muted">
            Typical performance (Llama 3.1 8B Q4_K_M, llama.cpp). Compare like with like: a smaller
            model runs faster.
          </p>
          <div className="overflow-hidden rounded-[var(--radius-md)] border border-border">
            <table className="w-full text-left text-xs">
              <thead className="bg-surface-overlay text-text-muted">
                <tr>
                  <th className="px-3 py-2 font-medium">Hardware</th>
                  <th className="px-3 py-2 font-medium">RAM</th>
                  <th className="px-3 py-2 font-medium">GPU</th>
                  <th className="px-3 py-2 font-medium text-right">Tokens/sec</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border text-text-secondary">
                {referenceBenchmarks.map((ref_, i) => {
                  const isClose = shown && Math.abs(shown.tokens_per_sec - ref_.toksPerSec) < 5;
                  return (
                    <tr key={i} className={isClose ? "bg-paw-500/5" : ""}>
                      <td className="px-3 py-2">{ref_.hardware}</td>
                      <td className="px-3 py-2 font-mono">{ref_.ram}</td>
                      <td className="px-3 py-2">{ref_.gpu}</td>
                      <td className="px-3 py-2 text-right font-mono">
                        {ref_.toksPerSec.toFixed(1)}
                        {isClose && <span className="ml-1.5 text-paw-400">&#8592; you</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {shown && (
            <p className="mt-3 text-xs text-text-muted">
              Your result: <span className="font-mono text-paw-400">{shown.tokens_per_sec.toFixed(1)} tok/s</span>
              {" — "}
              {shown.tokens_per_sec >= 30
                ? "Excellent! Well above average."
                : shown.tokens_per_sec >= 15
                ? "Good performance for most models."
                : shown.tokens_per_sec >= 5
                ? "Usable, but larger models may be slow."
                : "Consider using smaller quantizations or smaller models."}
            </p>
          )}
        </Card>

        {!shown && !busy && (
          <EmptyState
            icon={<Gauge size={24} />}
            title="See how fast your hardware runs AI"
            description="Run a 30-second benchmark, then share it to see how your machine compares."
          />
        )}
      </div>
    </Surface>
  );
}

const referenceBenchmarks = [
  { hardware: "Apple M3 Pro", ram: "36 GB", gpu: "Metal (18-core)", toksPerSec: 42.0 },
  { hardware: "Apple M2", ram: "16 GB", gpu: "Metal (10-core)", toksPerSec: 28.5 },
  { hardware: "Apple M1", ram: "16 GB", gpu: "Metal (8-core)", toksPerSec: 18.2 },
  { hardware: "RTX 4090", ram: "64 GB", gpu: "CUDA (24 GB)", toksPerSec: 95.0 },
  { hardware: "RTX 3080", ram: "32 GB", gpu: "CUDA (10 GB)", toksPerSec: 52.0 },
  { hardware: "RTX 3060", ram: "16 GB", gpu: "CUDA (12 GB)", toksPerSec: 35.0 },
  { hardware: "Intel i7-13700K", ram: "32 GB", gpu: "CPU only", toksPerSec: 8.5 },
  { hardware: "AMD Ryzen 7 5800X", ram: "32 GB", gpu: "CPU only", toksPerSec: 7.2 },
  { hardware: "Raspberry Pi 5", ram: "8 GB", gpu: "CPU only", toksPerSec: 1.8 },
];

function Stat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div>
      <p className="text-xs text-text-muted">{label}</p>
      <p className={`mt-0.5 truncate font-mono text-sm ${highlight ? "text-paw-400 font-semibold" : "text-text-primary"}`}>{value}</p>
    </div>
  );
}
