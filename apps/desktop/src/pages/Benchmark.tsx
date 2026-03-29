import { useState } from "react";
import { useBenchmark } from "../hooks/useBenchmark";
import { formatBytes, formatToksPerSec } from "../types";
import { Play, Loader } from "lucide-react";

export default function Benchmark() {
  const { result, running, error, run } = useBenchmark();
  const [duration, setDuration] = useState(30);

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-xl font-semibold">Benchmark</h1>
      <p className="text-sm text-text-secondary">
        Run an inference benchmark to measure your device's real performance.
      </p>

      <div className="flex items-end gap-4">
        <div>
          <label className="mb-1 block text-xs text-text-muted">Duration (seconds)</label>
          <input type="number" min={5} max={120} value={duration}
            onChange={(e) => setDuration(Number(e.target.value))} disabled={running}
            className="w-24 rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-paw-500 disabled:opacity-50" />
        </div>
        <button onClick={() => run(duration)} disabled={running}
          className="flex items-center gap-2 rounded-lg bg-paw-500 px-5 py-2 text-sm font-medium text-white hover:bg-paw-600 disabled:opacity-50">
          {running ? (<><Loader size={14} className="animate-spin" />Running...</>) : (<><Play size={14} />Run Benchmark</>)}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-danger/30 bg-danger/10 px-4 py-2 text-sm text-danger">{error}</div>
      )}

      {result && (
        <div className="rounded-xl border border-border bg-surface-raised p-6">
          <h2 className="mb-4 text-base font-medium">Results</h2>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Stat label="Tokens/sec" value={formatToksPerSec(result.tokens_per_sec)} highlight />
            <Stat label="Time to First Token" value={`${result.time_to_first_token_ms} ms`} />
            <Stat label="Tokens Generated" value={String(result.tokens_generated)} />
            <Stat label="Total Duration" value={`${(result.total_duration_ms / 1000).toFixed(1)}s`} />
            <Stat label="Peak Memory" value={formatBytes(result.peak_memory_bytes)} />
            <Stat label="CPU Utilization" value={`${(result.cpu_utilization * 100).toFixed(0)}%`} />
            <Stat label="GPU Utilization" value={result.gpu_utilization != null ? `${(result.gpu_utilization * 100).toFixed(0)}%` : "N/A"} />
            <Stat label="Model Used" value={result.model_used} />
          </div>
        </div>
      )}

      {/* Reference benchmarks for comparison */}
      <section className="rounded-xl border border-border bg-surface-raised p-6">
        <h2 className="mb-1 text-base font-medium">Reference Benchmarks</h2>
        <p className="mb-4 text-xs text-text-muted">
          Community-reported performance for common hardware (Llama 3.1 8B Q4_K_M, llama.cpp).
        </p>
        <div className="overflow-hidden rounded-lg border border-border">
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
              {referenceBenchmarks.map((ref_, i) => (
                <tr key={i} className={
                  result && Math.abs(result.tokens_per_sec - ref_.toksPerSec) < 5
                    ? "bg-paw-500/5"
                    : ""
                }>
                  <td className="px-3 py-2">{ref_.hardware}</td>
                  <td className="px-3 py-2 font-mono">{ref_.ram}</td>
                  <td className="px-3 py-2">{ref_.gpu}</td>
                  <td className="px-3 py-2 text-right font-mono">{ref_.toksPerSec.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {result && (
          <p className="mt-3 text-xs text-text-muted">
            Your result: <span className="font-mono text-paw-400">{result.tokens_per_sec.toFixed(1)} tok/s</span>
            {" — "}
            {result.tokens_per_sec >= 30
              ? "Excellent! Well above average."
              : result.tokens_per_sec >= 15
              ? "Good performance for most models."
              : result.tokens_per_sec >= 5
              ? "Usable, but larger models may be slow."
              : "Consider using smaller quantizations (Q3, Q2) or smaller models."}
          </p>
        )}
      </section>

      {!result && !running && (
        <div className="rounded-xl border border-dashed border-border p-8 text-center text-sm text-text-muted">
          Click "Run Benchmark" to measure inference speed on your hardware.
        </div>
      )}
    </div>
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
      <p className={`mt-0.5 font-mono text-sm ${highlight ? "text-paw-400" : "text-text-primary"}`}>{value}</p>
    </div>
  );
}
