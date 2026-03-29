import { useCallback, useState } from "react";
import type { BenchmarkResult } from "../types";
import { runBenchmark } from "../lib/invoke";

export function useBenchmark() {
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (durationSecs?: number) => {
    setRunning(true);
    setError(null);
    setResult(null);
    try { const r = await runBenchmark(durationSecs); setResult(r); }
    catch (e) { setError(String(e)); }
    finally { setRunning(false); }
  }, []);

  return { result, running, error, run };
}
