import { useCallback, useState } from "react";
import type { BenchmarkResult, ModelBenchmarkResult } from "../types";
import { runBenchmark, runModelBenchmark, shareBenchmark } from "../lib/invoke";
import { notify } from "../components/Toast";

/** The synthetic CPU estimate, for when no model is installed. */
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

/** A real benchmark of an installed model, and sharing it to the leaderboard. */
export function useModelBenchmark() {
  const [bench, setBench] = useState<ModelBenchmarkResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sharing, setSharing] = useState(false);
  const [shared, setShared] = useState(false);

  const run = useCallback(async (modelId: string) => {
    setRunning(true);
    setError(null);
    setBench(null);
    setShared(false);
    try { setBench(await runModelBenchmark(modelId)); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setRunning(false); }
  }, []);

  // Errors are toasted by invoke(); this only records success.
  const share = useCallback(async () => {
    if (!bench) return;
    setSharing(true);
    try {
      await shareBenchmark(bench);
      setShared(true);
      notify("Shared to the community leaderboard.", "success");
    } catch { /* toasted */ }
    finally { setSharing(false); }
  }, [bench]);

  return { bench, running, error, run, share, sharing, shared };
}
