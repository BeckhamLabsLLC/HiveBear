import { useEffect, useState } from "react";
import { getMeshConnectionStatus, type MeshConnectionStatus } from "../lib/invoke";

export interface MeshPulse {
  status: MeshConnectionStatus | null;
  loading: boolean;
  error: boolean;
}

/**
 * Polls the mesh connection status on an interval.
 * Errors are swallowed — the coordinator may be transiently unreachable,
 * and UI callers display idle/dim state when status is null.
 */
export function useMeshPulse(intervalMs = 15_000): MeshPulse {
  const [status, setStatus] = useState<MeshConnectionStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      try {
        const s = await getMeshConnectionStatus();
        if (cancelled) return;
        setStatus(s);
        setError(false);
      } catch {
        if (cancelled) return;
        setError(true);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs]);

  return { status, loading, error };
}
