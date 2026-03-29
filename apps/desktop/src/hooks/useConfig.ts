import { useCallback, useEffect, useState } from "react";
import type { Config, StorageReport } from "../types";
import { getConfig, getStorageReport, saveConfig } from "../lib/invoke";

export function useConfig() {
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getConfig()
      .then(setConfig)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const save = useCallback(async (updated: Config) => {
    setSaving(true);
    setError(null);
    try { await saveConfig(updated); setConfig(updated); }
    catch (e) { setError(String(e)); }
    finally { setSaving(false); }
  }, []);

  return { config, loading, saving, error, save };
}

export function useStorageReport() {
  const [report, setReport] = useState<StorageReport | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try { setReport(await getStorageReport()); } finally { setLoading(false); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  return { report, loading, refresh };
}
