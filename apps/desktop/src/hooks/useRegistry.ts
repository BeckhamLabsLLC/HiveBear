import { useCallback, useEffect, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { DownloadProgress, InstalledInfo, ModelMetadata, SearchResult } from "../types";
import { installModel, listInstalled, removeModel, searchModels } from "../lib/invoke";

export function useModelSearch() {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = useCallback(async (query: string) => {
    if (!query.trim()) { setResults([]); return; }
    setLoading(true);
    setError(null);
    try {
      const r = await searchModels(query, 20);
      setResults(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  return { results, loading, error, search };
}

export function useInstalledModels() {
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try { setModels(await listInstalled()); } finally { setLoading(false); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  return { models, loading, refresh };
}

export function useModelInstall() {
  const [installing, setInstalling] = useState<string | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const install = useCallback(async (modelId: string, quant?: string): Promise<InstalledInfo | null> => {
    setInstalling(modelId);
    setProgress(null);
    setError(null);

    let unlisten: UnlistenFn | undefined;
    try {
      unlisten = await listen<DownloadProgress>("download-progress", (event) => setProgress(event.payload));
      const info = await installModel(modelId, quant);
      return info;
    } catch (e) {
      setError(String(e));
      return null;
    } finally {
      unlisten?.();
      setInstalling(null);
      setProgress(null);
    }
  }, []);

  return { installing, progress, error, install };
}

export function useModelRemove() {
  const [removing, setRemoving] = useState<string | null>(null);

  const remove = useCallback(async (modelId: string): Promise<boolean> => {
    setRemoving(modelId);
    try { await removeModel(modelId); return true; }
    catch { return false; }
    finally { setRemoving(null); }
  }, []);

  return { removing, remove };
}
