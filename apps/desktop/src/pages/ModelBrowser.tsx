import { useCallback, useState } from "react";
import { useInstalledModels, useModelInstall, useModelRemove, useModelSearch } from "../hooks/useRegistry";
import ModelCard from "../components/ModelCard";
import DownloadProgress from "../components/DownloadProgress";
import { Search as SearchIcon } from "lucide-react";

export default function ModelBrowser() {
  const [query, setQuery] = useState("");
  const { results, loading: searching, search } = useModelSearch();
  const { models: installed, refresh } = useInstalledModels();
  const { installing, progress, error: installError, install } = useModelInstall();
  const { removing, remove } = useModelRemove();
  const [tab, setTab] = useState<"search" | "installed">("search");

  const handleSearch = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    search(query);
  }, [query, search]);

  const handleInstall = useCallback(async (modelId: string) => {
    const result = await install(modelId);
    if (result) refresh();
  }, [install, refresh]);

  const handleRemove = useCallback(async (modelId: string) => {
    const ok = await remove(modelId);
    if (ok) refresh();
  }, [remove, refresh]);

  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 space-y-4 border-b border-border p-6 pb-4">
        <h1 className="text-xl font-semibold">Models</h1>
        <div className="flex gap-1 rounded-lg bg-surface-overlay p-0.5">
          {(["search", "installed"] as const).map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`flex-1 rounded-md px-3 py-1.5 text-sm capitalize transition-colors ${
                tab === t ? "bg-surface-raised text-text-primary" : "text-text-muted hover:text-text-secondary"
              }`}>
              {t} {t === "installed" && `(${installed.length})`}
            </button>
          ))}
        </div>
        {tab === "search" && (
          <form onSubmit={handleSearch} className="flex gap-2">
            <div className="relative flex-1">
              <SearchIcon size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
              <input type="text" value={query} onChange={(e) => setQuery(e.target.value)}
                placeholder="Search models (e.g. llama, codestral, phi)..."
                className="w-full rounded-lg border border-border bg-surface py-2 pl-9 pr-3 text-sm outline-none placeholder:text-text-muted focus:border-paw-500" />
            </div>
            <button type="submit" className="rounded-lg bg-paw-500 px-4 py-2 text-sm font-medium text-white hover:bg-paw-600">
              Search
            </button>
          </form>
        )}
      </div>

      {installing && (
        <div className="shrink-0 px-6 pt-4">
          <DownloadProgress modelId={installing} progress={progress} />
        </div>
      )}

      {installError && (
        <div className="mx-6 mt-4 rounded-lg border border-danger/30 bg-danger/10 px-4 py-2 text-sm text-danger">
          {installError}
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-6 pt-4">
        {tab === "search" ? (
          searching ? (
            <p className="text-sm text-text-muted">Searching...</p>
          ) : results.length === 0 ? (
            <p className="text-sm text-text-muted">{query ? "No results found." : "Search for models to get started."}</p>
          ) : (
            <div className="grid gap-3">
              {results.map((r) => (
                <ModelCard key={r.metadata.id} result={r} installing={installing}
                  removing={removing} onInstall={handleInstall} onRemove={handleRemove} />
              ))}
            </div>
          )
        ) : installed.length === 0 ? (
          <p className="text-sm text-text-muted">No models installed yet. Search and install one to get started.</p>
        ) : (
          <div className="grid gap-3">
            {installed.map((m) => (
              <ModelCard key={m.id} result={{ metadata: m, is_installed: true, compatibility_score: null }}
                installing={installing} removing={removing} onInstall={handleInstall} onRemove={handleRemove} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
