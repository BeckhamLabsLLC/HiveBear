import { useCallback, useEffect, useRef, useState } from "react";
import { useInstalledModels, useModelInstall, useModelRemove, useModelSearch } from "../hooks/useRegistry";
import ModelCard from "../components/ModelCard";
import DownloadProgress from "../components/DownloadProgress";
import { Surface, EmptyState } from "../components/ui";
import { Search as SearchIcon, Package } from "lucide-react";

export default function ModelBrowser() {
  const [query, setQuery] = useState("");
  const { results, loading: searching, search } = useModelSearch();
  const { models: installed, refresh } = useInstalledModels();
  const { installing, progress, error: installError, install } = useModelInstall();
  const { removing, remove } = useModelRemove();
  const [tab, setTab] = useState<"search" | "installed">("search");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const handleSearch = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    search(query);
  }, [query, search]);

  // Debounced search-as-you-type
  const handleQueryChange = useCallback((val: string) => {
    setQuery(val);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (val.trim().length >= 2) {
      debounceRef.current = setTimeout(() => search(val), 300);
    }
  }, [search]);

  useEffect(() => {
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, []);

  const handleInstall = useCallback(async (modelId: string) => {
    const result = await install(modelId);
    if (result) refresh();
  }, [install, refresh]);

  const handleRemove = useCallback(async (modelId: string) => {
    const ok = await remove(modelId);
    if (ok) refresh();
  }, [remove, refresh]);

  return (
    <Surface>
      <div className="flex h-full flex-col -m-6 -mt-6">
        <div className="shrink-0 space-y-4 border-b border-border p-6 pb-4">
          <h1 className="text-xl font-semibold">Models</h1>

          {/* Tabs */}
          <div className="flex gap-1 rounded-[var(--radius-md)] bg-surface-overlay p-0.5">
            {(["search", "installed"] as const).map((t) => (
              <button key={t} onClick={() => setTab(t)}
                className={[
                  "flex-1 rounded-[var(--radius-sm)] px-3 py-1.5 text-sm capitalize interactive-hover",
                  tab === t ? "bg-surface-raised text-text-primary font-medium shadow-[var(--shadow-raised)]" : "text-text-muted hover:text-text-secondary",
                ].join(" ")}>
                {t} {t === "installed" && `(${installed.length})`}
              </button>
            ))}
          </div>

          {/* Search */}
          {tab === "search" && (
            <form onSubmit={handleSearch} className="flex gap-2">
              <div className="relative flex-1">
                <SearchIcon size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
                <input type="text" value={query} onChange={(e) => handleQueryChange(e.target.value)}
                  placeholder="Search models (e.g. llama, codestral, phi)..."
                  className="w-full rounded-[var(--radius-md)] border border-border bg-surface py-2 pl-9 pr-3 text-sm outline-none placeholder:text-text-muted focus:border-paw-500 focus:ring-2 focus:ring-paw-500/20" />
              </div>
              <button type="submit" className="interactive-hover press-scale rounded-[var(--radius-md)] bg-paw-500 px-4 py-2 text-sm font-medium text-white hover:bg-paw-600">
                Search
              </button>
            </form>
          )}
        </div>

        {/* Download progress */}
        {installing && (
          <div className="shrink-0 px-6 pt-4">
            <DownloadProgress modelId={installing} progress={progress} />
          </div>
        )}

        {installError && (
          <div className="mx-6 mt-4 rounded-[var(--radius-md)] border border-danger/30 bg-danger/10 px-4 py-2 text-sm text-danger">
            {installError}
          </div>
        )}

        {/* Results */}
        <div className="flex-1 overflow-y-auto p-6 pt-4">
          {tab === "search" ? (
            searching ? (
              <p className="text-sm text-text-muted">Searching...</p>
            ) : results.length === 0 ? (
              <EmptyState
                icon={<SearchIcon size={24} />}
                title={query ? "No results found" : "Search for models"}
                description={query ? "Try a different search term." : "Discover models from HuggingFace and install them locally."}
              >
                {!query && (
                  <div className="mt-2 flex flex-wrap justify-center gap-2">
                    {["llama", "mistral", "phi", "codestral", "qwen"].map((q) => (
                      <button
                        key={q}
                        onClick={() => { setQuery(q); search(q); }}
                        className="interactive-hover rounded-[var(--radius-md)] border border-border px-3 py-1.5 text-xs text-text-secondary hover:border-paw-500/30 hover:text-paw-400"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                )}
              </EmptyState>
            ) : (
              <div className="grid gap-3">
                {results.map((r) => (
                  <ModelCard key={r.metadata.id} result={r} installing={installing}
                    removing={removing} onInstall={handleInstall} onRemove={handleRemove} />
                ))}
              </div>
            )
          ) : installed.length === 0 ? (
            <EmptyState
              icon={<Package size={24} />}
              title="No models installed"
              description="Search and install a model to get started."
              action={{ label: "Browse Models", onClick: () => setTab("search") }}
            />
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
    </Surface>
  );
}
