import type { SearchResult } from "../types";
import { formatBytes } from "../types";
import { Download, Trash2, Check, Loader } from "lucide-react";

interface ModelCardProps {
  result: SearchResult;
  installing: string | null;
  removing: string | null;
  onInstall: (modelId: string) => void;
  onRemove: (modelId: string) => void;
}

export default function ModelCard({ result, installing, removing, onInstall, onRemove }: ModelCardProps) {
  const { metadata: m, is_installed, compatibility_score } = result;
  const isThisInstalling = installing === m.id;
  const isThisRemoving = removing === m.id;
  const busy = isThisInstalling || isThisRemoving;

  return (
    <div className="rounded-xl border border-border bg-surface-raised p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-medium">{m.name}</h3>
          <p className="mt-0.5 text-xs text-text-muted">
            {m.params_billions}B params · {m.category} · ctx {m.context_length}
          </p>
          {m.description && (
            <p className="mt-1.5 line-clamp-2 text-xs text-text-secondary">{m.description}</p>
          )}
        </div>
        <div className="shrink-0">
          {is_installed ? (
            <button disabled={busy} onClick={() => onRemove(m.id)}
              className="flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-xs text-text-secondary transition-colors hover:border-danger hover:text-danger disabled:opacity-50">
              {isThisRemoving ? <Loader size={12} className="animate-spin" /> : <Trash2 size={12} />}
              Remove
            </button>
          ) : (
            <button disabled={busy} onClick={() => onInstall(m.id)}
              className="flex items-center gap-1.5 rounded-lg bg-paw-500 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-paw-600 disabled:opacity-50">
              {isThisInstalling ? <Loader size={12} className="animate-spin" /> : <Download size={12} />}
              Install
            </button>
          )}
        </div>
      </div>
      <div className="mt-3 flex items-center gap-3 text-xs text-text-muted">
        {compatibility_score != null && (
          <span className={compatibility_score >= 0.7 ? "text-success" : compatibility_score >= 0.4 ? "text-warning" : "text-danger"}>
            {(compatibility_score * 100).toFixed(0)}% compatible
          </span>
        )}
        {m.installed && (
          <span className="flex items-center gap-1 text-success">
            <Check size={10} />{formatBytes(m.installed.size_bytes)}
          </span>
        )}
        {m.downloads_count != null && <span>{m.downloads_count.toLocaleString()} downloads</span>}
        {m.tags.length > 0 && <span className="truncate">{m.tags.slice(0, 3).join(" · ")}</span>}
      </div>
    </div>
  );
}
