import type { SearchResult } from "../types";
import { formatBytes } from "../types";
import { Download, Trash2, Check, Loader } from "lucide-react";
import { Badge } from "./ui";

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
    <div className={[
      "rounded-[var(--radius-lg)] border bg-surface-raised p-4 interactive-hover",
      is_installed
        ? "border-l-[3px] border-l-paw-500 border-t-border border-r-border border-b-border"
        : isThisInstalling
          ? "border-paw-500/40 shadow-[0_0_12px_-4px] shadow-paw-500/20"
          : "border-border",
    ].join(" ")}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-medium">{m.name}</h3>
          <p className="mt-0.5 flex items-center gap-1.5 text-xs text-text-muted">
            <span>{m.params_billions}B params</span>
            <span>·</span>
            <Badge variant="default">{m.category}</Badge>
            <span>·</span>
            <span>ctx {m.context_length.toLocaleString()}</span>
          </p>
          {m.description && (
            <p className="mt-1.5 line-clamp-2 text-xs text-text-secondary">{m.description}</p>
          )}
        </div>
        <div className="shrink-0">
          {is_installed ? (
            <button disabled={busy} onClick={() => onRemove(m.id)}
              className="interactive-hover press-scale flex items-center gap-1.5 rounded-[var(--radius-md)] border border-border px-3 py-1.5 text-xs text-text-secondary hover:border-danger hover:text-danger disabled:opacity-50">
              {isThisRemoving ? <Loader size={12} className="animate-spin" /> : <Trash2 size={12} />}
              Remove
            </button>
          ) : (
            <button disabled={busy} onClick={() => onInstall(m.id)}
              className="interactive-hover press-scale flex items-center gap-1.5 rounded-[var(--radius-md)] bg-paw-500 px-3 py-1.5 text-xs font-medium text-white hover:bg-paw-600 disabled:opacity-50">
              {isThisInstalling ? <Loader size={12} className="animate-spin" /> : <Download size={12} />}
              Install
            </button>
          )}
        </div>
      </div>
      <div className="mt-3 flex items-center gap-3 text-xs text-text-muted">
        {compatibility_score != null && (
          <Badge variant={compatibility_score >= 0.7 ? "success" : compatibility_score >= 0.4 ? "warning" : "danger"}>
            {(compatibility_score * 100).toFixed(0)}% compatible
          </Badge>
        )}
        {m.installed && (
          <Badge variant="success">
            <Check size={10} />{formatBytes(m.installed.size_bytes)}
          </Badge>
        )}
        {m.downloads_count != null && <span>{m.downloads_count.toLocaleString()} downloads</span>}
        {m.tags.length > 0 && <span className="truncate">{m.tags.slice(0, 3).join(" · ")}</span>}
      </div>
    </div>
  );
}
