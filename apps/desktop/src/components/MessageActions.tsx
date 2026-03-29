import { useState, useCallback } from "react";
import { Copy, Check, Pencil, RefreshCw } from "lucide-react";

interface MessageActionsProps {
  content: string;
  role: "user" | "assistant" | "system" | "tool";
  onEdit?: () => void;
  onRegenerate?: () => void;
}

export default function MessageActions({ content, role, onEdit, onRegenerate }: MessageActionsProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [content]);

  return (
    <div className="flex items-center gap-0.5 opacity-0 transition-opacity duration-[var(--duration-fast)] group-hover/msg:opacity-100">
      <button
        onClick={handleCopy}
        className="interactive-hover rounded-[var(--radius-sm)] p-1.5 text-text-muted hover:bg-surface-overlay hover:text-text-secondary"
        title="Copy"
      >
        {copied ? <Check size={13} className="text-success" /> : <Copy size={13} />}
      </button>

      {role === "user" && onEdit && (
        <button
          onClick={onEdit}
          className="interactive-hover rounded-[var(--radius-sm)] p-1.5 text-text-muted hover:bg-surface-overlay hover:text-text-secondary"
          title="Edit and resend"
        >
          <Pencil size={13} />
        </button>
      )}

      {role === "assistant" && onRegenerate && (
        <button
          onClick={onRegenerate}
          className="interactive-hover rounded-[var(--radius-sm)] p-1.5 text-text-muted hover:bg-surface-overlay hover:text-text-secondary"
          title="Regenerate"
        >
          <RefreshCw size={13} />
        </button>
      )}
    </div>
  );
}
