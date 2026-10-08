import { useNavigate } from "react-router-dom";
import { Gauge, Network, X } from "lucide-react";

const STORAGE_KEY = "hivebear.firstChatNudge.dismissed.v1";

export function firstChatNudgeDismissed(): boolean {
  try { return localStorage.getItem(STORAGE_KEY) === "true"; } catch { return false; }
}

function dismissForGood() {
  try { localStorage.setItem(STORAGE_KEY, "true"); } catch { /* ignore */ }
}

/**
 * Shown once, after the first successful reply: the moment someone has seen
 * HiveBear work is the moment a 30-second benchmark (and the leaderboard)
 * is an easy yes. Gone for good once clicked or dismissed.
 */
export default function FirstChatNudge({ onClose }: { onClose: () => void }) {
  const navigate = useNavigate();

  const go = (path: string) => {
    dismissForGood();
    onClose();
    navigate(path);
  };

  return (
    <div className="mx-auto mb-3 flex max-w-2xl items-center gap-3 rounded-[var(--radius-lg)] border border-paw-500/30 bg-paw-500/5 px-4 py-2.5 animate-[fade-in]">
      <Gauge size={16} className="shrink-0 text-paw-500" aria-hidden />
      <button
        onClick={() => go("/benchmark")}
        className="interactive-hover flex-1 text-left text-xs font-medium text-text-primary hover:text-paw-400"
      >
        See how fast your machine is — run a 30s benchmark →
      </button>
      <button
        onClick={() => go("/mesh")}
        className="interactive-hover hidden items-center gap-1 text-[11px] text-text-muted hover:text-text-secondary sm:inline-flex"
      >
        <Network size={11} aria-hidden /> Join the hive
      </button>
      <button
        onClick={() => { dismissForGood(); onClose(); }}
        className="interactive-hover rounded-[var(--radius-md)] p-1 text-text-muted hover:text-text-primary"
        aria-label="Dismiss"
      >
        <X size={12} />
      </button>
    </div>
  );
}
