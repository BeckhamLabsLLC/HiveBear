import { useNavigate } from "react-router-dom";
import { Network, Users, WifiOff } from "lucide-react";
import { useMeshPulse } from "../hooks/useMeshPulse";

/**
 * Compact mesh-status widget for the Dashboard header.
 *
 * Shows one of:
 *   - "Mesh idle" with a muted dot (not running yet — CTA to enable)
 *   - "Mesh connecting" while joining (peer_count 0 and running)
 *   - "Mesh active · N peers" when running with peers
 *   - "Mesh unreachable" when the poll fails repeatedly
 *
 * This is the retention lever: users need to see that the mesh is
 * running and that they are connected to others.
 */
export default function MeshStatusPill() {
  const { status, loading, error } = useMeshPulse();
  const navigate = useNavigate();

  const { label, tone, icon } = describe(status, loading, error);

  return (
    <button
      onClick={() => navigate("/mesh")}
      className={[
        "interactive-hover group inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs",
        tone === "active"
          ? "border-success/30 bg-success/10 text-success hover:bg-success/15"
          : tone === "connecting"
          ? "border-paw-500/30 bg-paw-500/10 text-paw-400 hover:bg-paw-500/15"
          : tone === "error"
          ? "border-warning/30 bg-warning/10 text-warning hover:bg-warning/15"
          : "border-border bg-surface-overlay text-text-muted hover:text-text-secondary",
      ].join(" ")}
      aria-label={label}
    >
      {icon}
      <span className="font-medium">{label}</span>
    </button>
  );
}

function describe(
  status: ReturnType<typeof useMeshPulse>["status"],
  loading: boolean,
  error: boolean,
) {
  if (loading && !status) {
    return {
      label: "Mesh …",
      tone: "idle" as const,
      icon: <span className="h-1.5 w-1.5 rounded-full bg-text-muted" aria-hidden />,
    };
  }
  if (error) {
    return {
      label: "Mesh unreachable",
      tone: "error" as const,
      icon: <WifiOff size={12} aria-hidden />,
    };
  }
  if (!status || !status.running) {
    return {
      label: "Mesh idle",
      tone: "idle" as const,
      icon: <Network size={12} aria-hidden />,
    };
  }
  if (status.peer_count === 0) {
    return {
      label: "Mesh connecting",
      tone: "connecting" as const,
      icon: <Network size={12} className="animate-pulse" aria-hidden />,
    };
  }
  return {
    label: `Mesh active · ${status.peer_count} peer${status.peer_count === 1 ? "" : "s"}`,
    tone: "active" as const,
    icon: <Users size={12} aria-hidden />,
  };
}
