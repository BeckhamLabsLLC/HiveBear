import { useCallback, useEffect, useState } from "react";
import { getMeshStatus, saveMeshConfig, getMeshConfig, getMeshConnectionStatus, joinMesh, leaveMesh } from "../lib/invoke";
import type { MeshConnectionStatus } from "../lib/invoke";
import type { MeshStatus as MeshStatusType } from "../types";
import { Link } from "react-router-dom";
import { Network, Settings, Power, Shield, Activity, Wifi, WifiOff } from "lucide-react";
import { Card, Button, Badge, Surface } from "../components/ui";
import { notify } from "../components/Toast";
import { EARLY_MESSAGE } from "../components/MeshStatusPill";

/** For errors from silent commands; the others are already toasted by invoke(). */
function reportMeshError(action: string, e: unknown) {
  notify(`${action}: ${e instanceof Error ? e.message : String(e)}`, "error");
}

export default function MeshStatus() {
  const [status, setStatus] = useState<MeshStatusType | null>(null);
  const [connection, setConnection] = useState<MeshConnectionStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [toggling, setToggling] = useState(false);
  const [joining, setJoining] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [s, c] = await Promise.all([getMeshStatus(), getMeshConnectionStatus()]);
      setStatus(s);
      setConnection(c);
    } catch (e) {
      // Both calls are silent (they also run on a timer), so say so here.
      reportMeshError("Couldn't read mesh status", e);
    }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Poll connection status every 30s
  useEffect(() => {
    const interval = setInterval(async () => {
      try { setConnection(await getMeshConnectionStatus()); } catch { /* ignore */ }
    }, 30_000);
    return () => clearInterval(interval);
  }, []);

  const toggleEnabled = useCallback(async () => {
    if (!status) return;
    setToggling(true);
    try {
      const config = await getMeshConfig();
      config.enabled = !config.enabled;
      await saveMeshConfig(config);
      // Disabling should also take the node off the hive now, not at restart.
      if (!config.enabled && connection?.running) await leaveMesh();
      await refresh();
    } catch { /* getMeshConfig/saveMeshConfig/leaveMesh already toasted */ }
    finally { setToggling(false); }
  }, [status, connection, refresh]);

  if (loading) {
    return <Surface><div className="flex h-full items-center justify-center text-text-muted text-sm">Loading mesh status...</div></Surface>;
  }

  if (!status) {
    return <Surface><div className="text-sm text-danger">Failed to load mesh status.</div></Surface>;
  }

  return (
    <Surface>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">P2P Mesh</h1>
          <Link to="/settings"
            className="interactive-hover flex items-center gap-1.5 rounded-[var(--radius-md)] border border-border px-3 py-1.5 text-xs text-text-secondary hover:text-text-primary">
            <Settings size={12} /> Configure
          </Link>
        </div>

        {/* Status header */}
        <Card padding="lg">
          <div className="flex items-center gap-4">
            <div className={[
              "flex h-12 w-12 items-center justify-center rounded-[var(--radius-lg)]",
              status.enabled ? "bg-success/10 text-success" : "bg-surface-overlay text-text-muted",
            ].join(" ")}>
              <Network size={24} />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-base font-medium">
                  Mesh {status.enabled ? "Enabled" : "Disabled"}
                </span>
                <Badge variant={status.tier === "paid" ? "accent" : "default"}>
                  {status.tier === "paid" ? "Paid" : "Free"}
                </Badge>
              </div>
              <p className="mt-0.5 text-sm text-text-muted">
                {status.enabled
                  ? `Listening on port ${status.port}`
                  : "Enable to join the mesh network and share compute."}
              </p>
            </div>
            <Button
              variant={status.enabled ? "secondary" : "primary"}
              onClick={toggleEnabled}
              disabled={toggling}
            >
              <Power size={14} />
              {status.enabled ? "Disable" : "Enable"}
            </Button>
          </div>
        </Card>

        {/* Config cards */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <StatCard icon={<Network size={14} />} label="Coordination Server" value={status.coordination_server.replace(/^https?:\/\//, "")} />
          <StatCard icon={<Shield size={14} />} label="Min Trust Score" value={`${(status.min_reputation * 100).toFixed(0)}%`} />
          <StatCard icon={<Activity size={14} />} label="Verification Rate" value={`${(status.verification_rate * 100).toFixed(0)}%`} />
        </div>

        {/* Contribution */}
        <Card>
          <h2 className="mb-3 text-sm font-semibold">Resource Contribution</h2>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className="mb-1 flex justify-between text-xs">
                <span className="text-text-muted">Share limit</span>
                <span className="font-mono text-text-secondary">
                  {(status.max_contribution_percent * 100).toFixed(0)}%
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-surface-overlay">
                <div className="h-full rounded-full bg-paw-500 interactive-hover" style={{ width: `${status.max_contribution_percent * 100}%` }} />
              </div>
            </div>
            <p className="max-w-48 text-xs text-text-muted">
              Maximum percentage of local resources shared with the mesh network.
            </p>
          </div>
        </Card>

        {/* Connection status */}
        {connection && (
          <Card padding="lg">
            <div className="flex items-center gap-4">
              <div className={[
                "flex h-12 w-12 items-center justify-center rounded-[var(--radius-lg)]",
                connection.registered
                  ? "bg-success/10 text-success"
                  : connection.running
                    ? "bg-warning/10 text-warning"
                    : "bg-surface-overlay text-text-muted",
              ].join(" ")}>
                {connection.registered ? <Wifi size={24} /> : <WifiOff size={24} />}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  {/* "Connected" means the coordination server has acknowledged
                      us, not merely that a local socket is open. Keying this off
                      `running` claimed we were on the hive while offline. */}
                  <span className="text-base font-medium">
                    {connection.registered
                      ? "Connected to Hive"
                      : connection.running
                        ? "Running locally — not on the hive"
                        : "Not Connected"}
                  </span>
                  {connection.registered && (
                    <Badge variant="default">{connection.peer_count} peer{connection.peer_count !== 1 ? "s" : ""}</Badge>
                  )}
                </div>
                <p className="mt-0.5 text-sm text-text-muted">
                  {connection.registered && connection.peer_count === 0
                    ? EARLY_MESSAGE
                    : connection.registered
                    ? `Node ${connection.node_id?.slice(0, 12)}… registered and sending heartbeats`
                    : connection.running
                      ? "The coordination server is unreachable, so other peers cannot find you. Retrying in the background."
                      : connection.last_error
                        ? `Last attempt failed: ${connection.last_error}`
                        : "Join to share idle compute with other bears. Nothing is shared until you do."}
                </p>
              </div>
              <Button
                variant={connection.running ? "secondary" : "primary"}
                disabled={joining}
                onClick={async () => {
                  setJoining(true);
                  try {
                    if (connection.running) {
                      await leaveMesh();
                    } else {
                      await joinMesh();
                      // The node starts in the background, so a failed bind
                      // or registration only shows up a moment later.
                      for (let i = 0; i < 10; i++) {
                        await new Promise((r) => setTimeout(r, 500));
                        const c = await getMeshConnectionStatus();
                        if (c.last_error) { reportMeshError("Couldn't join the hive", c.last_error); break; }
                        if (c.registered) break;
                      }
                    }
                    await refresh();
                  } catch { /* joinMesh/leaveMesh already toasted the reason */ }
                  finally { setJoining(false); }
                }}
              >
                {connection.running ? <WifiOff size={14} /> : <Wifi size={14} />}
                {joining ? "…" : connection.running ? "Leave" : "Join"}
              </Button>
            </div>
          </Card>
        )}
      </div>
    </Surface>
  );
}

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <Card>
      <div className="mb-2 flex items-center gap-2 text-text-muted">
        {icon}
        <span className="text-[11px] uppercase tracking-wider">{label}</span>
      </div>
      <p className="truncate text-sm font-medium">{value}</p>
    </Card>
  );
}
