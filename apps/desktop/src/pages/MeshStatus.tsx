import { useCallback, useEffect, useState } from "react";
import { getMeshStatus, saveMeshConfig, getMeshConfig } from "../lib/invoke";
import type { MeshStatus as MeshStatusType } from "../types";
import { Link } from "react-router-dom";
import { Network, Settings, Power, Shield, Activity } from "lucide-react";
import { Card, Button, Badge, Surface, EmptyState } from "../components/ui";

export default function MeshStatus() {
  const [status, setStatus] = useState<MeshStatusType | null>(null);
  const [loading, setLoading] = useState(true);
  const [toggling, setToggling] = useState(false);

  const refresh = useCallback(async () => {
    try { setStatus(await getMeshStatus()); } catch { /* ignore */ }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const toggleEnabled = useCallback(async () => {
    if (!status) return;
    setToggling(true);
    try {
      const config = await getMeshConfig();
      config.enabled = !config.enabled;
      await saveMeshConfig(config);
      await refresh();
    } catch { /* ignore */ }
    finally { setToggling(false); }
  }, [status, refresh]);

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

        {/* Peer discovery placeholder */}
        {status.enabled && (
          <EmptyState
            icon={<Network size={24} />}
            title="Waiting for peers"
            description={`Peer discovery will appear here when the coordination server is running at ${status.coordination_server}`}
          />
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
