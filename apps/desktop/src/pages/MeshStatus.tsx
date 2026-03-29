import { useCallback, useEffect, useState } from "react";
import { getMeshStatus, saveMeshConfig, getMeshConfig } from "../lib/invoke";
import type { MeshStatus as MeshStatusType } from "../types";
import { Link } from "react-router-dom";
import { Network, Settings, Power, Shield, Activity } from "lucide-react";

export default function MeshStatus() {
  const [status, setStatus] = useState<MeshStatusType | null>(null);
  const [loading, setLoading] = useState(true);
  const [toggling, setToggling] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setStatus(await getMeshStatus());
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const toggleEnabled = useCallback(async () => {
    if (!status) return;
    setToggling(true);
    try {
      const config = await getMeshConfig();
      config.enabled = !config.enabled;
      await saveMeshConfig(config);
      await refresh();
    } catch {
      /* ignore */
    } finally {
      setToggling(false);
    }
  }, [status, refresh]);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-text-muted">
        Loading mesh status...
      </div>
    );
  }

  if (!status) {
    return <div className="p-6 text-danger">Failed to load mesh status.</div>;
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">P2P Mesh</h1>
        <Link
          to="/settings"
          className="flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-xs text-text-secondary hover:text-text-primary"
        >
          <Settings size={12} />
          Configure
        </Link>
      </div>

      {/* Status header */}
      <div className="flex items-center gap-4 rounded-xl border border-border bg-surface-raised p-5">
        <div
          className={`flex h-12 w-12 items-center justify-center rounded-xl ${
            status.enabled
              ? "bg-success/10 text-success"
              : "bg-surface-overlay text-text-muted"
          }`}
        >
          <Network size={24} />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="text-base font-medium">
              Mesh {status.enabled ? "Enabled" : "Disabled"}
            </span>
            <span
              className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                status.tier === "paid"
                  ? "bg-paw-500/10 text-paw-400"
                  : "bg-surface-overlay text-text-muted"
              }`}
            >
              {status.tier === "paid" ? "Paid" : "Free"}
            </span>
          </div>
          <p className="mt-0.5 text-sm text-text-muted">
            {status.enabled
              ? `Listening on port ${status.port}`
              : "Enable to join the mesh network and share compute."}
          </p>
        </div>
        <button
          onClick={toggleEnabled}
          disabled={toggling}
          className={`flex items-center gap-1.5 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
            status.enabled
              ? "border border-border text-text-secondary hover:border-danger hover:text-danger"
              : "bg-paw-500 text-white hover:bg-paw-600"
          } disabled:opacity-50`}
        >
          <Power size={14} />
          {status.enabled ? "Disable" : "Enable"}
        </button>
      </div>

      {/* Config cards */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <StatCard
          icon={<Network size={16} />}
          label="Coordination Server"
          value={status.coordination_server.replace(/^https?:\/\//, "")}
        />
        <StatCard
          icon={<Shield size={16} />}
          label="Min Trust Score"
          value={`${(status.min_reputation * 100).toFixed(0)}%`}
        />
        <StatCard
          icon={<Activity size={16} />}
          label="Verification Rate"
          value={`${(status.verification_rate * 100).toFixed(0)}%`}
        />
      </div>

      {/* Contribution config */}
      <section className="rounded-xl border border-border bg-surface-raised p-5">
        <h2 className="mb-3 text-sm font-medium">Resource Contribution</h2>
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <div className="mb-1 flex justify-between text-xs">
              <span className="text-text-muted">Share limit</span>
              <span className="font-mono text-text-secondary">
                {(status.max_contribution_percent * 100).toFixed(0)}%
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-surface-overlay">
              <div
                className="h-full rounded-full bg-paw-500"
                style={{
                  width: `${status.max_contribution_percent * 100}%`,
                }}
              />
            </div>
          </div>
          <p className="max-w-48 text-xs text-text-muted">
            Maximum percentage of local compute resources shared with the mesh
            network.
          </p>
        </div>
      </section>

      {/* Placeholder for peer list (future) */}
      {status.enabled && (
        <section className="rounded-xl border border-dashed border-border p-6 text-center">
          <Network size={32} className="mx-auto mb-2 text-text-muted" />
          <p className="text-sm text-text-muted">
            Peer discovery will appear here when the coordination server is
            running.
          </p>
          <p className="mt-1 font-mono text-xs text-text-muted">
            {status.coordination_server}
          </p>
        </section>
      )}
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-surface-raised p-4">
      <div className="mb-2 flex items-center gap-2 text-text-muted">
        {icon}
        <span className="text-xs uppercase tracking-wider">{label}</span>
      </div>
      <p className="truncate text-sm font-medium">{value}</p>
    </div>
  );
}
