import { useProfile, useRecommendations } from "../hooks/useProfile";
import { useLoadedModels, useModelLoader } from "../hooks/useInference";
import { useInstalledModels } from "../hooks/useRegistry";
import { useModelInstall } from "../hooks/useRegistry";
import ResourceGauge from "../components/ResourceGauge";
import { formatBytes, formatToksPerSec } from "../types";
import { Cpu, HardDrive, MemoryStick, Monitor, Download, MessageSquare, Loader, Zap, Users } from "lucide-react";
import { useNavigate } from "react-router-dom";

export default function Dashboard() {
  const { profile, loading: profileLoading } = useProfile();
  const { recommendations, loading: recsLoading } = useRecommendations();
  const { models: loadedModels } = useLoadedModels();
  const { models: installedModels, refresh: refreshInstalled } = useInstalledModels();
  const { loading: modelLoading, load } = useModelLoader();
  const { installing, install } = useModelInstall();
  const navigate = useNavigate();

  const hasModels = installedModels.length > 0;
  const hasLoaded = loadedModels.length > 0;
  const topRec = recommendations[0] ?? null;

  const handleInstallTop = async () => {
    if (!topRec || installing) return;
    const result = await install(topRec.model_id, topRec.quantization);
    if (result) refreshInstalled();
  };

  const handleLoadAndChat = async (modelId: string) => {
    const result = await load(modelId);
    if (result) navigate("/chat");
  };

  if (profileLoading) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-text-muted">
        <Loader size={24} className="animate-spin text-paw-500" />
        <p className="text-sm">Sniffing your hardware...</p>
      </div>
    );
  }

  if (!profile) {
    return <div className="p-6 text-danger">Failed to detect hardware.</div>;
  }

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-xl font-semibold">Dashboard</h1>

      {/* First-run CTA: no models installed */}
      {!hasModels && !recsLoading && topRec && (
        <div className="rounded-2xl border-2 border-paw-500/30 bg-gradient-to-br from-paw-500/5 to-paw-600/10 p-6">
          <div className="flex items-start gap-4">
            <div className="rounded-xl bg-paw-500/10 p-3">
              <Zap size={24} className="text-paw-500" />
            </div>
            <div className="flex-1">
              <h2 className="text-lg font-semibold">Get Started</h2>
              <p className="mt-1 text-sm text-text-secondary">
                HiveBear analyzed your hardware and found the perfect model for this device.
                One click to install and start chatting.
              </p>
              <div className="mt-4 flex items-center gap-4">
                <button
                  onClick={handleInstallTop}
                  disabled={!!installing}
                  className="flex items-center gap-2 rounded-xl bg-paw-500 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-paw-600 disabled:opacity-50"
                >
                  {installing ? (
                    <Loader size={16} className="animate-spin" />
                  ) : (
                    <Download size={16} />
                  )}
                  Install {topRec.model_name} ({topRec.quantization})
                </button>
                <span className="text-xs text-text-muted">
                  {formatBytes(topRec.estimated_memory_usage_bytes)} ·{" "}
                  {formatToksPerSec(topRec.estimated_tokens_per_sec)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Models installed but none loaded — prompt to load and chat */}
      {hasModels && !hasLoaded && (
        <div className="rounded-2xl border border-paw-500/20 bg-paw-500/5 p-5">
          <div className="flex items-center gap-3">
            <MessageSquare size={20} className="text-paw-500" />
            <div className="flex-1">
              <p className="text-sm font-medium">Ready to chat</p>
              <p className="text-xs text-text-secondary">
                You have {installedModels.length} model{installedModels.length > 1 ? "s" : ""} installed. Load one to start chatting.
              </p>
            </div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {installedModels.slice(0, 3).map((m) => (
              <button
                key={m.id}
                onClick={() => handleLoadAndChat(m.id)}
                disabled={modelLoading}
                className="flex items-center gap-1.5 rounded-lg border border-paw-500/30 bg-surface px-3 py-2 text-xs font-medium text-paw-400 transition-colors hover:bg-paw-500/10 disabled:opacity-50"
              >
                {modelLoading ? <Loader size={12} className="animate-spin" /> : <Zap size={12} />}
                {m.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Hardware cards */}
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        <InfoCard
          icon={<Cpu size={18} />}
          label="CPU"
          value={profile.cpu.model_name}
          detail={`${profile.cpu.physical_cores} cores / ${profile.cpu.logical_cores} threads`}
        />
        <InfoCard
          icon={<MemoryStick size={18} />}
          label="Memory"
          value={formatBytes(profile.memory.total_bytes)}
          detail={`${profile.memory.estimated_bandwidth_gbps.toFixed(1)} GB/s bandwidth`}
        />
        {profile.gpus.length > 0 ? (
          <InfoCard
            icon={<Monitor size={18} />}
            label="GPU"
            value={profile.gpus[0].name}
            detail={`${formatBytes(profile.gpus[0].vram_bytes)} VRAM · ${profile.gpus[0].compute_api}`}
          />
        ) : (
          <InfoCard
            icon={<Monitor size={18} />}
            label="GPU"
            value="None detected"
            detail="CPU-only inference"
          />
        )}
        <InfoCard
          icon={<HardDrive size={18} />}
          label="Storage"
          value={`${formatBytes(profile.storage.available_bytes)} free`}
          detail={`${profile.storage.estimated_read_speed_mbps.toFixed(0)} MB/s read`}
        />
      </div>

      {/* Resource gauges */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <ResourceGauge
          label="Memory Usage"
          used={profile.memory.total_bytes - profile.memory.available_bytes}
          total={profile.memory.total_bytes}
          unit=""
        />
        {profile.gpus.length > 0 && (
          <ResourceGauge
            label="GPU VRAM"
            used={0}
            total={profile.gpus[0].vram_bytes}
            unit=""
          />
        )}
      </div>

      {/* Active models */}
      {hasLoaded && (
        <section>
          <h2 className="mb-3 text-base font-medium">Active Models</h2>
          <div className="space-y-2">
            {loadedModels.map((m) => (
              <div
                key={m.handle_id}
                className="flex items-center justify-between rounded-lg border border-paw-500/30 bg-paw-500/5 px-4 py-3"
              >
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-success animate-pulse" />
                  <span className="text-sm font-medium">
                    {m.model_path.split("/").pop()}
                  </span>
                  <span className="rounded bg-surface-overlay px-1.5 py-0.5 font-mono text-xs text-text-muted">
                    {m.engine}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs text-text-secondary">
                  <span>ctx: {m.context_length.toLocaleString()}</span>
                  {m.gpu_layers != null && <span>GPU layers: {m.gpu_layers}</span>}
                  <button
                    onClick={() => navigate("/chat")}
                    className="rounded-md bg-paw-500 px-2.5 py-1 text-white hover:bg-paw-600"
                  >
                    Chat
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Mesh callout */}
      {hasModels && (
        <div className="flex items-center gap-3 rounded-xl border border-border bg-surface-raised px-4 py-3">
          <Users size={16} className="shrink-0 text-paw-500" />
          <p className="flex-1 text-xs text-text-secondary">
            Share your idle compute with the hive.
            Help other bears run AI on devices that can't do it alone.
          </p>
          <button
            onClick={() => navigate("/mesh")}
            className="shrink-0 rounded-lg border border-paw-500/30 px-3 py-1.5 text-xs text-paw-400 hover:bg-paw-500/10"
          >
            Learn more
          </button>
        </div>
      )}

      {/* Top recommendations */}
      <section>
        <h2 className="mb-3 text-base font-medium">Recommended Models for This Device</h2>
        {recsLoading ? (
          <p className="text-sm text-text-muted">Calculating...</p>
        ) : recommendations.length === 0 ? (
          <p className="text-sm text-text-muted">No models found for this hardware profile.</p>
        ) : (
          <div className="space-y-2">
            {recommendations.slice(0, 5).map((rec) => (
              <div
                key={`${rec.model_id}-${rec.quantization}`}
                className="flex items-center justify-between rounded-lg border border-border bg-surface-raised px-4 py-3"
              >
                <div>
                  <span className="text-sm font-medium">{rec.model_name}</span>
                  <span className="ml-2 rounded bg-surface-overlay px-1.5 py-0.5 font-mono text-xs text-text-muted">
                    {rec.quantization}
                  </span>
                </div>
                <div className="flex items-center gap-4 text-xs">
                  <span className="text-text-secondary">
                    {formatBytes(rec.estimated_memory_usage_bytes)}
                  </span>
                  <span className={
                    rec.estimated_tokens_per_sec >= 20 ? "text-success"
                      : rec.estimated_tokens_per_sec >= 10 ? "text-warning"
                      : "text-danger"
                  }>
                    {formatToksPerSec(rec.estimated_tokens_per_sec)}
                  </span>
                  <span className="font-mono text-text-muted">
                    {(rec.confidence * 100).toFixed(0)}% conf
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function InfoCard({ icon, label, value, detail }: {
  icon: React.ReactNode; label: string; value: string; detail: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-surface-raised p-4">
      <div className="mb-2 flex items-center gap-2 text-text-muted">
        {icon}
        <span className="text-xs uppercase tracking-wider">{label}</span>
      </div>
      <p className="truncate text-sm font-medium">{value}</p>
      <p className="mt-0.5 truncate text-xs text-text-muted">{detail}</p>
    </div>
  );
}
