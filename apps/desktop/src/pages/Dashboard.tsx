import { useProfile, useRecommendations } from "../hooks/useProfile";
import { useLoadedModels, useModelLoader } from "../hooks/useInference";
import { useInstalledModels, useModelInstall } from "../hooks/useRegistry";
import ResourceGauge from "../components/ResourceGauge";
import { Card, Button, Badge, Surface, EmptyState } from "../components/ui";
import { formatBytes, formatToksPerSec } from "../types";
import {
  Cpu, HardDrive, MemoryStick, Monitor, Download, MessageSquare,
  Loader, Zap, Users, ArrowRight,
} from "lucide-react";
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
      <Surface>
        <EmptyState
          icon={<Loader size={24} className="animate-spin text-paw-500" />}
          title="Sniffing your hardware..."
          description="Profiling CPU, memory, GPU, and storage."
        />
      </Surface>
    );
  }

  if (!profile) {
    return (
      <Surface>
        <div className="text-danger text-sm">Failed to detect hardware.</div>
      </Surface>
    );
  }

  return (
    <Surface>
      <div className="space-y-6">
        {/* ── Zone 1: Hero / Status ─────────────────────────────────────── */}

        {!hasModels && !recsLoading && topRec ? (
          /* First-run: no models installed */
          <Card padding="lg" className="border-paw-500/20 bg-gradient-to-br from-paw-500/5 to-paw-600/10">
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-[var(--radius-lg)] bg-paw-500/10">
                <Zap size={24} className="text-paw-500" />
              </div>
              <div className="flex-1">
                <h2 className="text-lg font-semibold">Get Started</h2>
                <p className="mt-1 text-sm text-text-secondary">
                  HiveBear analyzed your hardware and found the perfect model.
                  One click to install and start chatting.
                </p>
                <div className="mt-4 flex items-center gap-4">
                  <Button onClick={handleInstallTop} disabled={!!installing}>
                    {installing ? <Loader size={16} className="animate-spin" /> : <Download size={16} />}
                    Install {topRec.model_name} ({topRec.quantization})
                  </Button>
                  <span className="text-xs text-text-muted">
                    {formatBytes(topRec.estimated_memory_usage_bytes)} ·{" "}
                    {formatToksPerSec(topRec.estimated_tokens_per_sec)}
                  </span>
                </div>
              </div>
            </div>
          </Card>
        ) : hasLoaded ? (
          /* Model is loaded and ready */
          <Card padding="lg" className="border-paw-500/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-[var(--radius-lg)] bg-success/10">
                  <span className="h-2.5 w-2.5 rounded-full bg-success animate-pulse" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold">
                      {loadedModels[0].model_path.split("/").pop()}
                    </span>
                    <Badge variant="default">{loadedModels[0].engine}</Badge>
                  </div>
                  <p className="text-xs text-text-muted">
                    Ready · ctx {loadedModels[0].context_length.toLocaleString()}
                    {loadedModels[0].gpu_layers != null && ` · ${loadedModels[0].gpu_layers} GPU layers`}
                  </p>
                </div>
              </div>
              <Button onClick={() => navigate("/chat")}>
                <MessageSquare size={14} />
                Open Chat
                <ArrowRight size={14} />
              </Button>
            </div>

            {/* Additional loaded models */}
            {loadedModels.length > 1 && (
              <div className="mt-3 flex flex-wrap gap-2 border-t border-border pt-3">
                {loadedModels.slice(1).map((m) => (
                  <Badge key={m.handle_id} variant="accent">
                    <span className="h-1.5 w-1.5 rounded-full bg-success" />
                    {m.model_path.split("/").pop()}
                  </Badge>
                ))}
              </div>
            )}
          </Card>
        ) : hasModels ? (
          /* Models installed but none loaded */
          <Card padding="lg" className="border-paw-500/15 bg-paw-500/[0.03]">
            <div className="flex items-center gap-3">
              <MessageSquare size={20} className="text-paw-500" />
              <div className="flex-1">
                <p className="text-sm font-medium">Ready to chat</p>
                <p className="text-xs text-text-secondary">
                  {installedModels.length} model{installedModels.length > 1 ? "s" : ""} installed. Load one to start.
                </p>
              </div>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {installedModels.slice(0, 3).map((m) => (
                <Button
                  key={m.id}
                  variant="secondary"
                  size="sm"
                  onClick={() => handleLoadAndChat(m.id)}
                  disabled={modelLoading}
                >
                  {modelLoading ? <Loader size={12} className="animate-spin" /> : <Zap size={12} />}
                  {m.name}
                </Button>
              ))}
            </div>
          </Card>
        ) : null}

        {/* ── Zone 2: System (bento grid) ──────────────────────────────── */}

        <div className="grid grid-cols-5 gap-4">
          {/* Left: Resource gauges (3 cols) */}
          <div className="col-span-3 space-y-4">
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

          {/* Right: Hardware summary (2 cols) */}
          <Card className="col-span-2">
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-text-muted">Hardware</h3>
            <div className="space-y-3">
              <HardwareRow icon={<Cpu size={14} />} label="CPU" value={profile.cpu.model_name} detail={`${profile.cpu.physical_cores}c / ${profile.cpu.logical_cores}t`} />
              <HardwareRow icon={<MemoryStick size={14} />} label="RAM" value={formatBytes(profile.memory.total_bytes)} detail={`${profile.memory.estimated_bandwidth_gbps.toFixed(1)} GB/s`} />
              {profile.gpus.length > 0 ? (
                <HardwareRow icon={<Monitor size={14} />} label="GPU" value={profile.gpus[0].name} detail={`${formatBytes(profile.gpus[0].vram_bytes)} · ${profile.gpus[0].compute_api}`} />
              ) : (
                <HardwareRow icon={<Monitor size={14} />} label="GPU" value="None detected" detail="CPU-only inference" />
              )}
              <HardwareRow icon={<HardDrive size={14} />} label="Storage" value={`${formatBytes(profile.storage.available_bytes)} free`} detail={`${profile.storage.estimated_read_speed_mbps.toFixed(0)} MB/s read`} />
            </div>
          </Card>
        </div>

        {/* ── Zone 3: Discovery ────────────────────────────────────────── */}

        <section>
          <h2 className="mb-3 text-sm font-semibold text-text-secondary">Recommended for This Device</h2>
          {recsLoading ? (
            <p className="text-xs text-text-muted">Calculating...</p>
          ) : recommendations.length === 0 ? (
            <p className="text-xs text-text-muted">No models found for this hardware profile.</p>
          ) : (
            <div className="space-y-2">
              {recommendations.slice(0, 3).map((rec) => (
                <Card
                  key={`${rec.model_id}-${rec.quantization}`}
                  interactive
                  padding="sm"
                  className="group"
                  onClick={() => navigate("/models")}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-medium">{rec.model_name}</span>
                      <Badge variant="default">{rec.quantization}</Badge>
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
                        {(rec.confidence * 100).toFixed(0)}%
                      </span>
                      <ArrowRight size={12} className="text-text-muted opacity-0 group-hover:opacity-100 interactive-hover" />
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </section>

        {/* Mesh callout */}
        {hasModels && (
          <Card padding="sm" interactive onClick={() => navigate("/mesh")} className="group">
            <div className="flex items-center gap-3">
              <Users size={14} className="shrink-0 text-paw-500" />
              <p className="flex-1 text-xs text-text-secondary">
                Share your idle compute with the hive. Help other bears run AI on devices that can't do it alone.
              </p>
              <ArrowRight size={12} className="text-text-muted opacity-0 group-hover:opacity-100 interactive-hover" />
            </div>
          </Card>
        )}
      </div>
    </Surface>
  );
}

function HardwareRow({ icon, label, value, detail }: {
  icon: React.ReactNode; label: string; value: string; detail: string;
}) {
  return (
    <div className="flex items-start gap-2.5">
      <div className="mt-0.5 text-text-muted">{icon}</div>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-[11px] uppercase tracking-wider text-text-muted">{label}</span>
        </div>
        <p className="truncate text-xs font-medium text-text-primary">{value}</p>
        <p className="truncate text-[11px] text-text-muted">{detail}</p>
      </div>
    </div>
  );
}
