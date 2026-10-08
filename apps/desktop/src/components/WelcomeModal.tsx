import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ArrowRight, Check, Cpu, Download, Loader, Network, ShieldCheck, Sparkles } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import Toggle from "./ui/Toggle";
import { useAppReady, whenAppReady } from "../hooks/useAppReady";
import { getConfig, getHardwareProfile, getRecommendations, saveConfig } from "../lib/invoke";
import { firstModel } from "../lib/firstModel";
import { requestInstall, type InstallRequest } from "../lib/pendingInstall";
import type { HardwareProfile, ModelRecommendation } from "../types";
import { formatBytes, formatDownloadSize, formatQuant, formatToksPerSec } from "../types";

const STORAGE_KEY = "hivebear.onboarded.v1";
const EVENT_KEY = "hivebear:onboarding-changed";

export function hasOnboarded(): boolean {
  try { return localStorage.getItem(STORAGE_KEY) === "true"; } catch { return true; }
}

function markOnboarded() {
  try { localStorage.setItem(STORAGE_KEY, "true"); } catch { /* ignore */ }
  try { window.dispatchEvent(new Event(EVENT_KEY)); } catch { /* ignore */ }
}

export function useHasOnboarded(): boolean {
  const [done, setDone] = useState(() => hasOnboarded());
  useEffect(() => {
    const handler = () => setDone(hasOnboarded());
    window.addEventListener(EVENT_KEY, handler);
    window.addEventListener("storage", handler);
    return () => {
      window.removeEventListener(EVENT_KEY, handler);
      window.removeEventListener("storage", handler);
    };
  }, []);
  return done;
}

interface Slide {
  icon: React.ReactNode;
  title: string;
  body: string;
  bullets?: string[];
}

const SLIDES: Slide[] = [
  {
    icon: <Sparkles size={20} className="text-paw-500" aria-hidden />,
    title: "Welcome to HiveBear",
    body: "Run open-source AI models on your own machine — no cloud, no subscription, no sign-up required. HiveBear finds the models that fit your hardware and lets you chat with them locally.",
  },
  {
    icon: <Network size={20} className="text-paw-500" aria-hidden />,
    title: "Bigger models, together",
    body: "Your laptop can't run Llama 3 70B alone — but a few peers together can. When you join the mesh, HiveBear pairs your idle compute with other bears so everyone can run models no single device could handle.",
    bullets: [
      "The mesh is off until you choose to join it.",
      "Your device identity is a local keypair. It never leaves your machine.",
    ],
  },
  {
    icon: <ShieldCheck size={20} className="text-paw-500" aria-hidden />,
    title: "If something breaks",
    body: "HiveBear sends anonymous crash reports so we can fix problems we'd otherwise never hear about. You can turn this off at any time in Settings.",
    bullets: [
      "Never your prompts, chat history, model files or account details.",
      "No name, email or IP address — just the error and a random ID.",
    ],
  },
  {
    icon: <Cpu size={20} className="text-paw-500" aria-hidden />,
    title: "One quick check",
    body: "HiveBear looked at your CPU, memory and GPU to pick a model that runs well here. Your hardware details stay on this device.",
  },
];

const PRIVACY_SLIDE = 2;

/**
 * Persist the welcome-screen choices, then send `first_launch` — in that
 * order, so switching usage counts off here is honoured before anything goes.
 * Waits for the backend, since the user can click through faster than it
 * finishes profiling.
 */
async function saveOnboardingChoices(usageEvents: boolean) {
  await whenAppReady();
  try {
    const config = await getConfig();
    if (config.telemetry?.usage_events !== usageEvents) {
      await saveConfig({ ...config, telemetry: { ...config.telemetry, usage_events: usageEvents } });
    }
  } catch { /* already toasted by invoke(); the default stays in place */ }
  void invoke("record_usage_event", { event: "first_launch" }).catch(() => {});
}

export default function WelcomeModal() {
  const done = useHasOnboarded();
  const ready = useAppReady();
  const [index, setIndex] = useState(0);
  const [usageEvents, setUsageEvents] = useState(true);
  const [profile, setProfile] = useState<HardwareProfile | null>(null);
  const [recs, setRecs] = useState<ModelRecommendation[] | null>(null);

  // Fetch as soon as the backend is up, so the last slide is filled in by the
  // time anyone reaches it.
  useEffect(() => {
    if (done || !ready) return;
    getHardwareProfile().then(setProfile).catch(() => {});
    getRecommendations().then(setRecs).catch(() => setRecs([]));
  }, [done, ready]);

  const finish = useCallback((install?: InstallRequest) => {
    markOnboarded();
    // Record that the crash-reporting notice has actually been shown. Opt-out
    // reporting is only defensible if people are told, and this is what lets the
    // Rust side stop owing the notice. Failing to record it is harmless — the
    // notice would simply be shown again.
    void invoke("acknowledge_telemetry_notice").catch(() => {});
    void saveOnboardingChoices(usageEvents).then(() => {
      // The Dashboard shows the download progress; see lib/pendingInstall.
      if (install) requestInstall(install);
    });
  }, [usageEvents]);
  const next = useCallback(() => {
    if (index >= SLIDES.length - 1) finish();
    else setIndex((i) => i + 1);
  }, [index, finish]);

  if (done) return null;
  const slide = SLIDES[index];
  const isLast = index === SLIDES.length - 1;
  const pick = recs ? firstModel(recs) : null;
  const size = pick ? formatDownloadSize(pick.model.estimated_download_bytes) : null;

  return (
    <AnimatePresence>
      <motion.div
        key="welcome-scrim"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-[200] flex items-center justify-center bg-black/60 p-6 backdrop-blur-sm"
        role="dialog"
        aria-modal="true"
        aria-labelledby="welcome-title"
      >
        <motion.div
          key={`slide-${index}`}
          initial={{ opacity: 0, y: 12, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -8, scale: 0.98 }}
          transition={{ type: "spring", damping: 26, stiffness: 320 }}
          className="w-full max-w-md overflow-hidden rounded-[var(--radius-xl)] border border-border bg-surface-raised shadow-[var(--shadow-overlay)]"
        >
          <div className="p-6">
            <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-[var(--radius-lg)] bg-paw-500/10">
              {slide.icon}
            </div>
            <h2 id="welcome-title" className="text-lg font-semibold text-text-primary">
              {slide.title}
            </h2>
            <p className="mt-2 text-sm leading-relaxed text-text-secondary">{slide.body}</p>
            {slide.bullets && (
              <ul className="mt-3 space-y-1.5">
                {slide.bullets.map((b) => (
                  <li key={b} className="flex items-start gap-2 text-xs text-text-muted">
                    <Check size={12} className="mt-0.5 shrink-0 text-success" aria-hidden />
                    <span>{b}</span>
                  </li>
                ))}
              </ul>
            )}

            {index === PRIVACY_SLIDE && (
              <div className="mt-4 flex items-start justify-between gap-3 rounded-[var(--radius-md)] border border-border bg-surface px-3 py-2.5">
                <p className="text-xs leading-relaxed text-text-secondary">
                  Send anonymous usage counts (first launch, first chat, benchmark shared) — no
                  prompts, no content, no IP stored
                </p>
                <Toggle checked={usageEvents} onChange={setUsageEvents} />
              </div>
            )}

            {isLast && (
              <div className="mt-4 space-y-3">
                {profile ? (
                  <ul className="space-y-1 rounded-[var(--radius-md)] border border-border bg-surface px-3 py-2.5 text-xs text-text-secondary">
                    <li className="truncate"><span className="text-text-muted">CPU</span> · {profile.cpu.model_name}</li>
                    <li><span className="text-text-muted">RAM</span> · {formatBytes(profile.memory.total_bytes)}</li>
                    <li className="truncate">
                      <span className="text-text-muted">GPU</span> ·{" "}
                      {profile.gpus.length > 0
                        ? `${profile.gpus[0].name} (${formatBytes(profile.gpus[0].vram_bytes)})`
                        : "None detected — CPU inference"}
                    </li>
                  </ul>
                ) : (
                  <p className="flex items-center gap-2 text-xs text-text-muted">
                    <Loader size={12} className="animate-spin" aria-hidden /> Checking your hardware…
                  </p>
                )}
                {pick && (
                  <div className="rounded-[var(--radius-md)] border border-paw-500/30 bg-paw-500/5 px-3 py-2.5">
                    <p className="text-xs text-text-muted">
                      {pick.isFallback ? "A small model to start with" : "Recommended for this device"}
                    </p>
                    <p className="mt-0.5 text-sm font-medium text-text-primary">
                      {pick.model.model_name}{" "}
                      <span className="font-mono text-xs text-text-muted">{formatQuant(pick.model.quantization)}</span>
                    </p>
                    <p className="mt-0.5 text-xs text-text-secondary">
                      {size ? `${size} download` : "Download size unknown"}
                      {pick.model.estimated_tokens_per_sec > 0 &&
                        ` · ${formatToksPerSec(pick.model.estimated_tokens_per_sec)} estimated`}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="flex items-center justify-between border-t border-border bg-surface px-6 py-3">
            <div className="flex gap-1.5" aria-hidden>
              {SLIDES.map((_, i) => (
                <span
                  key={i}
                  className={[
                    "h-1.5 rounded-full transition-all",
                    i === index ? "w-6 bg-paw-500" : "w-1.5 bg-border",
                  ].join(" ")}
                />
              ))}
            </div>
            <div className="flex items-center gap-2">
              {(!isLast || pick) && (
                <button
                  onClick={() => finish()}
                  className="interactive-hover rounded-[var(--radius-md)] px-3 py-1.5 text-xs text-text-muted hover:text-text-secondary"
                >
                  {isLast ? "Not now" : "Skip"}
                </button>
              )}
              {isLast && pick ? (
                <button
                  onClick={() =>
                    finish({ modelId: pick.model.model_id, quant: formatQuant(pick.model.quantization) })
                  }
                  className="interactive-hover inline-flex items-center gap-1.5 rounded-[var(--radius-md)] bg-paw-500 px-3.5 py-1.5 text-sm font-medium text-white hover:bg-paw-600"
                >
                  <Download size={14} aria-hidden />
                  Install {pick.model.model_name}{size ? ` (${size})` : ""}
                </button>
              ) : (
                <button
                  onClick={next}
                  className="interactive-hover inline-flex items-center gap-1.5 rounded-[var(--radius-md)] bg-paw-500 px-3.5 py-1.5 text-sm font-medium text-white hover:bg-paw-600"
                >
                  {isLast ? "Let's go" : "Next"}
                  <ArrowRight size={14} aria-hidden />
                </button>
              )}
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
