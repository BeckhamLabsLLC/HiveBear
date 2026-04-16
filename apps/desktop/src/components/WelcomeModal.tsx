import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ArrowRight, Check, Cpu, Network, Sparkles } from "lucide-react";

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
    body: "Your laptop can't run Llama 3 70B alone — but a few peers together can. When you enable mesh, HiveBear pairs your idle compute with other bears so everyone can run models no single device could handle.",
    bullets: [
      "Mesh is always optional. You control when it runs.",
      "Your device identity is a local keypair. It never leaves your machine.",
    ],
  },
  {
    icon: <Cpu size={20} className="text-paw-500" aria-hidden />,
    title: "One quick check",
    body: "Next, HiveBear will inspect your CPU, memory, GPU, and disk speed so it can recommend models that actually run well on this device. Everything stays local — nothing is uploaded.",
  },
];

export default function WelcomeModal() {
  const done = useHasOnboarded();
  const [index, setIndex] = useState(0);

  const finish = useCallback(() => markOnboarded(), []);
  const next = useCallback(() => {
    if (index >= SLIDES.length - 1) finish();
    else setIndex((i) => i + 1);
  }, [index, finish]);

  if (done) return null;
  const slide = SLIDES[index];
  const isLast = index === SLIDES.length - 1;

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
              {!isLast && (
                <button
                  onClick={finish}
                  className="interactive-hover rounded-[var(--radius-md)] px-3 py-1.5 text-xs text-text-muted hover:text-text-secondary"
                >
                  Skip
                </button>
              )}
              <button
                onClick={next}
                className="interactive-hover inline-flex items-center gap-1.5 rounded-[var(--radius-md)] bg-paw-500 px-3.5 py-1.5 text-sm font-medium text-white hover:bg-paw-600"
              >
                {isLast ? "Let's go" : "Next"}
                <ArrowRight size={14} aria-hidden />
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
