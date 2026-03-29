import { AnimatePresence, motion } from "motion/react";
import { Settings2, RotateCcw } from "lucide-react";
import { useState } from "react";

export interface ChatParams {
  temperature: number;
  topP: number;
  maxTokens: number;
}

export const DEFAULT_CHAT_PARAMS: ChatParams = {
  temperature: 0.7,
  topP: 0.9,
  maxTokens: 2048,
};

interface ChatParamsPanelProps {
  params: ChatParams;
  onChange: (params: ChatParams) => void;
}

export default function ChatParamsPanel({ params, onChange }: ChatParamsPanelProps) {
  const [open, setOpen] = useState(false);

  const isDefault =
    params.temperature === DEFAULT_CHAT_PARAMS.temperature &&
    params.topP === DEFAULT_CHAT_PARAMS.topP &&
    params.maxTokens === DEFAULT_CHAT_PARAMS.maxTokens;

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className={[
          "interactive-hover rounded-[var(--radius-md)] p-1.5",
          open || !isDefault
            ? "text-paw-400 hover:bg-paw-500/10"
            : "text-text-muted hover:bg-surface-overlay hover:text-text-secondary",
        ].join(" ")}
        title="Chat parameters"
      >
        <Settings2 size={14} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 8, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.96 }}
            transition={{ type: "spring", damping: 25, stiffness: 400 }}
            className="absolute bottom-full right-0 z-50 mb-2 w-64 rounded-[var(--radius-lg)] border border-border bg-surface-raised p-4 shadow-[var(--shadow-overlay)]"
          >
            <div className="mb-3 flex items-center justify-between">
              <span className="text-xs font-medium text-text-secondary">Parameters</span>
              {!isDefault && (
                <button
                  onClick={() => onChange({ ...DEFAULT_CHAT_PARAMS })}
                  className="interactive-hover flex items-center gap-1 rounded-[var(--radius-sm)] px-1.5 py-0.5 text-[10px] text-text-muted hover:text-text-secondary"
                >
                  <RotateCcw size={10} /> Reset
                </button>
              )}
            </div>

            <div className="space-y-4">
              <ParamSlider
                label="Temperature"
                value={params.temperature}
                min={0}
                max={2}
                step={0.1}
                onChange={(v) => onChange({ ...params, temperature: v })}
              />
              <ParamSlider
                label="Top P"
                value={params.topP}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => onChange({ ...params, topP: v })}
              />
              <div>
                <div className="mb-1.5 flex items-baseline justify-between">
                  <label className="text-xs text-text-muted">Max Tokens</label>
                  <span className="font-mono text-[10px] text-text-muted">{params.maxTokens}</span>
                </div>
                <input
                  type="number"
                  min={128}
                  max={131072}
                  step={256}
                  value={params.maxTokens}
                  onChange={(e) => onChange({ ...params, maxTokens: Number(e.target.value) })}
                  className="w-full rounded-[var(--radius-md)] border border-border bg-surface px-2.5 py-1.5 font-mono text-xs text-text-primary outline-none focus:border-paw-500 focus:ring-2 focus:ring-paw-500/20"
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <label className="text-xs text-text-muted">{label}</label>
        <span className="font-mono text-[10px] text-text-muted">{value.toFixed(step < 1 ? 1 : 0)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </div>
  );
}
