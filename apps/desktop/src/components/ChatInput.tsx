import { useCallback, useRef, useEffect, type KeyboardEvent } from "react";
import { Send, Square } from "lucide-react";
import ChatParamsPanel, { type ChatParams } from "./ChatParamsPanel";

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  streaming: boolean;
  disabled: boolean;
  /** Currently selected model display name */
  modelName: string | null;
  /** All loaded models for the selector */
  loadedModels: Array<{ handle_id: number; label: string; engine: string }>;
  activeHandleId: number | null;
  onModelChange: (handleId: number) => void;
  chatParams: ChatParams;
  onChatParamsChange: (params: ChatParams) => void;
}

export default function ChatInput({
  value,
  onChange,
  onSend,
  onStop,
  streaming,
  disabled,
  modelName,
  loadedModels,
  activeHandleId,
  onModelChange,
  chatParams,
  onChatParamsChange,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [value]);

  // Focus on mount
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!streaming && value.trim() && !disabled) {
          onSend();
        }
      }
    },
    [onSend, streaming, value, disabled],
  );

  const showModelSelector = loadedModels.length > 1;

  return (
    <div className="shrink-0 border-t border-border px-6 py-4">
      <div className="mx-auto max-w-2xl">
        {/* Model badge + params row */}
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            {modelName && (
              <div className="relative" ref={modelMenuRef}>
                {showModelSelector ? (
                  <select
                    value={activeHandleId ?? ""}
                    onChange={(e) => onModelChange(Number(e.target.value))}
                    className="interactive-hover appearance-none rounded-[var(--radius-sm)] border border-border bg-surface-raised px-2 py-1 pr-6 text-[11px] font-medium text-text-secondary outline-none hover:border-paw-500/30 hover:text-text-primary focus:border-paw-500"
                  >
                    {loadedModels.map((m) => (
                      <option key={m.handle_id} value={m.handle_id}>
                        {m.label} ({m.engine})
                      </option>
                    ))}
                  </select>
                ) : (
                  <span className="flex items-center gap-1 rounded-[var(--radius-sm)] bg-surface-raised px-2 py-1 text-[11px] font-medium text-text-secondary">
                    <span className="h-1.5 w-1.5 rounded-full bg-success" />
                    {modelName}
                  </span>
                )}
              </div>
            )}
          </div>
          <div className="flex items-center gap-1">
            <ChatParamsPanel params={chatParams} onChange={onChatParamsChange} />
          </div>
        </div>

        {/* Input container */}
        <div className="relative rounded-[var(--radius-lg)] border border-border bg-surface-raised shadow-[var(--shadow-raised)] focus-within:border-paw-500/50 focus-within:ring-2 focus-within:ring-paw-500/15 interactive-hover">
          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder="Message HiveBear..."
            className="block w-full resize-none bg-transparent px-4 py-3 pr-14 text-sm text-text-primary outline-none placeholder:text-text-muted disabled:cursor-not-allowed disabled:opacity-50"
          />

          {/* Send / Stop button */}
          <div className="absolute bottom-2 right-2">
            {streaming ? (
              <button
                onClick={onStop}
                className="interactive-hover press-scale flex h-8 w-8 items-center justify-center rounded-[var(--radius-md)] bg-surface-overlay text-text-secondary hover:bg-danger/20 hover:text-danger"
                title="Stop generating"
              >
                <Square size={14} fill="currentColor" />
              </button>
            ) : (
              <button
                onClick={onSend}
                disabled={!value.trim() || disabled}
                className="interactive-hover press-scale flex h-8 w-8 items-center justify-center rounded-[var(--radius-md)] bg-paw-500 text-white hover:bg-paw-600 disabled:opacity-30 disabled:hover:bg-paw-500"
                title="Send (Enter)"
              >
                <Send size={14} />
              </button>
            )}
          </div>
        </div>

        {/* Keyboard hint */}
        <div className="mt-1.5 text-center text-[10px] text-text-muted">
          <kbd className="rounded border border-border bg-surface-overlay px-1 py-0.5 font-mono text-[9px]">Enter</kbd>
          {" to send, "}
          <kbd className="rounded border border-border bg-surface-overlay px-1 py-0.5 font-mono text-[9px]">Shift+Enter</kbd>
          {" for new line"}
        </div>
      </div>
    </div>
  );
}
