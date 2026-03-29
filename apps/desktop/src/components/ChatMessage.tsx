import type { DisplayMessage } from "../hooks/useInference";
import { User, Bot, Wrench } from "lucide-react";
import MarkdownContent from "./MarkdownContent";
import MessageActions from "./MessageActions";
import StreamingIndicator, { StreamingCursor } from "./StreamingIndicator";

interface ChatMessageProps {
  message: DisplayMessage;
  streaming?: boolean;
  modelName?: string;
  onEdit?: () => void;
  onRegenerate?: () => void;
}

export default function ChatMessage({
  message,
  streaming = false,
  modelName,
  onEdit,
  onRegenerate,
}: ChatMessageProps) {
  const isUser = message.role === "user";
  const isTool = message.role === "tool";
  const isAssistant = message.role === "assistant";
  const isEmpty = message.content === "" && !message.toolCall;

  if (isTool) {
    return (
      <div className="group/msg flex items-start gap-3 rounded-[var(--radius-lg)] border border-warning/10 bg-warning/[0.03] px-4 py-3">
        <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-warning/10 text-warning">
          <Wrench size={12} />
        </div>
        <div className="min-w-0 flex-1">
          {message.toolCallId && (
            <span className="mb-1 block text-[11px] font-medium text-warning">Tool Result</span>
          )}
          <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-text-secondary">
            {message.content}
          </pre>
        </div>
      </div>
    );
  }

  return (
    <div className="group/msg flex items-start gap-3">
      {/* Avatar */}
      <div
        className={[
          "flex h-6 w-6 shrink-0 items-center justify-center rounded-full mt-0.5",
          isUser ? "bg-paw-500/15 text-paw-400" : "bg-surface-overlay text-text-muted",
        ].join(" ")}
      >
        {isUser ? <User size={12} /> : <Bot size={12} />}
      </div>

      {/* Content */}
      <div className="min-w-0 flex-1">
        {/* Role label + actions */}
        <div className="mb-1 flex items-center justify-between">
          <span className="text-[11px] font-medium text-text-muted">
            {isUser ? "You" : modelName || "Assistant"}
          </span>
          {!isEmpty && !streaming && (
            <MessageActions
              content={message.content}
              role={message.role}
              onEdit={isUser ? onEdit : undefined}
              onRegenerate={isAssistant ? onRegenerate : undefined}
            />
          )}
        </div>

        {/* Message body */}
        {isAssistant && message.toolCall ? (
          <div className="rounded-[var(--radius-md)] border border-paw-500/10 bg-paw-500/[0.03] px-3 py-2">
            <span className="mb-1 block text-[11px] font-medium text-paw-400">
              Calling: {message.toolCall.name}
            </span>
            <pre className="whitespace-pre-wrap font-mono text-xs text-text-secondary">
              {(() => {
                try {
                  return JSON.stringify(JSON.parse(message.toolCall.arguments || "{}"), null, 2);
                } catch {
                  return message.toolCall.arguments || "{}";
                }
              })()}
            </pre>
          </div>
        ) : isUser ? (
          <div className="text-sm leading-relaxed text-text-primary whitespace-pre-wrap">
            {message.content}
          </div>
        ) : isEmpty && streaming ? (
          <StreamingIndicator />
        ) : (
          <div className="text-sm leading-relaxed text-text-primary">
            <MarkdownContent content={message.content} />
            {streaming && <StreamingCursor />}
          </div>
        )}
      </div>
    </div>
  );
}
