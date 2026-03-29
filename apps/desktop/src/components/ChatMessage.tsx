import type { DisplayMessage } from "../hooks/useInference";
import { User, Bot, Wrench } from "lucide-react";
import MarkdownContent from "./MarkdownContent";

interface ChatMessageProps {
  message: DisplayMessage;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";
  const isTool = message.role === "tool";

  if (isTool) {
    return (
      <div className="flex gap-3">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-amber-500/20 text-amber-400">
          <Wrench size={14} />
        </div>
        <div className="max-w-[75%] rounded-2xl border border-amber-500/20 bg-amber-950/20 px-4 py-2.5 text-sm">
          {message.toolCallId && (
            <span className="mb-1 block text-xs font-medium text-amber-400">
              Tool Result
            </span>
          )}
          <pre className="whitespace-pre-wrap font-mono text-xs text-text-secondary">
            {message.content}
          </pre>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${
        isUser ? "bg-paw-500/20 text-paw-400" : "bg-surface-overlay text-text-muted"
      }`}>
        {isUser ? <User size={14} /> : <Bot size={14} />}
      </div>
      <div className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
        isUser ? "bg-paw-500 text-white" : "bg-surface-raised text-text-primary"
      }`}>
        {message.role === "assistant" && message.toolCall ? (
          <div>
            <span className="mb-1 block text-xs font-medium text-paw-400">
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
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <MarkdownContent content={message.content} />
        )}
        {!isUser && message.content === "" && !message.toolCall && (
          <span className="inline-block h-4 w-1 animate-pulse bg-text-muted" />
        )}
      </div>
    </div>
  );
}
