import { useCallback, useEffect, useRef, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { ChatMessage, LoadedModel, ModelInfo } from "../types";
import { userTextMessage, assistantMessage, systemMessage } from "../types";
import { listLoadedModels, loadModel, streamChat, unloadModel } from "../lib/invoke";

export function useLoadedModels() {
  const [models, setModels] = useState<ModelInfo[]>([]);

  const refresh = useCallback(async () => {
    try { setModels(await listLoadedModels()); } catch { /* ignore */ }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  return { models, refresh };
}

export function useModelLoader() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (
    modelPath: string, contextLength?: number, gpuLayers?: number,
  ): Promise<LoadedModel | null> => {
    setLoading(true);
    setError(null);
    try { return await loadModel(modelPath, contextLength, gpuLayers); }
    catch (e) { setError(String(e)); return null; }
    finally { setLoading(false); }
  }, []);

  const unload = useCallback(async (handleId: number) => {
    await unloadModel(handleId);
  }, []);

  return { loading, error, load, unload };
}

export interface DisplayMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  toolCall?: { name: string; arguments: string; callId: string };
  toolCallId?: string;
}

export function useChat(handleId: number | null) {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const streamBuffer = useRef("");
  const messagesRef = useRef<DisplayMessage[]>([]);
  messagesRef.current = messages;

  const send = useCallback(async (text: string): Promise<string | null> => {
    if (!handleId || streaming) return null;

    const userMsg: DisplayMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);

    // Build ChatMessage array matching Rust serde format: {role, content}
    const chatMessages: ChatMessage[] = messagesRef.current.concat(userMsg).map((m) => {
      switch (m.role) {
        case "system": return systemMessage(m.content);
        case "assistant": return assistantMessage(m.content);
        default: return userTextMessage(m.content);
      }
    });

    // Add empty assistant message for streaming
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
    setStreaming(true);
    streamBuffer.current = "";

    let tokenUnlisten: UnlistenFn | undefined;
    let errorUnlisten: UnlistenFn | undefined;
    let fullReply: string | null = null;

    try {
      tokenUnlisten = await listen<string>("chat-token", (event) => {
        streamBuffer.current += event.payload;
        const content = streamBuffer.current;
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content };
          return updated;
        });
      });

      errorUnlisten = await listen<string>("chat-error", (event) => {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: `Error: ${event.payload}` };
          return updated;
        });
      });

      fullReply = await streamChat(handleId, chatMessages);
    } catch (e) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: "assistant", content: `Error: ${String(e)}` };
        return updated;
      });
    } finally {
      tokenUnlisten?.();
      errorUnlisten?.();
      setStreaming(false);
    }

    return fullReply;
  }, [handleId, streaming]);

  const clear = useCallback(() => setMessages([]), []);

  return { messages, streaming, send, clear, setMessages };
}
