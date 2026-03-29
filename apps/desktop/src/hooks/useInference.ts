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
  const stopRef = useRef(false);
  const unlistenRefs = useRef<{ token?: UnlistenFn; error?: UnlistenFn }>({});
  messagesRef.current = messages;

  const send = useCallback(async (
    text: string,
    opts?: { temperature?: number; maxTokens?: number },
  ): Promise<string | null> => {
    if (!handleId || streaming) return null;

    stopRef.current = false;
    const userMsg: DisplayMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);

    const chatMessages: ChatMessage[] = messagesRef.current.concat(userMsg).map((m) => {
      switch (m.role) {
        case "system": return systemMessage(m.content);
        case "assistant": return assistantMessage(m.content);
        default: return userTextMessage(m.content);
      }
    });

    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
    setStreaming(true);
    streamBuffer.current = "";

    let fullReply: string | null = null;

    try {
      unlistenRefs.current.token = await listen<string>("chat-token", (event) => {
        if (stopRef.current) return;
        streamBuffer.current += event.payload;
        const content = streamBuffer.current;
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content };
          return updated;
        });
      });

      unlistenRefs.current.error = await listen<string>("chat-error", (event) => {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: `Error: ${event.payload}` };
          return updated;
        });
      });

      fullReply = await streamChat(handleId, chatMessages, opts?.temperature, opts?.maxTokens);
    } catch (e) {
      if (!stopRef.current) {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: `Error: ${String(e)}` };
          return updated;
        });
      }
    } finally {
      unlistenRefs.current.token?.();
      unlistenRefs.current.error?.();
      unlistenRefs.current = {};
      setStreaming(false);
    }

    // If stopped, finalize with whatever we have
    if (stopRef.current && streamBuffer.current) {
      fullReply = streamBuffer.current;
    }

    return fullReply;
  }, [handleId, streaming]);

  const stop = useCallback(() => {
    stopRef.current = true;
    // Unlisten immediately so we stop appending tokens
    unlistenRefs.current.token?.();
    unlistenRefs.current.error?.();
    unlistenRefs.current = {};
    // Finalize the partial content
    if (streamBuffer.current) {
      const content = streamBuffer.current;
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: "assistant", content };
        return updated;
      });
    }
    setStreaming(false);
  }, []);

  const regenerate = useCallback(async (
    opts?: { temperature?: number; maxTokens?: number },
  ): Promise<string | null> => {
    // Find the last user message and remove the last assistant response
    const msgs = [...messagesRef.current];
    if (msgs.length < 2) return null;
    const lastAssistant = msgs[msgs.length - 1];
    const lastUser = msgs[msgs.length - 2];
    if (lastAssistant.role !== "assistant" || lastUser.role !== "user") return null;

    // Remove last assistant message
    setMessages(msgs.slice(0, -1));
    // Wait for state to settle, then resend
    // We need to temporarily clear the ref so send() reads the truncated list
    messagesRef.current = msgs.slice(0, -1);

    return send(lastUser.content, opts);
  }, [send]);

  const editAndResend = useCallback(async (
    index: number,
    newText: string,
    opts?: { temperature?: number; maxTokens?: number },
  ): Promise<string | null> => {
    // Truncate to the message being edited (remove it and everything after)
    const truncated = messagesRef.current.slice(0, index);
    setMessages(truncated);
    messagesRef.current = truncated;

    return send(newText, opts);
  }, [send]);

  const clear = useCallback(() => setMessages([]), []);

  return { messages, streaming, send, stop, regenerate, editAndResend, clear, setMessages };
}
