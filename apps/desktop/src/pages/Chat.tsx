import { useCallback, useEffect, useRef, useState } from "react";
import { useChat, useLoadedModels, useModelLoader } from "../hooks/useInference";
import { useInstalledModels } from "../hooks/useRegistry";
import { useConversations } from "../hooks/useConversations";
import ChatMessage from "../components/ChatMessage";
import ChatInput from "../components/ChatInput";
import ConversationDrawer from "../components/ConversationDrawer";
import { DEFAULT_CHAT_PARAMS, type ChatParams } from "../components/ChatParamsPanel";
import type { DisplayMessage } from "../hooks/useInference";
import { PanelLeft, Plus, Loader, Sparkles } from "lucide-react";
import { EmptyState } from "../components/ui";

export default function Chat() {
  const { models: loaded, refresh: refreshLoaded } = useLoadedModels();
  const { models: installed } = useInstalledModels();
  const { loading: modelLoading, error: loadError, load } = useModelLoader();
  const [activeHandleId, setActiveHandleId] = useState<number | null>(null);
  const { messages, streaming, send, stop, regenerate, editAndResend, clear, setMessages } =
    useChat(activeHandleId);
  const [input, setInput] = useState("");
  const [chatParams, setChatParams] = useState<ChatParams>({ ...DEFAULT_CHAT_PARAMS });
  const bottomRef = useRef<HTMLDivElement>(null);

  // Conversation drawer
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Conversation persistence
  const {
    conversations, activeId, setActiveId, loading: convsLoading,
    create, remove, rename, getMessages, addMessage, refresh: refreshConvs,
  } = useConversations();

  // Editing state
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editText, setEditText] = useState("");

  useEffect(() => {
    if (loaded.length > 0 && activeHandleId === null) {
      setActiveHandleId(loaded[0].handle_id);
    }
  }, [loaded, activeHandleId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load messages when conversation changes
  useEffect(() => {
    if (!activeId) {
      clear();
      return;
    }
    (async () => {
      try {
        const persisted = await getMessages(activeId);
        const displayMsgs: DisplayMessage[] = persisted.map((m) => ({
          role: m.role === "User" ? "user" : m.role === "Assistant" ? "assistant" : "system",
          content: m.content,
        }));
        setMessages(displayMsgs);
      } catch (e) {
        console.error("Failed to load messages:", e);
      }
    })();
  }, [activeId, getMessages, clear, setMessages]);

  const handleSend = useCallback(async (text?: string) => {
    const msgText = (text ?? input).trim();
    if (!msgText) return;
    if (!text) setInput("");

    // Auto-create conversation if none is active
    let convId = activeId;
    if (!convId && loaded.length > 0) {
      const activeModel = loaded.find((m) => m.handle_id === activeHandleId);
      const modelName = activeModel ? activeModel.model_path.split("/").pop() || "chat" : "chat";
      const title = msgText.length > 40 ? msgText.slice(0, 40) + "..." : msgText;
      try {
        const conv = await create(title, modelName);
        convId = conv.id;
      } catch (err) {
        console.error("Failed to create conversation:", err);
      }
    }

    // Persist user message
    if (convId) {
      try { await addMessage(convId, "user", msgText); } catch { /* ignore */ }
    }

    // Send via inference
    const assistantReply = await send(msgText, {
      temperature: chatParams.temperature,
      maxTokens: chatParams.maxTokens,
    });

    // Persist assistant reply
    if (convId && assistantReply) {
      try {
        await addMessage(convId, "assistant", assistantReply);
        refreshConvs();
      } catch { /* ignore */ }
    }
  }, [input, activeId, loaded, activeHandleId, create, addMessage, send, refreshConvs, chatParams]);

  const handleLoadModel = useCallback(async (modelId: string) => {
    const result = await load(modelId);
    if (result) { setActiveHandleId(result.handle_id); refreshLoaded(); }
  }, [load, refreshLoaded]);

  const handleNewChat = useCallback(() => {
    setActiveId(null);
    clear();
    setDrawerOpen(false);
  }, [setActiveId, clear]);

  const handleSelectConversation = useCallback((id: string) => {
    setActiveId(id);
  }, [setActiveId]);

  const handleDeleteConversation = useCallback(async (id: string) => {
    try { await remove(id); } catch { /* ignore */ }
  }, [remove]);

  const handleRenameConversation = useCallback(async (id: string, title: string) => {
    try { await rename(id, title); } catch { /* ignore */ }
  }, [rename]);

  const handleRegenerate = useCallback(async () => {
    const reply = await regenerate({
      temperature: chatParams.temperature,
      maxTokens: chatParams.maxTokens,
    });
    if (activeId && reply) {
      try {
        await addMessage(activeId, "assistant", reply);
        refreshConvs();
      } catch { /* ignore */ }
    }
  }, [regenerate, activeId, addMessage, refreshConvs, chatParams]);

  const handleStartEdit = useCallback((index: number) => {
    setEditingIndex(index);
    setEditText(messages[index].content);
  }, [messages]);

  const handleSubmitEdit = useCallback(async () => {
    if (editingIndex === null) return;
    const reply = await editAndResend(editingIndex, editText, {
      temperature: chatParams.temperature,
      maxTokens: chatParams.maxTokens,
    });
    setEditingIndex(null);
    setEditText("");
    if (activeId && reply) {
      try {
        await addMessage(activeId, "assistant", reply);
        refreshConvs();
      } catch { /* ignore */ }
    }
  }, [editingIndex, editText, editAndResend, activeId, addMessage, refreshConvs, chatParams]);

  // Derive model name for display
  const activeModel = loaded.find((m) => m.handle_id === activeHandleId);
  const modelName = activeModel ? activeModel.model_path.split("/").pop() || "Model" : null;

  const loadedModelOptions = loaded.map((m) => ({
    handle_id: m.handle_id,
    label: m.model_path.split("/").pop() || "Model",
    engine: m.engine,
  }));

  return (
    <div className="relative flex h-full flex-col">
      {/* Conversation drawer overlay */}
      <ConversationDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        conversations={conversations}
        activeId={activeId}
        loading={convsLoading}
        onSelect={handleSelectConversation}
        onNew={handleNewChat}
        onDelete={handleDeleteConversation}
        onRename={handleRenameConversation}
      />

      {/* Top bar */}
      <div className="flex shrink-0 items-center justify-between border-b border-border px-4 py-2.5">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setDrawerOpen(!drawerOpen)}
            className="interactive-hover rounded-[var(--radius-md)] p-1.5 text-text-muted hover:bg-surface-raised hover:text-text-primary"
            title="Conversations"
          >
            <PanelLeft size={16} />
          </button>
          <h1 className="text-sm font-medium text-text-primary">
            {activeId
              ? conversations.find((c) => c.id === activeId)?.title || "Chat"
              : "New Chat"}
          </h1>
        </div>
        <button
          onClick={handleNewChat}
          className="interactive-hover rounded-[var(--radius-md)] p-1.5 text-text-muted hover:bg-surface-raised hover:text-text-primary"
          title="New chat"
        >
          <Plus size={16} />
        </button>
      </div>

      {/* Messages or empty state */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        {loaded.length === 0 ? (
          /* No model loaded */
          <div className="flex h-full items-center justify-center">
            <EmptyState
              icon={<Sparkles size={24} />}
              title="No model loaded"
              description="Load an installed model to start chatting."
            >
              {installed.length > 0 ? (
                <div className="mt-2 flex flex-wrap justify-center gap-2">
                  {installed.slice(0, 4).map((m) => (
                    <button
                      key={m.id}
                      disabled={modelLoading}
                      onClick={() => handleLoadModel(m.id)}
                      className="interactive-hover press-scale flex items-center gap-1.5 rounded-[var(--radius-md)] border border-border px-3 py-2 text-xs text-text-secondary hover:border-paw-500/30 hover:text-paw-400 disabled:opacity-50"
                    >
                      {modelLoading ? <Loader size={12} className="animate-spin" /> : null}
                      {m.name}
                    </button>
                  ))}
                </div>
              ) : (
                <p className="mt-1 text-xs text-text-muted">
                  Install a model from the Model Browser first.
                </p>
              )}
              {loadError && <p className="mt-2 text-xs text-danger">{loadError}</p>}
            </EmptyState>
          </div>
        ) : messages.length === 0 ? (
          /* Empty conversation */
          <div className="flex h-full flex-col items-center justify-center">
            <div className="flex flex-col items-center gap-3 animate-[fade-in]">
              <div className="flex h-16 w-16 items-center justify-center rounded-[var(--radius-xl)] bg-paw-500/10">
                <Sparkles size={28} className="text-paw-500" />
              </div>
              <div className="text-center">
                <p className="text-sm font-medium text-text-primary">
                  {modelName ? `Chat with ${modelName}` : "What can I help you with?"}
                </p>
                <p className="mt-1 text-xs text-text-muted">
                  Try a suggestion or type your own message.
                </p>
              </div>

              <div className="mt-4 flex w-full max-w-md flex-col gap-2">
                {[
                  { title: "Explain quantum computing", desc: "Break it down in simple terms" },
                  { title: "Write a Python sort function", desc: "With type hints and docstring" },
                  { title: "Compare Rust vs Go", desc: "Pros, cons, and use cases" },
                ].map((s) => (
                  <button
                    key={s.title}
                    onClick={() => handleSend(s.title + " — " + s.desc)}
                    className="interactive-hover press-scale rounded-[var(--radius-lg)] border border-border bg-surface-raised px-4 py-3 text-left hover:border-paw-500/30 hover:shadow-[var(--shadow-raised)]"
                  >
                    <span className="text-xs font-medium text-text-primary">{s.title}</span>
                    <span className="mt-0.5 block text-[11px] text-text-muted">{s.desc}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          /* Messages */
          <div className="mx-auto max-w-2xl space-y-6">
            {messages.map((msg, i) => {
              const isLastMessage = i === messages.length - 1;
              const isStreaming = streaming && isLastMessage && msg.role === "assistant";

              if (editingIndex === i && msg.role === "user") {
                return (
                  <div key={i} className="flex items-start gap-3">
                    <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-paw-500/15 text-paw-400 mt-0.5">
                      <span className="text-[10px] font-bold">You</span>
                    </div>
                    <div className="min-w-0 flex-1">
                      <textarea
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        className="w-full resize-none rounded-[var(--radius-md)] border border-paw-500 bg-surface-raised px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-paw-500/20"
                        rows={3}
                        autoFocus
                      />
                      <div className="mt-2 flex gap-2">
                        <button
                          onClick={handleSubmitEdit}
                          className="interactive-hover press-scale rounded-[var(--radius-md)] bg-paw-500 px-3 py-1.5 text-xs font-medium text-white hover:bg-paw-600"
                        >
                          Save & Submit
                        </button>
                        <button
                          onClick={() => { setEditingIndex(null); setEditText(""); }}
                          className="interactive-hover rounded-[var(--radius-md)] px-3 py-1.5 text-xs text-text-muted hover:text-text-primary"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  </div>
                );
              }

              return (
                <ChatMessage
                  key={i}
                  message={msg}
                  streaming={isStreaming}
                  modelName={modelName || undefined}
                  onEdit={msg.role === "user" && !streaming ? () => handleStartEdit(i) : undefined}
                  onRegenerate={
                    msg.role === "assistant" && isLastMessage && !streaming
                      ? handleRegenerate
                      : undefined
                  }
                />
              );
            })}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input */}
      {loaded.length > 0 && (
        <ChatInput
          value={input}
          onChange={setInput}
          onSend={() => handleSend()}
          onStop={stop}
          streaming={streaming}
          disabled={activeHandleId === null}
          modelName={modelName}
          loadedModels={loadedModelOptions}
          activeHandleId={activeHandleId}
          onModelChange={setActiveHandleId}
          chatParams={chatParams}
          onChatParamsChange={setChatParams}
        />
      )}
    </div>
  );
}
