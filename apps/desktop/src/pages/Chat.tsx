import { useCallback, useEffect, useRef, useState } from "react";
import { useChat, useLoadedModels, useModelLoader } from "../hooks/useInference";
import { useInstalledModels } from "../hooks/useRegistry";
import { useConversations } from "../hooks/useConversations";
import ChatMessage from "../components/ChatMessage";
import type { DisplayMessage } from "../hooks/useInference";
import { Send, Plus, Trash2, Loader, Power, MessageSquare, Pencil, Search, X, Sparkles } from "lucide-react";

export default function Chat() {
  const { models: loaded, refresh: refreshLoaded } = useLoadedModels();
  const { models: installed } = useInstalledModels();
  const { loading: modelLoading, error: loadError, load, unload } = useModelLoader();
  const [activeHandleId, setActiveHandleId] = useState<number | null>(null);
  const { messages, streaming, send, clear, setMessages } = useChat(activeHandleId);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  // Conversation persistence
  const {
    conversations, activeId, setActiveId, loading: convsLoading,
    create, remove, rename, getMessages, addMessage, refresh: refreshConvs,
  } = useConversations();

  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameText, setRenameText] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [showSearch, setShowSearch] = useState(false);

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

  const handleSend = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text) return;
    setInput("");

    // Auto-create conversation if none is active
    let convId = activeId;
    if (!convId && loaded.length > 0) {
      const activeModel = loaded.find((m) => m.handle_id === activeHandleId);
      const modelName = activeModel ? activeModel.model_path.split("/").pop() || "chat" : "chat";
      const title = text.length > 40 ? text.slice(0, 40) + "..." : text;
      try {
        const conv = await create(title, modelName);
        convId = conv.id;
      } catch (err) {
        console.error("Failed to create conversation:", err);
      }
    }

    // Persist user message
    if (convId) {
      try {
        await addMessage(convId, "user", text);
      } catch (err) {
        console.error("Failed to persist user message:", err);
      }
    }

    // Send via inference (this updates messages state in useChat)
    const assistantReply = await send(text);

    // Persist assistant reply
    if (convId && assistantReply) {
      try {
        await addMessage(convId, "assistant", assistantReply);
        refreshConvs();
      } catch (err) {
        console.error("Failed to persist assistant message:", err);
      }
    }
  }, [input, activeId, loaded, activeHandleId, create, addMessage, send, refreshConvs]);

  const handleLoadModel = useCallback(async (modelId: string) => {
    const result = await load(modelId);
    if (result) { setActiveHandleId(result.handle_id); refreshLoaded(); }
  }, [load, refreshLoaded]);

  const handleUnload = useCallback(async (handleId: number) => {
    await unload(handleId);
    if (activeHandleId === handleId) setActiveHandleId(null);
    refreshLoaded();
  }, [unload, activeHandleId, refreshLoaded]);

  const handleNewChat = useCallback(() => {
    setActiveId(null);
    clear();
  }, [setActiveId, clear]);

  const handleSelectConversation = useCallback((id: string) => {
    setActiveId(id);
  }, [setActiveId]);

  const handleDeleteConversation = useCallback(async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await remove(id);
    } catch (err) {
      console.error("Failed to delete conversation:", err);
    }
  }, [remove]);

  const handleStartRename = useCallback((id: string, currentTitle: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setRenamingId(id);
    setRenameText(currentTitle);
  }, []);

  const handleSubmitRename = useCallback(async (id: string) => {
    if (renameText.trim()) {
      try {
        await rename(id, renameText.trim());
      } catch (err) {
        console.error("Failed to rename conversation:", err);
      }
    }
    setRenamingId(null);
    setRenameText("");
  }, [rename, renameText]);

  const filteredConversations = searchQuery
    ? conversations.filter((c) => c.title.toLowerCase().includes(searchQuery.toLowerCase()))
    : conversations;

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="flex w-64 shrink-0 flex-col border-r border-border bg-surface">
        {/* Sidebar header */}
        <div className="flex items-center justify-between border-b border-border px-3 py-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">Conversations</span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setShowSearch(!showSearch)}
              className="rounded-md p-1.5 text-text-muted hover:bg-surface-raised hover:text-text-primary"
              title="Search conversations"
            >
              <Search size={14} />
            </button>
            <button
              onClick={handleNewChat}
              className="rounded-md p-1.5 text-text-muted hover:bg-surface-raised hover:text-text-primary"
              title="New conversation"
            >
              <Plus size={14} />
            </button>
          </div>
        </div>

        {/* Search bar */}
        {showSearch && (
          <div className="border-b border-border px-3 py-2">
            <div className="relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search..."
                className="w-full rounded-md border border-border bg-surface-raised px-2.5 py-1.5 pr-7 text-xs outline-none placeholder:text-text-muted focus:border-paw-500"
                autoFocus
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery("")}
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary"
                >
                  <X size={12} />
                </button>
              )}
            </div>
          </div>
        )}

        {/* Conversation list */}
        <div className="flex-1 overflow-y-auto">
          {convsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader size={16} className="animate-spin text-text-muted" />
            </div>
          ) : filteredConversations.length === 0 ? (
            <div className="px-3 py-8 text-center text-xs text-text-muted">
              {searchQuery ? "No matching conversations." : "No conversations yet."}
            </div>
          ) : (
            <div className="py-1">
              {filteredConversations.map((conv) => (
                <div
                  key={conv.id}
                  onClick={() => handleSelectConversation(conv.id)}
                  className={`group mx-1.5 mb-0.5 flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-2 text-xs transition-colors ${
                    activeId === conv.id
                      ? "bg-paw-500/10 text-paw-400"
                      : "text-text-secondary hover:bg-surface-raised hover:text-text-primary"
                  }`}
                >
                  <MessageSquare size={14} className="shrink-0" />
                  <div className="min-w-0 flex-1">
                    {renamingId === conv.id ? (
                      <input
                        type="text"
                        value={renameText}
                        onChange={(e) => setRenameText(e.target.value)}
                        onBlur={() => handleSubmitRename(conv.id)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handleSubmitRename(conv.id);
                          if (e.key === "Escape") { setRenamingId(null); setRenameText(""); }
                        }}
                        className="w-full rounded border border-paw-500 bg-surface-raised px-1 py-0.5 text-xs outline-none"
                        autoFocus
                        onClick={(e) => e.stopPropagation()}
                      />
                    ) : (
                      <span className="block truncate">{conv.title}</span>
                    )}
                    <span className="block text-[10px] text-text-muted">
                      {conv.message_count} message{conv.message_count !== 1 ? "s" : ""}
                    </span>
                  </div>
                  {renamingId !== conv.id && (
                    <div className="flex shrink-0 items-center gap-0.5 opacity-0 group-hover:opacity-100">
                      <button
                        onClick={(e) => handleStartRename(conv.id, conv.title, e)}
                        className="rounded p-1 hover:bg-surface-overlay"
                        title="Rename"
                      >
                        <Pencil size={11} />
                      </button>
                      <button
                        onClick={(e) => handleDeleteConversation(conv.id, e)}
                        className="rounded p-1 hover:bg-surface-overlay hover:text-danger"
                        title="Delete"
                      >
                        <Trash2 size={11} />
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex flex-1 flex-col">
        {/* Top bar */}
        <div className="flex shrink-0 items-center justify-between border-b border-border px-6 py-3">
          <div className="flex items-center gap-3">
            <h1 className="text-base font-semibold">Chat</h1>
            {loaded.length > 0 && (
              <select value={activeHandleId ?? ""} onChange={(e) => setActiveHandleId(e.target.value ? Number(e.target.value) : null)}
                className="rounded-lg border border-border bg-surface px-2.5 py-1.5 text-xs outline-none focus:border-paw-500">
                {loaded.map((m) => (
                  <option key={m.handle_id} value={m.handle_id}>
                    {m.model_path.split("/").pop()} ({m.engine})
                  </option>
                ))}
              </select>
            )}
          </div>
          <div className="flex items-center gap-2">
            {activeHandleId != null && (
              <button onClick={() => handleUnload(activeHandleId)}
                className="rounded-lg border border-border px-2.5 py-1.5 text-xs text-text-secondary hover:border-danger hover:text-danger"
                title="Unload model">
                <Power size={12} />
              </button>
            )}
            <button onClick={handleNewChat}
              className="rounded-lg border border-border px-2.5 py-1.5 text-xs text-text-secondary hover:text-text-primary"
              title="New chat">
              <Plus size={12} />
            </button>
          </div>
        </div>

        {/* Messages or empty state */}
        <div className="flex-1 overflow-y-auto p-6">
          {loaded.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-4 text-text-muted">
              <p className="text-sm">No model loaded. Load one to start chatting.</p>
              {installed.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {installed.slice(0, 4).map((m) => (
                    <button key={m.id} disabled={modelLoading} onClick={() => handleLoadModel(m.id)}
                      className="flex items-center gap-1.5 rounded-lg border border-border px-3 py-2 text-xs hover:border-paw-500 hover:text-paw-400 disabled:opacity-50">
                      {modelLoading ? <Loader size={12} className="animate-spin" /> : <Plus size={12} />}
                      {m.name}
                    </button>
                  ))}
                </div>
              ) : (
                <p className="text-xs">Install a model from the Model Browser first.</p>
              )}
              {loadError && <p className="text-xs text-danger">{loadError}</p>}
            </div>
          ) : messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-6">
              <div className="flex flex-col items-center gap-2">
                <Sparkles size={28} className="text-paw-500" />
                <p className="text-sm font-medium text-text-primary">What can I help you with?</p>
                <p className="text-xs text-text-muted">Try one of these, or type your own message.</p>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {[
                  "Explain quantum computing in simple terms",
                  "Write a Python function to sort a list",
                  "What are the pros and cons of Rust vs Go?",
                  "Help me write a cover letter",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setInput(suggestion)}
                    className="rounded-xl border border-border bg-surface-raised px-4 py-3 text-left text-xs text-text-secondary transition-colors hover:border-paw-500/50 hover:text-text-primary"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-4">
              {messages.map((msg, i) => (
                <ChatMessage key={i} message={msg} />
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Input bar */}
        {loaded.length > 0 && (
          <form onSubmit={handleSend} className="shrink-0 border-t border-border px-6 py-4">
            <div className="mx-auto flex max-w-3xl gap-2">
              <input type="text" value={input} onChange={(e) => setInput(e.target.value)}
                placeholder="Type a message..." disabled={streaming || activeHandleId === null}
                className="flex-1 rounded-xl border border-border bg-surface-raised px-4 py-2.5 text-sm outline-none placeholder:text-text-muted focus:border-paw-500 disabled:opacity-50" />
              <button type="submit" disabled={streaming || !input.trim() || activeHandleId === null}
                className="rounded-xl bg-paw-500 px-4 py-2.5 text-white transition-colors hover:bg-paw-600 disabled:opacity-40">
                {streaming ? <Loader size={16} className="animate-spin" /> : <Send size={16} />}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
