import { useState, useEffect, useCallback } from 'react';
import type { Conversation } from '../types';
import {
  listConversations,
  createConversation,
  getConversationMessages,
  addMessage as invokeAddMessage,
  deleteConversation,
  renameConversation,
} from '../lib/invoke';

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const convs = await listConversations();
      setConversations(convs);
    } catch (e) {
      console.error('Failed to list conversations:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const create = useCallback(async (title: string, modelId: string) => {
    const conv = await createConversation(title, modelId);
    await refresh();
    setActiveId(conv.id);
    return conv;
  }, [refresh]);

  const remove = useCallback(async (id: string) => {
    await deleteConversation(id);
    if (activeId === id) setActiveId(null);
    await refresh();
  }, [activeId, refresh]);

  const rename = useCallback(async (id: string, newTitle: string) => {
    await renameConversation(id, newTitle);
    await refresh();
  }, [refresh]);

  const getMessages = useCallback(async (id: string) => {
    return getConversationMessages(id);
  }, []);

  const addMessage = useCallback(async (conversationId: string, role: string, content: string) => {
    return invokeAddMessage(conversationId, role, content);
  }, []);

  return {
    conversations, activeId, setActiveId, loading,
    create, remove, rename, getMessages, addMessage, refresh,
  };
}
