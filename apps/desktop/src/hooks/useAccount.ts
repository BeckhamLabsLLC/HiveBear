import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { AccountInfo, AuthResult, UsageSummary, ApiKeyInfo, NewApiKey } from "../types";

export function useAccount() {
  const [account, setAccount] = useState<AccountInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const info = await invoke<AccountInfo | null>("get_account");
      setAccount(info);
      setError(null);
    } catch {
      // Offline or not authenticated — both are fine
      setAccount(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const login = useCallback(async (email: string, password: string) => {
    setError(null);
    try {
      await invoke<AuthResult>("login", { email, password });
      await refresh();
    } catch (e) {
      setError(String(e));
      throw e;
    }
  }, [refresh]);

  const register = useCallback(async (email: string, password: string, displayName: string) => {
    setError(null);
    try {
      await invoke<AuthResult>("register", { email, password, displayName });
      await refresh();
    } catch (e) {
      setError(String(e));
      throw e;
    }
  }, [refresh]);

  const activateDevice = useCallback(async () => {
    setError(null);
    try {
      await invoke<AuthResult>("activate_device");
      await refresh();
    } catch (e) {
      setError(String(e));
      throw e;
    }
  }, [refresh]);

  const logout = useCallback(async () => {
    try {
      await invoke("logout");
    } catch {
      // Best-effort
    }
    setAccount(null);
  }, []);

  return {
    account,
    loading,
    error,
    login,
    register,
    activateDevice,
    logout,
    refresh,
    isLoggedIn: account !== null,
    isAnonymous: account?.auth_mode === "device",
    isPro: account?.tier === "pro" || account?.tier === "team" || account?.tier === "enterprise",
    isTeam: account?.tier === "team" || account?.tier === "enterprise",
    tier: account?.tier ?? "community",
  };
}

export function useUsage() {
  const [summary, setSummary] = useState<UsageSummary | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await invoke<UsageSummary>("get_usage_summary");
      setSummary(data);
    } catch {
      // Offline — silently fail
    } finally {
      setLoading(false);
    }
  }, []);

  return { summary, loading, refresh };
}

export function useApiKeys() {
  const [keys, setKeys] = useState<ApiKeyInfo[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await invoke<ApiKeyInfo[]>("list_api_keys");
      setKeys(data);
    } catch {
      // Offline
    } finally {
      setLoading(false);
    }
  }, []);

  const create = useCallback(async (label: string): Promise<NewApiKey> => {
    const key = await invoke<NewApiKey>("create_api_key", { label });
    await refresh();
    return key;
  }, [refresh]);

  const revoke = useCallback(async (keyId: string) => {
    await invoke("revoke_api_key", { keyId });
    await refresh();
  }, [refresh]);

  return { keys, loading, refresh, create, revoke };
}
