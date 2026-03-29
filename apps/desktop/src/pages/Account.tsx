import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useAccount, useUsage, useApiKeys } from "../hooks/useAccount";
import {
  UserCircle,
  KeyRound,
  Shield,
  Copy,
  Trash2,
  Plus,
  Fingerprint,
  Mail,
  Crown,
  Eye,
  EyeOff,
} from "lucide-react";

// ── Tier badge ──────────────────────────────────────────────────────

function TierBadge({ tier }: { tier: string }) {
  const styles: Record<string, string> = {
    community: "bg-zinc-700/50 text-zinc-300 border-zinc-600",
    pro: "bg-amber-900/30 text-amber-300 border-amber-700",
    team: "bg-blue-900/30 text-blue-300 border-blue-700",
    enterprise: "bg-purple-900/30 text-purple-300 border-purple-700",
  };

  const icons: Record<string, typeof Crown> = {
    pro: Shield,
    team: KeyRound,
    enterprise: Crown,
  };

  const Icon = icons[tier];

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium capitalize ${styles[tier] ?? styles.community}`}
    >
      {Icon && <Icon size={12} />}
      {tier}
    </span>
  );
}

// ── Auth choice (when not logged in) ────────────────────────────────

function AuthChoice({
  onEmail,
  onDevice,
  loading,
  error,
}: {
  onEmail: () => void;
  onDevice: () => void;
  loading: boolean;
  error: string | null;
}) {
  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div className="text-center">
        <h2 className="text-xl font-semibold text-text-primary">Choose how to sign in</h2>
        <p className="mt-1 text-sm text-text-secondary">
          Both paths give you the same features. Pick what suits you.
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
          {error}
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        <button
          onClick={onEmail}
          disabled={loading}
          className="flex flex-col items-center gap-3 rounded-xl border border-border bg-surface-raised p-6 text-center transition-colors hover:border-paw-500/50 hover:bg-surface-overlay"
        >
          <Mail size={32} className="text-paw-500" />
          <div>
            <h3 className="font-medium text-text-primary">Email Account</h3>
            <p className="mt-1 text-xs text-text-secondary">
              For teams & billing management. Password recovery included.
            </p>
          </div>
        </button>

        <button
          onClick={onDevice}
          disabled={loading}
          className="flex flex-col items-center gap-3 rounded-xl border border-border bg-surface-raised p-6 text-center transition-colors hover:border-paw-500/50 hover:bg-surface-overlay"
        >
          <Fingerprint size={32} className="text-paw-500" />
          <div>
            <h3 className="font-medium text-text-primary">Device Key</h3>
            <p className="mt-1 text-xs text-text-secondary">
              Maximum privacy. No email required. Mullvad-style anonymous.
            </p>
          </div>
        </button>
      </div>
    </div>
  );
}

// ── Email login/register form ───────────────────────────────────────

function EmailForm({
  onBack,
  onLogin,
  onRegister,
  loading,
  error,
}: {
  onBack: () => void;
  onLogin: (email: string, password: string) => Promise<void>;
  onRegister: (email: string, password: string, name: string) => Promise<void>;
  loading: boolean;
  error: string | null;
}) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === "login") {
      await onLogin(email, password);
    } else {
      await onRegister(email, password, name);
    }
  };

  return (
    <div className="mx-auto max-w-sm space-y-4">
      <button
        onClick={onBack}
        className="text-sm text-text-secondary hover:text-text-primary"
      >
        &larr; Back
      </button>

      <div className="flex rounded-lg border border-border bg-surface p-0.5">
        <button
          onClick={() => setMode("login")}
          className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${mode === "login" ? "bg-surface-overlay text-text-primary" : "text-text-secondary"}`}
        >
          Sign In
        </button>
        <button
          onClick={() => setMode("register")}
          className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${mode === "register" ? "bg-surface-overlay text-text-primary" : "text-text-secondary"}`}
        >
          Register
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-3">
        {mode === "register" && (
          <input
            type="text"
            placeholder="Display name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text-primary placeholder:text-text-muted"
            required
          />
        )}
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text-primary placeholder:text-text-muted"
          required
        />
        <div className="relative">
          <input
            type={showPassword ? "text" : "password"}
            placeholder="Password (min 8 characters)"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-lg border border-border bg-surface px-3 py-2 pr-10 text-sm text-text-primary placeholder:text-text-muted"
            required
            minLength={8}
          />
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-2 top-2 text-text-muted hover:text-text-secondary"
          >
            {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
          </button>
        </div>
        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-lg bg-paw-500 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-paw-600 disabled:opacity-50"
        >
          {loading ? "..." : mode === "login" ? "Sign In" : "Create Account"}
        </button>
      </form>
    </div>
  );
}

// ── Authenticated account view ──────────────────────────────────────

function AccountDashboard({
  account,
  onLogout,
}: {
  account: NonNullable<ReturnType<typeof useAccount>["account"]>;
  onLogout: () => void;
}) {
  const { summary, loading: usageLoading, refresh: refreshUsage } = useUsage();
  const { keys, loading: keysLoading, refresh: refreshKeys, create, revoke } = useApiKeys();
  const [newKeyLabel, setNewKeyLabel] = useState("");
  const [newKey, setNewKey] = useState<string | null>(null);

  useEffect(() => {
    refreshUsage();
    refreshKeys();
  }, [refreshUsage, refreshKeys]);

  const handleCreateKey = async () => {
    const label = newKeyLabel.trim() || "default";
    const result = await create(label);
    setNewKey(result.key);
    setNewKeyLabel("");
  };

  const handleUpgrade = async (plan: string) => {
    try {
      const url = await invoke<string>("create_checkout", { plan });
      // Open in system browser
      await invoke("plugin:shell|open", { path: url });
    } catch (e) {
      alert(String(e));
    }
  };

  return (
    <div className="space-y-6">
      {/* Identity section */}
      <section className="rounded-xl border border-border bg-surface-raised p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {account.auth_mode === "device" ? (
              <Fingerprint size={24} className="text-paw-500" />
            ) : (
              <UserCircle size={24} className="text-paw-500" />
            )}
            <div>
              {account.auth_mode === "email" ? (
                <>
                  <p className="font-medium text-text-primary">{account.display_name}</p>
                  <p className="text-sm text-text-secondary">{account.email}</p>
                </>
              ) : (
                <>
                  <p className="font-medium text-text-primary">Anonymous Device</p>
                  <p className="font-mono text-sm text-text-secondary">
                    {account.pubkey?.slice(0, 16)}...
                  </p>
                </>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
            <TierBadge tier={account.tier} />
            <button
              onClick={onLogout}
              className="text-sm text-text-secondary hover:text-danger"
            >
              Sign out
            </button>
          </div>
        </div>
      </section>

      {/* Subscription section */}
      <section className="rounded-xl border border-border bg-surface-raised p-5">
        <h3 className="mb-3 text-sm font-medium text-text-secondary">Subscription</h3>
        {account.tier === "community" ? (
          <div className="space-y-3">
            <p className="text-sm text-text-primary">
              You're on the Community plan. Upgrade for priority mesh, private hives, and unlimited cloud.
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => handleUpgrade("pro")}
                className="rounded-lg bg-paw-500 px-4 py-2 text-sm font-medium text-white hover:bg-paw-600"
              >
                Upgrade to Pro — $15/mo
              </button>
              {account.auth_mode === "email" && (
                <button
                  onClick={() => handleUpgrade("team")}
                  className="rounded-lg border border-border px-4 py-2 text-sm font-medium text-text-primary hover:bg-surface-overlay"
                >
                  Team — $10/seat/mo
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-between">
            <p className="text-sm text-text-primary">
              <TierBadge tier={account.tier} /> plan active
              {account.current_period_end && (
                <span className="ml-2 text-text-secondary">
                  · renews {new Date(account.current_period_end).toLocaleDateString()}
                </span>
              )}
            </p>
          </div>
        )}
      </section>

      {/* Usage section */}
      <section className="rounded-xl border border-border bg-surface-raised p-5">
        <h3 className="mb-3 text-sm font-medium text-text-secondary">Usage</h3>
        {usageLoading ? (
          <p className="text-sm text-text-muted">Loading...</p>
        ) : summary ? (
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-2xl font-bold text-text-primary">{summary.cloud_requests_today}</p>
              <p className="text-xs text-text-secondary">
                Cloud requests today
                {summary.cloud_requests_limit && ` / ${summary.cloud_requests_limit}`}
              </p>
            </div>
            <div>
              <p className="text-2xl font-bold text-text-primary">
                {Math.round(summary.mesh_seconds_contributed / 60)}m
              </p>
              <p className="text-xs text-text-secondary">Mesh contributed</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-text-primary">{summary.model_downloads_this_month}</p>
              <p className="text-xs text-text-secondary">Downloads this month</p>
            </div>
          </div>
        ) : (
          <p className="text-sm text-text-muted">Offline — usage data unavailable</p>
        )}
      </section>

      {/* API Keys section */}
      <section className="rounded-xl border border-border bg-surface-raised p-5">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-medium text-text-secondary">API Keys</h3>
          <div className="flex items-center gap-2">
            <input
              type="text"
              placeholder="Label"
              value={newKeyLabel}
              onChange={(e) => setNewKeyLabel(e.target.value)}
              className="rounded-md border border-border bg-surface px-2 py-1 text-xs text-text-primary placeholder:text-text-muted"
            />
            <button
              onClick={handleCreateKey}
              className="flex items-center gap-1 rounded-md bg-paw-500 px-2 py-1 text-xs text-white hover:bg-paw-600"
            >
              <Plus size={12} /> Create
            </button>
          </div>
        </div>

        {newKey && (
          <div className="mb-3 rounded-lg border border-success/30 bg-success/10 p-3">
            <p className="mb-1 text-xs font-medium text-success">New API key created — copy it now, it won't be shown again:</p>
            <div className="flex items-center gap-2">
              <code className="flex-1 break-all rounded bg-surface px-2 py-1 font-mono text-xs text-text-primary">
                {newKey}
              </code>
              <button
                onClick={() => navigator.clipboard.writeText(newKey)}
                className="text-text-secondary hover:text-text-primary"
              >
                <Copy size={14} />
              </button>
            </div>
          </div>
        )}

        {keysLoading ? (
          <p className="text-sm text-text-muted">Loading...</p>
        ) : keys.length === 0 ? (
          <p className="text-sm text-text-muted">No API keys yet.</p>
        ) : (
          <div className="space-y-2">
            {keys.map((key) => (
              <div
                key={key.id}
                className="flex items-center justify-between rounded-lg border border-border bg-surface px-3 py-2"
              >
                <div>
                  <code className="text-xs text-text-primary">{key.key_prefix}...</code>
                  <span className="ml-2 text-xs text-text-secondary">{key.label}</span>
                </div>
                <button
                  onClick={() => revoke(key.id)}
                  className="text-text-muted hover:text-danger"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

// ── Main Account page ───────────────────────────────────────────────

export default function Account() {
  const { account, loading, error, login, register, activateDevice, logout } = useAccount();
  const [showEmail, setShowEmail] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-text-muted">Loading account...</p>
      </div>
    );
  }

  if (account) {
    return (
      <div className="mx-auto max-w-2xl space-y-4 p-6">
        <h1 className="text-lg font-semibold text-text-primary">Account</h1>
        <AccountDashboard account={account} onLogout={logout} />
      </div>
    );
  }

  // Not authenticated
  return (
    <div className="flex h-full flex-col items-center justify-center p-6">
      {showEmail ? (
        <EmailForm
          onBack={() => setShowEmail(false)}
          onLogin={async (e, p) => {
            setSubmitting(true);
            try {
              await login(e, p);
            } finally {
              setSubmitting(false);
            }
          }}
          onRegister={async (e, p, n) => {
            setSubmitting(true);
            try {
              await register(e, p, n);
            } finally {
              setSubmitting(false);
            }
          }}
          loading={submitting}
          error={error}
        />
      ) : (
        <AuthChoice
          onEmail={() => setShowEmail(true)}
          onDevice={async () => {
            setSubmitting(true);
            try {
              await activateDevice();
            } finally {
              setSubmitting(false);
            }
          }}
          loading={submitting}
          error={error}
        />
      )}
    </div>
  );
}
