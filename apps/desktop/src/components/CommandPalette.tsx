import { useState, useCallback, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "motion/react";
import {
  LayoutDashboard, Search, MessageSquare, Gauge, Network,
  Settings, UserCircle, Plus, Command,
} from "lucide-react";

interface CommandItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  action: () => void;
  keywords?: string;
}

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
}

export default function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const commands: CommandItem[] = [
    { id: "new-chat", label: "New Chat", icon: <Plus size={14} />, action: () => navigate("/chat"), keywords: "conversation" },
    { id: "dashboard", label: "Go to Dashboard", icon: <LayoutDashboard size={14} />, action: () => navigate("/") },
    { id: "models", label: "Browse Models", icon: <Search size={14} />, action: () => navigate("/models"), keywords: "search install download" },
    { id: "chat", label: "Open Chat", icon: <MessageSquare size={14} />, action: () => navigate("/chat") },
    { id: "benchmark", label: "Run Benchmark", icon: <Gauge size={14} />, action: () => navigate("/benchmark"), keywords: "performance speed" },
    { id: "mesh", label: "Mesh Status", icon: <Network size={14} />, action: () => navigate("/mesh"), keywords: "p2p peer network" },
    { id: "account", label: "Account", icon: <UserCircle size={14} />, action: () => navigate("/account"), keywords: "login auth subscription" },
    { id: "settings", label: "Settings", icon: <Settings size={14} />, action: () => navigate("/settings"), keywords: "preferences config" },
  ];

  const filtered = query.trim()
    ? commands.filter((c) => {
        const q = query.toLowerCase();
        return c.label.toLowerCase().includes(q) || c.keywords?.toLowerCase().includes(q);
      })
    : commands;

  // Reset on open
  useEffect(() => {
    if (open) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  // Keep selection in bounds
  useEffect(() => {
    if (selectedIndex >= filtered.length) setSelectedIndex(Math.max(0, filtered.length - 1));
  }, [filtered.length, selectedIndex]);

  const handleSelect = useCallback((item: CommandItem) => {
    onClose();
    item.action();
  }, [onClose]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((i) => Math.min(i + 1, filtered.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === "Enter" && filtered[selectedIndex]) {
        e.preventDefault();
        handleSelect(filtered[selectedIndex]);
      } else if (e.key === "Escape") {
        onClose();
      }
    },
    [filtered, selectedIndex, handleSelect, onClose],
  );

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.1 }}
            className="fixed inset-0 z-[200] bg-black/40"
            onClick={onClose}
          />

          {/* Palette */}
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: -8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: -8 }}
            transition={{ type: "spring", damping: 30, stiffness: 500 }}
            className="fixed left-1/2 top-[20%] z-[201] w-full max-w-md -translate-x-1/2 overflow-hidden rounded-[var(--radius-xl)] border border-border bg-surface-raised shadow-[var(--shadow-floating)]"
          >
            {/* Search input */}
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              <Command size={14} className="text-text-muted" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => { setQuery(e.target.value); setSelectedIndex(0); }}
                onKeyDown={handleKeyDown}
                placeholder="Type a command..."
                className="flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-muted"
              />
              <kbd className="rounded-[var(--radius-sm)] border border-border bg-surface-overlay px-1.5 py-0.5 font-mono text-[10px] text-text-muted">
                esc
              </kbd>
            </div>

            {/* Results */}
            <div className="max-h-64 overflow-y-auto py-1">
              {filtered.length === 0 ? (
                <div className="px-4 py-6 text-center text-sm text-text-muted">No results</div>
              ) : (
                filtered.map((item, i) => (
                  <button
                    key={item.id}
                    onClick={() => handleSelect(item)}
                    onMouseEnter={() => setSelectedIndex(i)}
                    className={[
                      "flex w-full items-center gap-3 px-4 py-2 text-left text-sm interactive-hover",
                      i === selectedIndex
                        ? "bg-surface-overlay text-text-primary"
                        : "text-text-secondary",
                    ].join(" ")}
                  >
                    <span className="text-text-muted">{item.icon}</span>
                    {item.label}
                  </button>
                ))
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
