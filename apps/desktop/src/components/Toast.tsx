import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { X, CheckCircle2, AlertCircle, Info } from "lucide-react";

type ToastType = "success" | "error" | "info";

interface Toast {
  id: number;
  message: string;
  type: ToastType;
}

interface ToastContextValue {
  toast: (message: string, type?: ToastType) => void;
}

const ToastContext = createContext<ToastContextValue>({ toast: () => {} });

export function useToast() {
  return useContext(ToastContext);
}

let nextId = 0;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const toast = useCallback((message: string, type: ToastType = "info") => {
    const id = nextId++;
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}

      {/* Toast container */}
      <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2">
        <AnimatePresence>
          {toasts.map((t) => (
            <motion.div
              key={t.id}
              initial={{ opacity: 0, y: 16, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.95 }}
              transition={{ type: "spring", damping: 25, stiffness: 400 }}
              className="flex items-center gap-2.5 rounded-[var(--radius-lg)] border border-border bg-surface-raised px-4 py-3 shadow-[var(--shadow-overlay)] min-w-[280px] max-w-sm"
            >
              <ToastIcon type={t.type} />
              <span className="flex-1 text-sm text-text-primary">{t.message}</span>
              <button
                onClick={() => dismiss(t.id)}
                className="interactive-hover rounded-[var(--radius-sm)] p-1 text-text-muted hover:text-text-primary"
              >
                <X size={12} />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

function ToastIcon({ type }: { type: ToastType }) {
  switch (type) {
    case "success":
      return <CheckCircle2 size={16} className="shrink-0 text-success" />;
    case "error":
      return <AlertCircle size={16} className="shrink-0 text-danger" />;
    case "info":
      return <Info size={16} className="shrink-0 text-paw-400" />;
  }
}
