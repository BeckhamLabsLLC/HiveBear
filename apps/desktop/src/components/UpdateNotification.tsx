import { useEffect, useState } from "react";
import { check, type Update } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";
import { motion, AnimatePresence } from "motion/react";
import { Download, X, RefreshCw } from "lucide-react";

type Stage = "available" | "downloading" | "ready" | "error";

export default function UpdateNotification() {
  const [update, setUpdate] = useState<Update | null>(null);
  const [stage, setStage] = useState<Stage>("available");
  const [progress, setProgress] = useState(0);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const checkForUpdate = async () => {
      try {
        const result = await check();
        if (!cancelled && result?.available) {
          setUpdate(result);
        }
      } catch {
        // Silent — update check failures are non-fatal
      }
    };

    checkForUpdate();

    // Re-check every 4 hours
    const interval = setInterval(checkForUpdate, 4 * 60 * 60 * 1000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (!update || dismissed) return null;

  const handleInstall = async () => {
    setStage("downloading");
    try {
      let totalBytes = 0;
      await update.downloadAndInstall((event) => {
        if (event.event === "Started" && event.data.contentLength) {
          totalBytes = event.data.contentLength;
        } else if (event.event === "Progress" && totalBytes > 0) {
          setProgress((prev) => Math.min(prev + (event.data.chunkLength / totalBytes) * 100, 100));
        } else if (event.event === "Finished") {
          setProgress(100);
        }
      });
      setStage("ready");
    } catch {
      setStage("error");
    }
  };

  const handleRelaunch = async () => {
    await relaunch();
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 16, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 8, scale: 0.95 }}
        transition={{ type: "spring", damping: 25, stiffness: 400 }}
        className="fixed bottom-4 right-4 z-50 flex items-center gap-3 rounded-[var(--radius-lg)] border border-border bg-surface-raised px-4 py-3 shadow-[var(--shadow-overlay)] min-w-[300px] max-w-sm"
      >
        {stage === "available" && (
          <>
            <Download size={16} className="shrink-0 text-paw-400" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-text-primary">Update available</p>
              <p className="text-xs text-text-muted truncate">v{update.version}</p>
            </div>
            <button
              onClick={handleInstall}
              className="shrink-0 rounded-[var(--radius-md)] bg-paw-500 px-3 py-1.5 text-xs font-medium text-white hover:bg-paw-600 transition-colors"
            >
              Install
            </button>
          </>
        )}

        {stage === "downloading" && (
          <>
            <RefreshCw size={16} className="shrink-0 text-paw-400 animate-spin" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-text-primary">Downloading...</p>
              <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-overlay">
                <div
                  className="h-full rounded-full bg-paw-500 transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          </>
        )}

        {stage === "ready" && (
          <>
            <Download size={16} className="shrink-0 text-success" />
            <p className="flex-1 text-sm font-medium text-text-primary">Ready to restart</p>
            <button
              onClick={handleRelaunch}
              className="shrink-0 rounded-[var(--radius-md)] bg-success px-3 py-1.5 text-xs font-medium text-white hover:opacity-90 transition-opacity"
            >
              Restart
            </button>
          </>
        )}

        {stage === "error" && (
          <>
            <Download size={16} className="shrink-0 text-danger" />
            <p className="flex-1 text-sm text-danger">Update failed</p>
          </>
        )}

        {stage !== "downloading" && (
          <button
            onClick={() => setDismissed(true)}
            className="shrink-0 interactive-hover rounded-[var(--radius-sm)] p-1 text-text-muted hover:text-text-primary"
          >
            <X size={12} />
          </button>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
