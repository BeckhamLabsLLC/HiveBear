import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { appReady } from "../lib/invoke";

// The backend profiles hardware and opens its databases on a background
// thread so the window can paint immediately. Until it finishes, commands
// that need that state reject, so anything that calls them waits on this.
let ready = false;
const waiters = new Set<() => void>();

function markReady() {
  if (ready) return;
  ready = true;
  for (const w of waiters) w();
  waiters.clear();
}

let started = false;
function start() {
  if (started) return;
  started = true;
  void listen("app-ready", markReady).catch(() => {});
  // The event may have fired before we subscribed, so also ask, and keep
  // asking: profiling normally takes well under a few seconds.
  const poll = async () => {
    if (ready) return;
    try {
      if (await appReady()) { markReady(); return; }
    } catch { /* backend not reachable yet */ }
    setTimeout(poll, 250);
  };
  void poll();
}

export function useAppReady(): boolean {
  const [isReady, setIsReady] = useState(ready);
  useEffect(() => {
    if (ready) { setIsReady(true); return; }
    start();
    const w = () => setIsReady(true);
    waiters.add(w);
    return () => { waiters.delete(w); };
  }, []);
  return isReady;
}

/** Resolves once the backend is ready (immediately if it already is). */
export function whenAppReady(): Promise<void> {
  if (ready) return Promise.resolve();
  start();
  return new Promise((resolve) => { waiters.add(() => resolve()); });
}
