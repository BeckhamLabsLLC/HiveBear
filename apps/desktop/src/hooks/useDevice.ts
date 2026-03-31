import { useEffect, useSyncExternalStore, useState } from "react";
import { getDeviceStatus, type DeviceStatus } from "../lib/invoke";

/** Whether the viewport is mobile-sized (<640px). SSR-safe, sync, no flicker. */
export function useIsMobile() {
  return useSyncExternalStore(
    (callback) => {
      const mq = window.matchMedia("(max-width: 639px)");
      mq.addEventListener("change", callback);
      return () => mq.removeEventListener("change", callback);
    },
    () => window.matchMedia("(max-width: 639px)").matches,
    () => false, // server snapshot fallback
  );
}

/** Full device status including battery, WiFi, and thermal state. */
export function useDeviceStatus() {
  const [status, setStatus] = useState<DeviceStatus | null>(null);

  useEffect(() => {
    getDeviceStatus()
      .then(setStatus)
      .catch(() => {});

    // Re-poll every 30s for battery/thermal changes
    const interval = setInterval(() => {
      getDeviceStatus().then(setStatus).catch(() => {});
    }, 30_000);

    return () => clearInterval(interval);
  }, []);

  return status;
}
