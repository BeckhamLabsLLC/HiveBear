// Hand-off for "install this model" between components that are not mounted
// together. The welcome screen sits outside the router and closes itself on
// Install, so it cannot show download progress; the Dashboard can. The
// request waits here until the Dashboard picks it up.

export interface InstallRequest {
  modelId: string;
  quant?: string;
}

let pending: InstallRequest | null = null;
const listeners = new Set<() => void>();

export function requestInstall(req: InstallRequest) {
  pending = req;
  for (const l of listeners) l();
}

export function takePendingInstall(): InstallRequest | null {
  const p = pending;
  pending = null;
  return p;
}

export function onInstallRequest(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}
