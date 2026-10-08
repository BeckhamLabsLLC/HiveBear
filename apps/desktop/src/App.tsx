import type { ReactNode } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Loader } from "lucide-react";
import { ToastProvider } from "./components/Toast";
import UpdateNotification from "./components/UpdateNotification";
import ErrorBoundary from "./components/ErrorBoundary";
import WelcomeModal from "./components/WelcomeModal";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import ModelBrowser from "./pages/ModelBrowser";
import Chat from "./pages/Chat";
import Benchmark from "./pages/Benchmark";
import MeshStatus from "./pages/MeshStatus";
import Account from "./pages/Account";
import Settings from "./pages/Settings";
import { useAppReady } from "./hooks/useAppReady";

/**
 * Hold the routed UI until the backend has finished profiling hardware and
 * opening its databases. The window itself shows immediately; the welcome
 * screen and the update check do not need that state, so they sit outside.
 */
function AppReadyGate({ children }: { children: ReactNode }) {
  const ready = useAppReady();
  if (ready) return <>{children}</>;
  return (
    <div className="flex h-screen flex-col items-center justify-center gap-3 bg-surface text-text-muted">
      <Loader size={22} className="animate-spin text-paw-500" aria-hidden />
      <p className="text-sm">Starting HiveBear…</p>
      <p className="text-xs">Checking your CPU, memory and GPU.</p>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <ToastProvider>
        <WelcomeModal />
        <UpdateNotification />
        <AppReadyGate>
          <BrowserRouter>
            <Routes>
              <Route element={<Layout />}>
                <Route index element={<Dashboard />} />
                <Route path="models" element={<ModelBrowser />} />
                <Route path="chat" element={<Chat />} />
                <Route path="benchmark" element={<Benchmark />} />
                <Route path="mesh" element={<MeshStatus />} />
                <Route path="account" element={<Account />} />
                <Route path="settings" element={<Settings />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </AppReadyGate>
      </ToastProvider>
    </ErrorBoundary>
  );
}
