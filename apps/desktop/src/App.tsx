import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ToastProvider } from "./components/Toast";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import ModelBrowser from "./pages/ModelBrowser";
import Chat from "./pages/Chat";
import Benchmark from "./pages/Benchmark";
import MeshStatus from "./pages/MeshStatus";
import Account from "./pages/Account";
import Settings from "./pages/Settings";

export default function App() {
  return (
    <ToastProvider>
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
    </ToastProvider>
  );
}
