import { useCallback, useEffect, useState } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import Sidebar from "./Sidebar";
import BottomTabs from "./BottomTabs";
import CommandPalette from "./CommandPalette";
import { useIsMobile } from "../hooks/useDevice";

export default function Layout() {
  const location = useLocation();
  const navigate = useNavigate();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const isMobile = useIsMobile();

  // Global keyboard shortcuts (desktop only)
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (isMobile) return;
      const mod = e.metaKey || e.ctrlKey;

      if (mod && e.key === "k") {
        e.preventDefault();
        setPaletteOpen((prev) => !prev);
      } else if (mod && e.key === "n") {
        e.preventDefault();
        navigate("/chat");
      } else if (mod && e.key === ",") {
        e.preventDefault();
        navigate("/settings");
      } else if (e.key === "Escape") {
        setPaletteOpen(false);
      }
    },
    [navigate, isMobile],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  if (isMobile) {
    return (
      <div className="flex h-screen flex-col">
        <main className="flex-1 overflow-y-auto overscroll-contain pb-[calc(56px+env(safe-area-inset-bottom))]">
          <div key={location.pathname} className="animate-[page-enter] min-h-0">
            <Outlet />
          </div>
        </main>
        <BottomTabs />
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <div key={location.pathname} className="animate-[page-enter] h-full">
          <Outlet />
        </div>
      </main>

      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </div>
  );
}
