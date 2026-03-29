import { useCallback, useEffect, useState } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import Sidebar from "./Sidebar";
import CommandPalette from "./CommandPalette";

export default function Layout() {
  const location = useLocation();
  const navigate = useNavigate();
  const [paletteOpen, setPaletteOpen] = useState(false);

  // Global keyboard shortcuts
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
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
    [navigate],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

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
