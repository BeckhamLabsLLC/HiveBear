import { NavLink } from "react-router-dom";
import { LayoutDashboard, Search, MessageSquare, Gauge, Network, Settings, UserCircle } from "lucide-react";

const links = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/models", label: "Models", icon: Search },
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/benchmark", label: "Benchmark", icon: Gauge },
  { to: "/mesh", label: "Mesh", icon: Network },
  { to: "/account", label: "Account", icon: UserCircle },
  { to: "/settings", label: "Settings", icon: Settings },
];

export default function Sidebar() {
  return (
    <aside className="flex h-full w-56 shrink-0 flex-col border-r border-border bg-surface">
      <div className="flex h-14 items-center gap-2 px-4">
        <img src="/assets/logo.png" alt="HiveBear" className="h-7 w-7 rounded-lg" />
        <span className="text-sm font-semibold tracking-tight">HiveBear</span>
      </div>
      <nav className="flex flex-1 flex-col gap-0.5 px-2 py-2">
        {links.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors ${
                isActive
                  ? "bg-surface-overlay text-text-primary"
                  : "text-text-secondary hover:bg-surface-overlay/50 hover:text-text-primary"
              }`
            }
          >
            <Icon size={16} strokeWidth={1.8} />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="border-t border-border px-4 py-3 text-xs text-text-muted">
        v0.1.0
      </div>
    </aside>
  );
}
