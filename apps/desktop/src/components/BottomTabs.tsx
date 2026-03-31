import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  MessageSquare,
  Search,
  Settings,
  UserCircle,
} from "lucide-react";

const tabs = [
  { to: "/", label: "Home", icon: LayoutDashboard },
  { to: "/models", label: "Models", icon: Search },
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/account", label: "Account", icon: UserCircle },
  { to: "/settings", label: "Settings", icon: Settings },
];

export default function BottomTabs() {
  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 flex border-t border-border bg-surface pb-[env(safe-area-inset-bottom)]">
      {tabs.map((tab) => (
        <NavLink
          key={tab.to}
          to={tab.to}
          className={({ isActive }) =>
            [
              "flex flex-1 flex-col items-center gap-0.5 py-2 text-[10px]",
              isActive
                ? "text-paw-500"
                : "text-text-muted active:text-text-secondary",
            ].join(" ")
          }
        >
          {({ isActive }) => (
            <>
              <tab.icon
                size={20}
                strokeWidth={isActive ? 2 : 1.5}
              />
              <span>{tab.label}</span>
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );
}
