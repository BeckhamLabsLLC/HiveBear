import type { ReactNode } from "react";
import Button from "./Button";

interface EmptyStateProps {
  icon: ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  children?: ReactNode;
}

export default function EmptyState({ icon, title, description, action, children }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-16 animate-[fade-in]">
      <div className="flex h-14 w-14 items-center justify-center rounded-[var(--radius-xl)] bg-surface-raised text-text-muted">
        {icon}
      </div>
      <div className="text-center">
        <h3 className="text-sm font-medium text-text-primary">{title}</h3>
        {description && (
          <p className="mt-1 max-w-sm text-xs text-text-muted">{description}</p>
        )}
      </div>
      {action && (
        <Button variant="primary" size="sm" onClick={action.onClick}>
          {action.label}
        </Button>
      )}
      {children}
    </div>
  );
}
