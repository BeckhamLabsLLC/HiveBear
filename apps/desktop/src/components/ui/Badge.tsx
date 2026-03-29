import type { HTMLAttributes } from "react";

type BadgeVariant = "default" | "success" | "warning" | "danger" | "accent";

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

const variantStyles: Record<BadgeVariant, string> = {
  default: "bg-surface-overlay text-text-secondary border-border",
  success: "bg-success/10 text-success border-success/20",
  warning: "bg-warning/10 text-warning border-warning/20",
  danger: "bg-danger/10 text-danger border-danger/20",
  accent: "bg-paw-500/10 text-paw-400 border-paw-500/20",
};

export default function Badge({
  variant = "default",
  className = "",
  children,
  ...props
}: BadgeProps) {
  return (
    <span
      className={[
        "inline-flex items-center gap-1 rounded-[var(--radius-sm)] border",
        "px-2 py-0.5 text-xs font-medium",
        variantStyles[variant],
        className,
      ].join(" ")}
      {...props}
    >
      {children}
    </span>
  );
}
