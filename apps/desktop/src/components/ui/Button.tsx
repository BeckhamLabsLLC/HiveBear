import { forwardRef, type ButtonHTMLAttributes } from "react";

type ButtonVariant = "primary" | "secondary" | "ghost";
type ButtonSize = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

const variantStyles: Record<ButtonVariant, string> = {
  primary:
    "bg-paw-500 text-white hover:bg-paw-600 disabled:opacity-40",
  secondary:
    "border border-border text-text-secondary hover:border-paw-500/50 hover:text-text-primary disabled:opacity-40",
  ghost:
    "text-text-secondary hover:bg-surface-overlay hover:text-text-primary disabled:opacity-40",
};

const sizeStyles: Record<ButtonSize, string> = {
  sm: "px-2.5 py-1 text-xs gap-1.5",
  md: "px-4 py-2 text-sm gap-2",
  lg: "px-6 py-2.5 text-sm gap-2",
};

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "primary", size = "md", className = "", children, ...props }, ref) => (
    <button
      ref={ref}
      className={[
        "inline-flex items-center justify-center font-medium",
        "rounded-[var(--radius-md)]",
        "interactive-hover press-scale",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-paw-500/40 focus-visible:ring-offset-1 focus-visible:ring-offset-surface",
        variantStyles[variant],
        sizeStyles[size],
        className,
      ].join(" ")}
      {...props}
    >
      {children}
    </button>
  ),
);

Button.displayName = "Button";
export default Button;
