import { forwardRef, type HTMLAttributes } from "react";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  interactive?: boolean;
  padding?: "sm" | "md" | "lg";
}

const paddingStyles = {
  sm: "p-3",
  md: "p-4",
  lg: "p-6",
};

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ interactive = false, padding = "md", className = "", children, ...props }, ref) => (
    <div
      ref={ref}
      className={[
        "rounded-[var(--radius-lg)] border border-border bg-surface-raised",
        "shadow-[var(--shadow-raised)]",
        paddingStyles[padding],
        interactive &&
          "interactive-hover cursor-pointer hover:border-paw-500/30 hover:shadow-[var(--shadow-overlay)]",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
      {...props}
    >
      {children}
    </div>
  ),
);

Card.displayName = "Card";
export default Card;
