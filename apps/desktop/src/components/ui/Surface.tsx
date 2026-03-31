import type { HTMLAttributes } from "react";

interface SurfaceProps extends HTMLAttributes<HTMLDivElement> {
  /** Max-width constraint for content. Default is full width. */
  maxWidth?: "sm" | "md" | "lg" | "full";
}

const maxWidthStyles = {
  sm: "max-w-2xl",
  md: "max-w-4xl",
  lg: "max-w-6xl",
  full: "",
};

export default function Surface({
  maxWidth = "full",
  className = "",
  children,
  ...props
}: SurfaceProps) {
  return (
    <div
      className={[
        "animate-[page-enter] p-4 sm:p-6",
        maxWidthStyles[maxWidth],
        className,
      ].join(" ")}
      {...props}
    >
      {children}
    </div>
  );
}
