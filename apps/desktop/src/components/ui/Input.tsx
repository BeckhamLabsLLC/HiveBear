import { forwardRef, type InputHTMLAttributes, type TextareaHTMLAttributes } from "react";

const baseStyles = [
  "w-full rounded-[var(--radius-md)] border border-border bg-surface",
  "text-sm text-text-primary placeholder:text-text-muted",
  "outline-none interactive-hover",
  "focus:ring-2 focus:ring-paw-500/30 focus:border-paw-500",
  "disabled:opacity-50 disabled:cursor-not-allowed",
].join(" ");

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  inputSize?: "sm" | "md";
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ inputSize = "md", className = "", ...props }, ref) => (
    <input
      ref={ref}
      className={[
        baseStyles,
        inputSize === "sm" ? "px-2.5 py-1.5 text-xs" : "px-3 py-2",
        className,
      ].join(" ")}
      {...props}
    />
  ),
);

Input.displayName = "Input";

interface TextAreaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  inputSize?: "sm" | "md";
}

const TextArea = forwardRef<HTMLTextAreaElement, TextAreaProps>(
  ({ inputSize = "md", className = "", ...props }, ref) => (
    <textarea
      ref={ref}
      className={[
        baseStyles,
        "resize-none",
        inputSize === "sm" ? "px-2.5 py-1.5 text-xs" : "px-3 py-2",
        className,
      ].join(" ")}
      {...props}
    />
  ),
);

TextArea.displayName = "TextArea";

export { Input as default, TextArea };
