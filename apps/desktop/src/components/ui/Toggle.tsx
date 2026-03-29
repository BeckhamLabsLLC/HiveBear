interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  label?: string;
}

export default function Toggle({ checked, onChange, disabled = false, label }: ToggleProps) {
  return (
    <label className={`relative inline-flex items-center gap-2.5 ${disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        className="peer sr-only"
      />
      <div
        className={[
          "relative h-5 w-9 rounded-full",
          "after:absolute after:left-0.5 after:top-0.5 after:h-4 after:w-4 after:rounded-full",
          "after:transition-transform after:duration-[200ms] after:[transition-timing-function:var(--ease-out-expo)]",
          checked
            ? "bg-paw-500 after:translate-x-4 after:bg-white"
            : "bg-surface-overlay after:bg-text-muted",
        ].join(" ")}
      />
      {label && <span className="text-sm text-text-primary">{label}</span>}
    </label>
  );
}
