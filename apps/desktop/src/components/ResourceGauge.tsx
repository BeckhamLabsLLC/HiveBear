interface ResourceGaugeProps {
  label: string;
  used: number;
  total: number;
  unit: string;
  formatValue?: (n: number) => string;
}

function defaultFormat(n: number): string {
  if (n >= 1024 * 1024 * 1024) return `${(n / (1024 ** 3)).toFixed(1)} GB`;
  if (n >= 1024 * 1024) return `${(n / (1024 ** 2)).toFixed(0)} MB`;
  return `${n}`;
}

export default function ResourceGauge({
  label, used, total, unit, formatValue = defaultFormat,
}: ResourceGaugeProps) {
  const pct = total > 0 ? Math.min((used / total) * 100, 100) : 0;
  const color = pct > 90 ? "bg-danger" : pct > 70 ? "bg-warning" : "bg-success";

  return (
    <div className="rounded-xl border border-border bg-surface-raised p-4">
      <div className="mb-3 flex items-baseline justify-between">
        <span className="text-sm text-text-secondary">{label}</span>
        <span className="font-mono text-xs text-text-muted">
          {formatValue(used)} / {formatValue(total)} {unit}
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-surface-overlay">
        <div
          className={`h-full rounded-full transition-all ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
