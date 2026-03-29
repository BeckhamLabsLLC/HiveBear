/**
 * Three-dot typing indicator shown before the first token arrives.
 * After content starts streaming, the parent should stop rendering this
 * and show the streaming text with a cursor instead.
 */
export default function StreamingIndicator() {
  return (
    <span className="inline-flex items-center gap-1 py-1">
      <span className="h-1.5 w-1.5 rounded-full bg-text-muted animate-[streaming-dot]" />
      <span className="h-1.5 w-1.5 rounded-full bg-text-muted animate-[streaming-dot] [animation-delay:0.2s]" />
      <span className="h-1.5 w-1.5 rounded-full bg-text-muted animate-[streaming-dot] [animation-delay:0.4s]" />
    </span>
  );
}

/**
 * Thin blinking pipe cursor shown at the end of streaming text.
 */
export function StreamingCursor() {
  return (
    <span className="ml-0.5 inline-block h-4 w-[2px] rounded-full bg-paw-500 align-middle animate-[streaming-cursor]" />
  );
}
