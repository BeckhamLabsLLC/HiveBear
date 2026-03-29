import { useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { Copy, Check } from "lucide-react";

interface MarkdownContentProps {
  content: string;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-text-muted transition-colors hover:bg-surface-raised hover:text-text-primary"
      title="Copy code"
    >
      {copied ? (
        <>
          <Check size={12} />
          Copied!
        </>
      ) : (
        <>
          <Copy size={12} />
          Copy
        </>
      )}
    </button>
  );
}

export default function MarkdownContent({ content }: MarkdownContentProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={{
        pre({ children }) {
          // Extract code text for copy button
          let codeText = "";
          let language = "";

          if (
            children &&
            typeof children === "object" &&
            "props" in (children as React.ReactElement)
          ) {
            const child = children as React.ReactElement<{
              children?: string;
              className?: string;
            }>;
            codeText = typeof child.props.children === "string" ? child.props.children : "";
            const className = child.props.className || "";
            const match = className.match(/language-(\w+)/);
            if (match) language = match[1];
          }

          return (
            <div className="group relative my-3 overflow-hidden rounded-lg border border-border bg-surface-overlay">
              <div className="flex items-center justify-between border-b border-border px-3 py-1.5">
                <span className="text-[10px] font-medium uppercase tracking-wider text-text-muted">
                  {language || "code"}
                </span>
                <CopyButton text={codeText} />
              </div>
              <div className="overflow-x-auto p-3">
                {children}
              </div>
            </div>
          );
        },
        code({ className, children, ...props }) {
          const isBlock = className?.includes("language-") || className?.includes("hljs");
          if (isBlock) {
            return (
              <code className={`${className || ""} text-xs`} {...props}>
                {children}
              </code>
            );
          }
          return (
            <code className="rounded bg-surface-overlay px-1.5 py-0.5 text-xs text-paw-400" {...props}>
              {children}
            </code>
          );
        },
        a({ href, children }) {
          return (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-paw-400 underline decoration-paw-400/30 hover:decoration-paw-400"
            >
              {children}
            </a>
          );
        },
        table({ children }) {
          return (
            <div className="my-3 overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-xs">{children}</table>
            </div>
          );
        },
        th({ children }) {
          return (
            <th className="border-b border-border bg-surface-overlay px-3 py-2 text-left font-semibold text-text-secondary">
              {children}
            </th>
          );
        },
        td({ children }) {
          return (
            <td className="border-b border-border px-3 py-2 text-text-primary">
              {children}
            </td>
          );
        },
        ul({ children }) {
          return <ul className="my-2 ml-4 list-disc space-y-1 text-text-primary">{children}</ul>;
        },
        ol({ children }) {
          return <ol className="my-2 ml-4 list-decimal space-y-1 text-text-primary">{children}</ol>;
        },
        blockquote({ children }) {
          return (
            <blockquote className="my-3 border-l-2 border-paw-500/50 pl-3 text-text-secondary italic">
              {children}
            </blockquote>
          );
        },
        h1({ children }) {
          return <h1 className="mb-2 mt-4 text-lg font-bold text-text-primary">{children}</h1>;
        },
        h2({ children }) {
          return <h2 className="mb-2 mt-3 text-base font-bold text-text-primary">{children}</h2>;
        },
        h3({ children }) {
          return <h3 className="mb-1 mt-3 text-sm font-semibold text-text-primary">{children}</h3>;
        },
        p({ children }) {
          return <p className="my-1.5 leading-relaxed">{children}</p>;
        },
        hr() {
          return <hr className="my-4 border-border" />;
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
