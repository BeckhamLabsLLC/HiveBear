import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  handleReload = () => {
    this.setState({ error: null });
  };

  handleHardReload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.error) return this.props.children;

    const message = this.state.error.message || "An unexpected error occurred.";

    return (
      <div className="flex h-screen w-screen items-center justify-center bg-surface p-6 text-text-primary">
        <div className="max-w-md rounded-[var(--radius-xl)] border border-border bg-surface-raised p-6 shadow-[var(--shadow-overlay)]">
          <div className="mb-4 flex items-center gap-3">
            <AlertTriangle size={24} className="text-danger" aria-hidden />
            <h1 className="text-base font-semibold">Something broke</h1>
          </div>
          <p className="mb-4 text-sm text-text-secondary">
            HiveBear ran into an unexpected error. Your data and device identity are safe — this is a
            rendering issue.
          </p>
          <pre className="mb-4 max-h-40 overflow-auto whitespace-pre-wrap rounded-[var(--radius-md)] bg-surface px-3 py-2 font-mono text-xs text-text-muted">
            {message}
          </pre>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={this.handleReload}
              className="interactive-hover inline-flex items-center gap-2 rounded-[var(--radius-md)] border border-border bg-surface-overlay px-3 py-2 text-sm hover:bg-surface-raised"
            >
              <RefreshCw size={14} aria-hidden />
              Try again
            </button>
            <button
              onClick={this.handleHardReload}
              className="interactive-hover inline-flex items-center gap-2 rounded-[var(--radius-md)] border border-border bg-surface-overlay px-3 py-2 text-sm hover:bg-surface-raised"
            >
              Reload app
            </button>
          </div>
        </div>
      </div>
    );
  }
}
