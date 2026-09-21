import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { initTelemetry } from "./lib/telemetry";
import "./styles/index.css";
import "highlight.js/styles/github-dark.css";

// Not awaited: this costs one IPC round trip to ask Rust whether the user has
// opted out, and blocking first paint on that would be a worse trade than
// missing errors thrown in the intervening few milliseconds.
void initTelemetry();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
