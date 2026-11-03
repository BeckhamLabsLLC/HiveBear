import { describe, expect, it, vi } from "vitest";

// invoke.ts pulls in the Tauri bridge, Sentry and the toast component; none of
// them matter for parsing errors.
vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@sentry/react", () => ({ addBreadcrumb: vi.fn(), captureException: vi.fn() }));
vi.mock("../components/Toast", () => ({ notify: vi.fn() }));

import { errorMessage } from "./invoke";

describe("errorMessage", () => {
  it("unpacks the backend's JSON CommandError", () => {
    const raw = JSON.stringify({ code: "download_failed", message: "Disk full" });
    expect(errorMessage(raw)).toBe("Disk full");
  });

  it("keeps plain strings as they are", () => {
    expect(errorMessage("Config lock poisoned")).toBe("Config lock poisoned");
  });

  it("does not choke on strings that merely start with a brace", () => {
    expect(errorMessage("{not json")).toBe("{not json");
  });

  it("uses Error messages", () => {
    expect(errorMessage(new Error("boom"))).toBe("boom");
  });
});
