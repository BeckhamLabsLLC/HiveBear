import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { sentryVitePlugin } from "@sentry/vite-plugin";
import pkg from "./package.json" with { type: "json" };

const host = process.env.TAURI_DEV_HOST;

// Uploading source maps needs a build-time token, which only CI has. Without it
// the plugin is skipped entirely rather than failing the build, so a local
// `npm run build` still works.
const sentryAuthToken = process.env.SENTRY_AUTH_TOKEN;

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    ...(sentryAuthToken
      ? [
          sentryVitePlugin({
            org: "beckham-labs-llc",
            project: "hivebear-client",
            authToken: sentryAuthToken,
            // Must match the `release` passed to Sentry.init in main.tsx, or the
            // maps are filed under a release the events never reference and
            // every frame stays minified.
            release: { name: `hivebear@${pkg.version}` },
            sourcemaps: {
              // Upload them, then delete them: this is a shipped desktop app,
              // and the maps would otherwise sit inside the bundle.
              filesToDeleteAfterUpload: ["**/dist/**/*.map"],
            },
          }),
        ]
      : []),
  ],
  clearScreen: false,
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
  },
  build: {
    // Previously unset, so Vite defaulted to false and the shipped bundle had no
    // maps at all — every frontend stack trace would have been minified noise.
    // "hidden" emits them for upload without leaving a //# sourceMappingURL
    // comment pointing at a file we delete.
    sourcemap: "hidden",
  },
  server: {
    port: 5173,
    strictPort: true,
    host: host || false,
    hmr: host ? { protocol: "ws", host, port: 5174 } : undefined,
    watch: { ignored: ["**/src-tauri/**"] },
  },
});
