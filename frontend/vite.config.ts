import { defineConfig } from "vite";

// Localhost only (global rule): the dev server binds 127.0.0.1 and proxies the Python sim server.
export default defineConfig({
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
      "/api": { target: "http://127.0.0.1:8765" },
    },
  },
  preview: { host: "127.0.0.1" },
});
