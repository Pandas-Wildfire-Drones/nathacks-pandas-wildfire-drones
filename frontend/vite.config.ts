import { defineConfig } from "vite";

// Localhost only (global rule): the dev server binds 127.0.0.1 and proxies the Python sim server.
// SIM_PORT (default 8765) lets a second checkout run its own sim server side by side.
const SIM_PORT = process.env.SIM_PORT ?? "8765";
export default defineConfig({
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/ws": { target: `ws://127.0.0.1:${SIM_PORT}`, ws: true },
      "/api": { target: `http://127.0.0.1:${SIM_PORT}` },
    },
  },
  preview: { host: "127.0.0.1" },
});
