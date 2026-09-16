import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build lands in docs/play/ so GitHub Pages serves it next to the
// project page; site/server.py serves the same directory locally.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../../docs/play",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8123",
    },
  },
});
