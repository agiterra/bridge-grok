#!/usr/bin/env bun
import { startServer } from "./src/mcp-server.js";

startServer().catch((e) => {
  console.error("[bridge] fatal:", e);
  process.exit(1);
});
