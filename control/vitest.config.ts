import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// The control-plane logic (gateway validation/auth/shaping, fleet state, request
// routing) is pure and runs in a plain node environment. The `cloudflare:workers`
// module (DurableObject base) only exists in workerd, so it is aliased to a stub for
// node tests. Full workerd integration tests (the live Durable Object + storage) are
// added at deploy time with @cloudflare/vitest-pool-workers — owner-gated.
export default defineConfig({
  resolve: {
    alias: {
      "cloudflare:workers": fileURLToPath(new URL("./test/stubs/cloudflare-workers.ts", import.meta.url)),
    },
  },
  test: {
    include: ["test/**/*.test.ts"],
    environment: "node",
  },
});
