// Node-only stub for the `cloudflare:workers` module (which exists only in workerd).
// Lets the gateway/routing tests import the Worker without the real runtime.
export class DurableObject {
  ctx: unknown;
  env: unknown;
  constructor(ctx: unknown, env: unknown) {
    this.ctx = ctx;
    this.env = env;
  }
}
