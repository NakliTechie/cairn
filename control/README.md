# control/ — control plane (Cloudflare Workers + Durable Objects)

The one ingress (spec §7). **Never in the per-token hot path** (invariant #4).

- **Gateway (Workers):** OpenAI-compatible API — auth, request validation, admission,
  route setup, token streaming back to the client. A `/version` endpoint (handoff §14).
- **Fleet state (Durable Objects):** registry · topology · health · the durable stream
  token-history (IDs, not tensors — invariant #2) · the drain/retry/reassign loop.

Deployed with `wrangler deploy` (a wrangler project lands here). **Status: not started.**
The recovery-orchestration *logic* is prototyped simulation-first in
`scheduler/` (rung 1) and ports to the DO once proven.
