# Cairn infra — SkyPilot tag-isolation discipline

If this AWS account / SkyPilot install is shared with other workloads, Cairn must stay in
its own lane. SkyPilot keeps account-global shared state — the local API server
(`python -m sky.server.server` under `~/.sky/`), the jobs controller
(`sky-jobs-controller-*`), the SSH mux sockets, and the cluster catalog. **The project
boundary is AWS tags, not SkyPilot state.**

## The rule: isolate by TAG, never touch SkyPilot's shared/root config

Every Cairn box carries `cairn=true` (set via `labels:` in each `*.sky.yaml`). That tag —
not the shared SkyPilot daemon — is how we bring boxes up, tear them down, and report cost.
Do all three by tag so a shared install is never disturbed:

| Op | Do this (tag-scoped) | NOT this (touches shared state) |
|----|----------------------|----------------------------------|
| **UP** | `bash infra/skypilot/launch.sh` — pins the shared api server to Cairn's identity, then `sky launch … -d` (detached). `CLUSTER=… YAML=…` env-select the box (default `cairn-fleet`). | a bare `sky launch` — skips the identity guard, so a shared box can provision under another identity's key |
| **DOWN** | `bash infra/skypilot/teardown.sh` → terminates **by `cairn=true`** across all configured regions with Cairn's own key, then purges the sky record. `infra/skypilot/nuke.sh --force` is the break-glass EC2-only kill (read-only by default). | `sky down` of the **jobs controller** |
| **REPORT** | `infra/skypilot/cost-report.sh` / Cost Explorer filtered on tag `cairn=true` | reading another workload's clusters out of `sky status` |

> **Why the wrapper, not a bare `sky launch`:** SkyPilot runs ONE shared local api server
> that provisions with the identity it was *started* with — not your `AWS_PROFILE`. On a
> shared box a naive launch can silently use another key, breaking isolation + the
> tag-gated kill switch. `launch.sh` restarts the server under Cairn first. **Caveat:**
> that `sky api stop`s the shared server, so only **one** workload's fleet can be driven at
> a time.

EC2 (filtered by `cairn=true`) is the **ground truth** for "what is Cairn billing" —
`sky status` can lie, and it shows *every* workload's clusters. `nuke.sh` (dry-run by
default) is the canonical check.

## Hands OFF the shared control plane

These are global — touching them affects every workload on a shared install:

- **The jobs controller** (`sky-jobs-controller-*`, often in a different region). It is
  **not** `cairn`-tagged on purpose, so `nuke.sh`/`teardown.sh` skip it. Cairn never uses
  managed jobs (`sky jobs launch`) for the live fleet anyway — it uses plain `sky launch` +
  its own in-code recovery (`cairn_node/serve.py`, `spot_watch`). **Never `sky down` it.**
- **The shared API server.** `launch.sh` deliberately restarts it under Cairn's identity at
  launch (`sky api stop` → `sky check aws` with the Cairn profile) — this prevents a launch
  silently provisioning under whatever profile last started the server. **Because the
  restart stops the shared server, run only one workload's fleet at a time.** Avoid
  arbitrary edits to `~/.sky/` config/catalogs.

## Credentials: one named profile, no default

Use a **named** profile (e.g. `[cairn-skypilot]`) and avoid a `[default]` profile, so:

- Always run Cairn AWS/sky commands with `AWS_PROFILE=cairn-skypilot` (the scripts default
  to it).
- The shared API server holds **one** profile's creds at a time (whatever env started it).
  A bare-started server has **none** → `NoCredentialsError` on every provision.

## If the shared API server is already dead (the "rsync-255 / wedged-in-INIT" flake)

`launch.sh` already restarts the server under Cairn on every launch, so you rarely need
this by hand. That flake is the shared server crashing mid-launch (it boots from one
workload's env, goes unhealthy, dies, and orphans Cairn boxes in `INIT`). **Only if it has
already died on its own** and you must bring it back for a Cairn launch:

```sh
export AWS_PROFILE=cairn-skypilot
sky api stop && sky api start          # fresh server WITH Cairn creds in its env
sky check aws                          # must print "AWS: enabled [compute, storage]"
# then launch; if a half-dead cluster is wedged in INIT:
nuke.sh --force                        # terminate the orphans by tag (EC2 ground truth)
```

Leave it running healthy when done.
**Longer-term fix worth doing:** give Cairn its *own* isolated API server (separate
`SKYPILOT_API_SERVER_ENDPOINT` / home) so workloads can't destabilize each other.
