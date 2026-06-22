# Cairn infra — SkyPilot on a SHARED AWS account

**This machine runs several projects' SkyPilot fleets on one AWS account and one SkyPilot install**
(Cairn, PitchLab, …). They share the SkyPilot *control plane* — the local API server
(`python -m sky.server.server` under `~/.sky/`), the account-global **jobs controller**
(`sky-jobs-controller-*`), the SSH mux sockets, and the cluster catalog. **Cairn must stay in its own
lane.** The project boundary is **AWS tags**, not SkyPilot state.

## The rule: isolate by TAG, never touch SkyPilot's shared/root config

Every Cairn box carries `cairn=true` (set via `labels:` in each `*.sky.yaml`). That tag — not the
shared SkyPilot daemon — is how we bring boxes up, tear them down, and report cost. Do **all three**
by tag so we never disturb another project:

| Op | Do this (tag-scoped) | NOT this (touches shared state) |
|----|----------------------|----------------------------------|
| **UP** | `sky launch -c cairn-<x> <yaml> … --async` with `labels: {cairn: "true"}` in the yaml | — |
| **DOWN** | `infra/skypilot/nuke.sh --force` → terminates **by `cairn=true`** via the EC2 API (ground truth) | `sky down` of the **jobs controller**; `sky api stop` |
| **REPORT** | `infra/skypilot/cost-report.sh` / Cost Explorer filtered on tag `cairn=true` | reading another project's clusters out of `sky status` |

EC2 (filtered by `cairn=true`) is the **ground truth** for "what is Cairn billing" — `sky status` can
lie, and it shows *every* project's clusters. `nuke.sh` (dry-run by default) is the canonical check.

## Hands OFF the shared control plane

These are global — touching them breaks **every** project on this machine:

- **The jobs controller** (`sky-jobs-controller-*`, often in a different region e.g. `ap-northeast-2`).
  It is **not** `cairn`-tagged on purpose, so `nuke.sh` skips it. Cairn never uses managed jobs
  (`sky jobs launch`) anyway — we use plain `sky launch` + our own in-code recovery
  (`cairn_node/serve.py`, `spot_watch`). **Never `sky down` it.**
- **The shared API server / `sky api stop|start`** and anything under `~/.sky/` (config, catalogs,
  mux sockets). Don't reconfigure or restart it for Cairn's sake.

## Credentials: one named profile per project, no default

`~/.aws` has **named** profiles only — `[cairn-skypilot]` (Cairn, acct `615809814090`) and `[pitch]`
(PitchLab) — and **no `[default]`**. So:

- Always run Cairn AWS/sky commands with `AWS_PROFILE=cairn-skypilot` (the scripts default to it).
- The shared API server holds **one** profile's creds at a time (whatever env started it). A
  bare-started server has **none** → `NoCredentialsError` on every provision.

## If the shared API server is already dead (the "rsync-255 / wedged-in-INIT" flake)

That flake = the shared server crashing mid-launch (seen 2026-06-23: it had booted from PitchLab's
venv, went unhealthy, and died, orphaning Cairn boxes in `INIT`). **Only if it has already died on its
own** and you must bring it back for a Cairn launch:

```sh
export AWS_PROFILE=cairn-skypilot
sky api stop && sky api start          # fresh server WITH Cairn creds in its env
sky check aws                          # must print "AWS: enabled [compute, storage]"
# then launch; if a half-dead cluster is wedged in INIT:
nuke.sh --force                        # terminate the orphans by tag (EC2 ground truth)
```

Leave it running healthy when done; PitchLab restarts it with `AWS_PROFILE=pitch` for its own work.
**Longer-term fix worth doing:** give Cairn its *own* isolated API server (separate
`SKYPILOT_API_SERVER_ENDPOINT` / home) so the projects can't destabilize each other.

> Cross-project AWS/SkyPilot bring-up notes also live in `~/Code/infra/` (the shared infra-docs home).
