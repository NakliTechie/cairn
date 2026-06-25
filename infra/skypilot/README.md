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
| **UP** | `bash infra/skypilot/launch.sh` — the **sanctioned** path: it pins the shared api server to cairn's identity (gotcha #9), then `sky launch … -d` (detached, gotcha #10). `CLUSTER=… YAML=…` env-select the box (default `cairn-fleet`). | a bare `sky launch` — skips the identity guard, so a shared box can provision under a sibling's key |
| **DOWN** | `bash infra/skypilot/teardown.sh` → terminates **by `cairn=true`** across all 4 regions with cairn's own key, then purges the sky record. `infra/skypilot/nuke.sh --force` stays the break-glass EC2-only kill (read-only by default). | `sky down` of the **jobs controller** |
| **REPORT** | `infra/skypilot/cost-report.sh` / Cost Explorer filtered on tag `cairn=true` | reading another project's clusters out of `sky status` |

> **Why the wrapper, not a bare `sky launch`** (`~/Code/infra/aws/README.md` gotcha #9): SkyPilot runs
> ONE shared local api server that provisions with the identity it was *started* with — not your
> `AWS_PROFILE`. On this shared box a naive launch can silently use a sibling project's key, breaking
> isolation + the tag-gated kill switch. `launch.sh` restarts the server under cairn first. **Caveat:**
> that `sky api stop`s the shared server, so only **one** project's fleet can be driven at a time here.

EC2 (filtered by `cairn=true`) is the **ground truth** for "what is Cairn billing" — `sky status` can
lie, and it shows *every* project's clusters. `nuke.sh` (dry-run by default) is the canonical check.

## Hands OFF the shared control plane

These are global — touching them breaks **every** project on this machine:

- **The jobs controller** (`sky-jobs-controller-*`, often in a different region e.g. `ap-northeast-2`).
  It is **not** `cairn`-tagged on purpose, so `nuke.sh`/`teardown.sh` skip it. Cairn never uses managed
  jobs (`sky jobs launch`) anyway — we use plain `sky launch` + our own in-code recovery
  (`cairn_node/serve.py`, `spot_watch`). **Never `sky down` it.**
- **The shared API server.** UPDATED (gotcha #9, 2026-06-25): we now *deliberately* restart it under
  cairn's identity at launch — that's what `launch.sh` does (`sky api stop` → `sky check aws` with
  `AWS_PROFILE=cairn-skypilot`). This supersedes the old "never restart it" rule, which let a launch
  silently provision under whatever sibling profile last started the server. **Because the restart stops
  the shared server, run only one project's fleet at a time on this machine.** Still hands-off:
  arbitrary edits to `~/.sky/` config/catalogs for cairn's sake.

## Credentials: one named profile per project, no default

`~/.aws` has **named** profiles only — `[cairn-skypilot]` (Cairn, acct `AWS_ACCOUNT_ID`) and `[pitch]`
(PitchLab) — and **no `[default]`**. So:

- Always run Cairn AWS/sky commands with `AWS_PROFILE=cairn-skypilot` (the scripts default to it).
- The shared API server holds **one** profile's creds at a time (whatever env started it). A
  bare-started server has **none** → `NoCredentialsError` on every provision.

## If the shared API server is already dead (the "rsync-255 / wedged-in-INIT" flake)

`launch.sh` already restarts the server under cairn on every launch (gotcha #9), so you rarely need
this by hand. That flake = the shared server crashing mid-launch (seen 2026-06-23: it had booted from
PitchLab's venv, went unhealthy, and died, orphaning Cairn boxes in `INIT`). **Only if it has already
died on its own** and you must bring it back for a Cairn launch:

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
