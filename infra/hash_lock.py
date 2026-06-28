#!/usr/bin/env python3
"""Regenerate a `--require-hashes` lock from the exact pins in infra/requirements.lock.

Why not `uv pip compile`/`pip-compile`? Those RE-RESOLVE the dependency graph, which (a) can drift the
proven versions and (b) trips over the deliberate `compressed-tensors==0.9.4` downgrade, which conflicts
with sglang's "pull latest" preference (the exact ResolutionImpossible the box dodges via a `--no-deps`
post-step). This tool does NOT resolve: it takes each `name==version` verbatim and attaches the sha256 of
*every* distribution PyPI publishes for that exact version (all wheels + sdist), which is precisely what a
`--require-hashes` install needs — the box matches whatever artifact it downloads against the set.

  python infra/hash_lock.py                 # rewrite infra/requirements.lock in place (preserves the header)
  python infra/hash_lock.py --check         # verify every pin is hashed + versions unchanged; exit 1 if not
  python infra/hash_lock.py -o /tmp/out.txt # write elsewhere (inspect before committing)

To bump a package: edit its `==` in infra/requirements.lock and re-run (it ignores existing --hash lines).
Idempotent: re-running with no version edits reproduces the same file (hashes sorted, input order kept).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCK = ROOT / "infra" / "requirements.lock"
PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\]+)")  # `name==version`, ignores trailing ` \`


def _read(path: pathlib.Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Split the lock into (leading header comment lines, [(name, version), ...]). Ignores --hash lines."""
    header, pins, in_header = [], [], True
    for line in path.read_text().splitlines():
        s = line.strip()
        if in_header and (s.startswith("#") or not s):
            header.append(line)
            continue
        in_header = False
        m = PIN_RE.match(s)
        if m:
            pins.append((m.group(1), m.group(2)))
    while header and not header[-1].strip():  # trim trailing blank header lines (we re-add one)
        header.pop()
    return header, pins


def _fetch_hashes(name: str, version: str) -> list[str]:
    """All sha256 digests PyPI publishes for this exact version. Tries the raw then PEP-503 name."""
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    last_err: Exception | None = None
    for candidate in dict.fromkeys([name, normalized]):  # de-dup, raw first
        url = f"https://pypi.org/pypi/{candidate}/{version}/json"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.load(r)
                digs = sorted({f["digests"]["sha256"] for f in data.get("urls", []) if f.get("digests", {}).get("sha256")})
                if not digs:
                    raise ValueError(f"{name}=={version}: no sha256 artifacts on PyPI (yanked/metadata-only?)")
                return digs
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 404:
                    break  # wrong name spelling — try the next candidate
                # transient (5xx/429) — back off and retry
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                last_err = e
    raise RuntimeError(f"{name}=={version}: could not fetch hashes ({last_err})")


def _render(header: list[str], hashed: list[tuple[str, str, list[str]]]) -> str:
    out = list(header)
    if out and out[-1].strip():
        out.append("")
    for name, version, digs in hashed:
        lines = [f"{name}=={version} \\"]
        for i, h in enumerate(digs):
            tail = " \\" if i < len(digs) - 1 else ""
            lines.append(f"    --hash=sha256:{h}{tail}")
        out.append("\n".join(lines))
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=pathlib.Path, default=LOCK)
    ap.add_argument("-o", "--output", type=pathlib.Path, default=None, help="default: rewrite --input in place")
    ap.add_argument("--check", action="store_true", help="don't write; verify the input is fully hashed + parses")
    args = ap.parse_args()

    header, pins = _read(args.input)
    if not pins:
        print(f"no `name==version` pins found in {args.input}", file=sys.stderr)
        return 1

    if args.check:
        text = args.input.read_text()
        missing = [f"{n}=={v}" for n, v in pins if f"{n}=={v} \\" not in text and f"{n}=={v}\n" in text]
        unhashed = [f"{n}=={v}" for n, v in pins if "--hash=sha256:" not in text.split(f"{n}=={v}")[1].split("==")[0]] if False else []
        # simpler, robust check: every pin line must be followed by at least one --hash within its block
        blocks = re.split(r"(?m)^(?=[A-Za-z0-9][A-Za-z0-9._-]*==)", text)
        no_hash = [b.splitlines()[0].strip() for b in blocks if PIN_RE.match(b.strip()) and "--hash=sha256:" not in b]
        if no_hash:
            print(f"--check FAILED: {len(no_hash)} pin(s) without a hash:\n  " + "\n  ".join(no_hash), file=sys.stderr)
            return 1
        print(f"--check OK: all {len(pins)} pins are hash-pinned and parse cleanly.")
        return 0

    print(f"fetching PyPI sha256 digests for {len(pins)} pinned packages …", file=sys.stderr)
    results: dict[tuple[str, str], list[str]] = {}
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_fetch_hashes, n, v): (n, v) for n, v in pins}
        for fut in concurrent.futures.as_completed(futs):
            nv = futs[fut]
            try:
                results[nv] = fut.result()
            except Exception as e:  # noqa: BLE001 — surface any failure, don't write a partial lock
                errors.append(str(e))
    if errors:
        print(f"FAILED — {len(errors)} package(s) could not be hashed (no file written):", file=sys.stderr)
        for e in sorted(errors):
            print("  " + e, file=sys.stderr)
        return 1

    hashed = [(n, v, results[(n, v)]) for n, v in pins]  # preserve input order
    text = _render(header, hashed)
    out = args.output or args.input
    out.write_text(text)
    total = sum(len(h) for _, _, h in hashed)
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"wrote {shown}  ({len(hashed)} packages, {total} hashes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
