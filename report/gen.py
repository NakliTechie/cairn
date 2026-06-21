#!/usr/bin/env python3
"""Cairn live report — a single-file, 2-tab HTML dashboard, human-readable, regenerated each turn.

  - Progress: a per-turn journal (what we did / what we learnt). Appended with `gen.py log`.
  - Infra & Cost: a visual of the launch ledger (infra/skypilot/launch-log.jsonl) — which boxes ran,
    which instance type, how long, ~how much. Running boxes tick live (client-side JS, ~30s).

    python report/gen.py log --title "Fixed the OOM" --did "forced mem-fraction" --learnt "env didn't inherit"
    python report/gen.py build        # just rebuild (refresh the cost tab from the ledger)

Output: report/cairn-report.html (open it in a browser). Estimated cost = duration × the spot quote at
launch; the bill of record is AWS Cost Explorer by the `cairn` tag (infra/skypilot/cost-report.sh).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RDIR = ROOT / "report"
JOURNAL = RDIR / "journal.jsonl"
LEDGER = ROOT / "infra" / "skypilot" / "launch-log.jsonl"
OUT = RDIR / "cairn-report.html"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _read(p: pathlib.Path) -> list:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def build() -> None:
    RDIR.mkdir(exist_ok=True)
    html = (TEMPLATE
            .replace("__JOURNAL__", json.dumps(_read(JOURNAL)))
            .replace("__EVENTS__", json.dumps(_read(LEDGER)))
            .replace("__GENERATED__", _now()))
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(_read(JOURNAL))} entries, {len(_read(LEDGER))} ledger events)")


def cmd_log(a: argparse.Namespace) -> None:
    RDIR.mkdir(exist_ok=True)
    rec = {"ts": _now(), "title": a.title, "did": a.did, "learnt": a.learnt, "status": a.status}
    with JOURNAL.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    build()


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cairn — live report</title>
<style>
 :root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--green:#3fb950;--blue:#58a6ff}
 *{box-sizing:border-box} body{margin:0;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text)}
 header{padding:16px 24px;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
 header h1{margin:0;font-size:17px;letter-spacing:.3px} .sub{color:var(--muted);font-size:12px}
 .tabs{display:flex;gap:4px;padding:0 24px;border-bottom:1px solid var(--border)}
 .tab{padding:10px 16px;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;user-select:none}
 .tab.active{color:var(--text);border-bottom-color:var(--blue)}
 main{padding:22px 24px;max-width:1080px}
 .hide{display:none}
 .entry{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:13px 16px;margin-bottom:11px}
 .entry .top{display:flex;justify-content:space-between;gap:12px;align-items:baseline}
 .entry .ttl{font-weight:600} .entry .ts{color:var(--muted);font-size:12px;white-space:nowrap}
 .entry .row{margin-top:5px} .lbl{color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;margin-right:7px}
 .badge{font-size:11px;padding:1px 8px;border-radius:10px;border:1px solid var(--border);color:var(--muted)}
 .summary{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:18px}
 .card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:13px 18px;min-width:130px}
 .card .n{font-size:23px;font-weight:600} .card .l{color:var(--muted);font-size:12px}
 table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden}
 th,td{text-align:left;padding:9px 14px;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums}
 th{color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.6px} tr:last-child td{border-bottom:none}
 .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;background:var(--muted)}
 .run .dot{background:var(--green);animation:pulse 1.4s infinite} @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
 .foot{color:var(--muted);font-size:12px;margin-top:14px} code{background:#21262d;padding:1px 5px;border-radius:4px}
</style></head><body>
<header><h1>⛰ Cairn — live report</h1><span class="sub" id="gen"></span><span class="sub" id="refreshed"></span></header>
<div class="tabs"><div class="tab active" data-t="progress">Progress</div><div class="tab" data-t="cost">Infra &amp; Cost</div></div>
<main><div id="progress"></div><div id="cost" class="hide"></div></main>
<script>
const JOURNAL=__JOURNAL__, EVENTS=__EVENTS__, GENERATED="__GENERATED__";
const esc=s=>(s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const fmtDur=h=>h<1?(h*60).toFixed(0)+"m":h.toFixed(2)+"h";
const tidy=t=>esc(t).replace("T"," ").replace("+00:00","").slice(0,16);
function renderProgress(){
 const el=document.getElementById("progress");
 el.innerHTML=JOURNAL.length?JOURNAL.slice().reverse().map(e=>`<div class="entry">
   <div class="top"><span class="ttl">${esc(e.title)}</span><span class="ts">${tidy(e.ts)} UTC</span></div>
   ${e.did?`<div class="row"><span class="lbl">did</span>${esc(e.did)}</div>`:""}
   ${e.learnt?`<div class="row"><span class="lbl">learnt</span>${esc(e.learnt)}</div>`:""}
   ${e.status?`<div class="row"><span class="badge">${esc(e.status)}</span></div>`:""}</div>`).join("")
  :'<p style="color:#8b949e">No entries yet.</p>';
}
function sessions(){
 const open={},done=[];
 for(const e of EVENTS){const c=e.cluster||"";
  if(e.event==="up"){(open[c]=open[c]||[]).push(e)}
  else if(e.event==="down"&&open[c]&&open[c].length){done.push([open[c].shift(),e,false])}}
 const run=[];for(const c in open)for(const u of open[c])run.push([u,null,true]);
 return done.concat(run).sort((a,b)=>new Date(a[0].ts)-new Date(b[0].ts));
}
function renderCost(){
 const el=document.getElementById("cost"),now=Date.now();let total=0,nrun=0,rows="";
 for(const [u,d,run] of sessions()){
  const end=run?now:new Date(d.ts).getTime(),hrs=(end-new Date(u.ts).getTime())/3.6e6,cost=hrs*(u.price_hr||0);
  total+=cost; if(run)nrun++;
  rows+=`<tr class="${run?'run':'done'}"><td><span class="dot"></span>${esc(u.cluster)}</td>
   <td>${esc(u.instance||"")}${u.spot?' <span class="badge">spot</span>':""}</td><td>${esc(u.region||"")}</td>
   <td>${tidy(u.ts)}</td><td>${fmtDur(hrs)}</td><td>$${cost.toFixed(2)}</td><td>${run?"RUNNING":"done"}</td></tr>`;
 }
 el.innerHTML=`<div class="summary">
   <div class="card"><div class="n">${nrun}</div><div class="l">boxes up now</div></div>
   <div class="card"><div class="n">${sessions().length}</div><div class="l">sessions</div></div>
   <div class="card"><div class="n">$${total.toFixed(2)}</div><div class="l">est. dev spend</div></div></div>
  <table><thead><tr><th>cluster</th><th>instance</th><th>region</th><th>started (UTC)</th><th>duration</th><th>~cost</th><th>status</th></tr></thead>
  <tbody>${rows||'<tr><td colspan="7" style="color:#8b949e">No launches logged yet.</td></tr>'}</tbody></table>
  <p class="foot">Estimated = duration × spot quote at launch; running rows tick live. Bill of record: AWS Cost
  Explorer by tag <code>cairn=true</code> (<code>infra/skypilot/cost-report.sh</code>).</p>`;
}
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{
 document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("active",x===t));
 document.getElementById("progress").classList.toggle("hide",t.dataset.t!=="progress");
 document.getElementById("cost").classList.toggle("hide",t.dataset.t!=="cost");});
document.getElementById("gen").textContent="generated "+tidy(GENERATED)+" UTC";
renderProgress();renderCost();
setInterval(()=>{renderCost();document.getElementById("refreshed").textContent="· live "+new Date().toISOString().slice(11,19)+"Z";},30000);
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Cairn live HTML report")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log", help="append a progress entry + rebuild")
    lg.add_argument("--title", required=True)
    lg.add_argument("--did", default="")
    lg.add_argument("--learnt", default="")
    lg.add_argument("--status", default="")
    lg.set_defaults(func=cmd_log)
    sub.add_parser("build", help="rebuild from the ledger + journal").set_defaults(func=lambda a: build())
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
