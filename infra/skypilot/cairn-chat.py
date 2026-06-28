#!/usr/bin/env python3
"""Cairn local chat — a tiny LOCAL proxy + chat UI for the live BYOK endpoint.

Use when serve_http is already running WITHOUT the built-in chat page (otherwise just open
http://localhost:8000/ after tunneling). This serves a chat page on http://localhost:8001 and forwards
/v1/* to the SSH-tunneled endpoint at 127.0.0.1:8000 (key added server-side → no CORS, no key in the page).

    # 1. tunnel the live endpoint to your laptop:
    ssh -L 8000:localhost:8000 cairn-dsv4 -N &
    # 2. run this + open the page:
    python3 infra/skypilot/cairn-chat.py          # → http://localhost:8001
    open http://localhost:8001

Env: CAIRN_UPSTREAM (default http://127.0.0.1:8000), CAIRN_API_KEY (sk-cairn-demo),
     CAIRN_MODEL (sgl-project/DeepSeek-V4-Flash-FP8), CAIRN_CHAT_PORT (8001).
"""
import json, os, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("CAIRN_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
KEY = os.environ.get("CAIRN_API_KEY", "sk-cairn-demo")
MODEL = os.environ.get("CAIRN_MODEL", "sgl-project/DeepSeek-V4-Flash-FP8")
PORT = int(os.environ.get("CAIRN_CHAT_PORT", "8001"))

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Cairn — DeepSeek-V4-Flash</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root{--teal:#0e5347;--ink:#1d2b27}
 *{box-sizing:border-box} body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;
   background:#f5f7f6;color:var(--ink);display:flex;flex-direction:column;height:100vh}
 header{padding:14px 18px;background:#fff;border-bottom:1px solid #e4e8e6}
 header h1{margin:0;font-size:17px;color:var(--teal)} header div{font-size:12px;color:#8a948f;margin-top:2px}
 #log{flex:1;overflow-y:auto;padding:18px;max-width:820px;width:100%;margin:0 auto}
 .msg{margin:10px 0;display:flex} .msg.u{justify-content:flex-end}
 .bub{max-width:78%;padding:10px 14px;border-radius:14px;white-space:pre-wrap;line-height:1.45;font-size:15px}
 .u .bub{background:var(--teal);color:#fff;border-bottom-right-radius:4px}
 .a .bub{background:#fff;border:1px solid #e4e8e6;border-bottom-left-radius:4px}
 .meta{font-size:11px;color:#9aa49f;margin:4px 8px 0;text-align:right} .rec{color:#2f6b25;font-weight:600}
 form{display:flex;gap:8px;padding:12px 18px;background:#fff;border-top:1px solid #e4e8e6;max-width:820px;width:100%;margin:0 auto}
 #q{flex:1;padding:11px 13px;border:1px solid #cfd6d3;border-radius:10px;font-size:15px}
 button{padding:11px 18px;background:var(--teal);color:#fff;border:0;border-radius:10px;font-size:15px;cursor:pointer}
 button:disabled{opacity:.5;cursor:default}
</style></head><body>
<header><h1>Cairn — DeepSeek-V4-Flash</h1><div>__MODEL__ · distributed across spot GPUs · survives reclaims with live failover</div></header>
<div id="log"></div>
<form id="f"><input id="q" autocomplete="off" placeholder="Ask anything…" /><button id="b">Send</button></form>
<script>
const MODEL="__MODEL__";
const log=document.getElementById("log"),q=document.getElementById("q"),b=document.getElementById("b"),f=document.getElementById("f");
const msgs=[];
function add(role,text){const m=document.createElement("div");m.className="msg "+(role==="user"?"u":"a");
  const x=document.createElement("div");x.className="bub";x.textContent=text;m.appendChild(x);log.appendChild(m);log.scrollTop=log.scrollHeight;return x;}
f.onsubmit=async(e)=>{e.preventDefault();const text=q.value.trim();if(!text)return;
  q.value="";b.disabled=true;add("user",text);msgs.push({role:"user",content:text});
  const bub=add("assistant","…");const t0=performance.now();
  try{const r=await fetch("/v1/chat/completions",{method:"POST",headers:{"content-type":"application/json"},
      body:JSON.stringify({model:MODEL,messages:msgs,max_tokens:256,temperature:0.7})});
    const j=await r.json();
    if(j.error){bub.textContent="⚠ "+(j.error.message||JSON.stringify(j.error));}
    else{const c=j.choices[0].message.content;bub.textContent=c;msgs.push({role:"assistant",content:c});
      const dt=((performance.now()-t0)/1000).toFixed(1);let meta=dt+"s · "+(j.usage?j.usage.completion_tokens:"?")+" tok";
      if(j.cairn&&j.cairn.recovered)meta+=' · <span class="rec">recovered mid-request ('+j.cairn.mttr_s+'s)</span>';
      bub.parentNode.insertAdjacentHTML("afterend",'<div class="meta">'+meta+'</div>');}
  }catch(err){bub.textContent="⚠ "+err;}
  b.disabled=false;q.focus();};
q.focus();
</script></body></html>""".replace("__MODEL__", MODEL)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path in ("/", "/chat"):
            body = HTML.encode()
            self.send_response(200); self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("content-length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        req = urllib.request.Request(UPSTREAM + self.path, data=raw,
            headers={"content-type": "application/json", "authorization": "Bearer " + KEY})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = r.read(); code = r.getcode()
        except urllib.error.HTTPError as e:
            out = e.read(); code = e.code
        except Exception as e:
            out = json.dumps({"error": {"message": str(e)}}).encode(); code = 502
        self.send_response(code); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out))); self.end_headers(); self.wfile.write(out)


if __name__ == "__main__":
    print(f"Cairn chat → http://localhost:{PORT}   (proxying {UPSTREAM}, model {MODEL})")
    print(f"Make sure the tunnel is up:  ssh -L 8000:localhost:8000 cairn-dsv4 -N &")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
