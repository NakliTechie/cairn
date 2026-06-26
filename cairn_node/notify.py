"""cairn_node.notify — a signal path back to the HUMAN when a box drains or dies.

Detection already exists (box-side `cairn_node.spot_watch` polling IMDS for the ~2-min interruption
notice; dev-side `infra/skypilot/watch.py` reconciling `sky status` against the ledger). What was missing
is a channel that reaches *us* when we're away from the dev machine — so a spot reclaim isn't silent and
we're not "waiting in the blind." This is that channel. Used by:
  - serve.py `_on_spot_notice` (box-side) — fires on the box's OWN ~2-min notice (the earliest signal,
    and it works even if the dev machine / watch.py isn't running).
  - infra/skypilot/watch.py (dev-side) — fires when a box that was UP vanishes (the abrupt-death safety net).

Channel-agnostic. Set EITHER (in infra/secrets.env, passed to boxes via --env):
  - CAIRN_ALERT_WEBHOOK : any URL we POST to. Zero-setup default = ntfy:
        CAIRN_ALERT_WEBHOOK=https://ntfy.sh/<your-unguessable-topic>   (install the ntfy app, subscribe)
      Slack/Discord incoming-webhook URLs are auto-detected and sent as JSON {"text": ...}.
  - CALLMEBOT_PHONE + CALLMEBOT_APIKEY : WhatsApp via CallMeBot (the global playbook's "reach me" channel).
Also best-effort local desktop (macOS) so the dev box still beeps. NEVER raises — a dead notifier must
never take down serving or the watcher.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.parse
import urllib.request


def alert(msg: str, *, title: str = "Cairn fleet") -> bool:
    """Fire the alert on every configured channel + local desktop. Returns True if any remote channel was
    attempted (i.e. a channel is configured). Never raises."""
    sent = _webhook(msg, title) | _callmebot(f"{title}: {msg}")
    _desktop(msg, title)
    return sent


def _webhook(msg: str, title: str) -> bool:
    url = os.environ.get("CAIRN_ALERT_WEBHOOK")
    if not url:
        return False
    try:
        if "slack" in url or "discord" in url:               # incoming-webhook → JSON
            data = json.dumps({"text": f"{title}: {msg}"}).encode()
            headers = {"Content-Type": "application/json"}
        else:                                                # ntfy / generic → raw body IS the message
            data = msg.encode()
            headers = {"Title": title, "Priority": "high", "Tags": "rotating_light"}
        urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method="POST"),
                               timeout=5)
    except Exception:
        pass
    return True


def _callmebot(text: str) -> bool:
    phone, key = os.environ.get("CALLMEBOT_PHONE"), os.environ.get("CALLMEBOT_APIKEY")
    if not (phone and key):
        return False
    try:
        q = urllib.parse.urlencode({"phone": phone, "text": text, "apikey": key})
        urllib.request.urlopen(f"https://api.callmebot.com/whatsapp.php?{q}", timeout=8)
    except Exception:
        pass
    return True


def _desktop(msg: str, title: str) -> None:
    try:                                                     # macOS only; harmless no-op on the Linux boxes
        subprocess.run(["osascript", "-e", f'display notification {json.dumps(msg)} with title {json.dumps(title)}'],
                       check=False, capture_output=True, timeout=5)
    except Exception:
        pass
