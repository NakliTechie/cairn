"""cairn_node.spot_watch — the AWS spot-interruption watcher (proactive recovery's signal source).

Pure logic, no EC2 / no GPU: an injected `http(method, path, headers, timeout) -> (status, body)` fakes
IMDS so we can assert: a pending interruption / rebalance notice is detected; a quiet box returns None;
IMDS being unreachable (not on EC2) is treated as 'no notice' (never raises)."""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cairn_node.spot_watch import check_interruption, watch  # noqa: E402


def _imds(action=None, rebalance=False):
    """A fake IMDS: PUT /token -> 200 TOK; the interruption/rebalance GETs return 200+body or 404."""
    def http(method, path, headers=None, timeout=1.0):
        if method == "PUT" and "token" in path:
            return 200, "TOK"
        if "instance-action" in path:
            return (200, '{"action":"%s","time":"2026-06-22T12:00:00Z"}' % action) if action else (404, "")
        if "rebalance" in path:
            return (200, '{"noticeTime":"2026-06-22T11:58:00Z"}') if rebalance else (404, "")
        return 404, ""
    return http


def test_detects_interruption_notice():
    n = check_interruption(http=_imds(action="terminate"))
    assert n is not None and n["action"] == "terminate" and n["_kind"] == "interruption"


def test_detects_rebalance_recommendation():
    n = check_interruption(http=_imds(rebalance=True))
    assert n is not None and n["_kind"] == "rebalance"


def test_quiet_box_returns_none():
    assert check_interruption(http=_imds()) is None


def test_imds_unreachable_is_treated_as_no_notice():
    def boom(method, path, headers=None, timeout=1.0):
        raise OSError("not on EC2 — 169.254.169.254 unreachable")
    assert check_interruption(http=boom) is None          # never raises → safe to run anywhere


def test_watch_fires_callback_once_on_notice():
    fired = []
    n = watch(lambda notice: fired.append(notice), interval=0, http=_imds(action="terminate"))
    assert n is not None and len(fired) == 1 and fired[0]["action"] == "terminate"
