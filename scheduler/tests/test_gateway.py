import pytest

from cairn_scheduler.gateway import Gateway, GatewayError


def _gw():
    return Gateway(api_keys={"sk-good"}, model_names={"gpt-oss-120b"})


def _body(**over):
    b = {"model": "gpt-oss-120b", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 8}
    b.update(over)
    return b


def test_auth_valid_missing_invalid():
    gw = _gw()
    assert gw.authenticate("Bearer sk-good") == "sk-good"
    for bad in (None, "", "sk-good", "Bearer ", "Bearer sk-nope"):
        with pytest.raises(GatewayError) as e:
            gw.authenticate(bad)
        assert e.value.status == 401


def test_parse_valid_request():
    req = _gw().parse_request(_body())
    assert req.model == "gpt-oss-120b"
    assert req.max_tokens == 8 and req.stream is False


@pytest.mark.parametrize("body,status", [
    ({"messages": [{"role": "user", "content": "x"}]}, 400),          # missing model
    (_body(model="other-model"), 404),                               # unknown model
    (_body(messages=[]), 400),                                       # empty messages
    (_body(messages=[{"role": "user"}]), 400),                       # malformed message
    (_body(max_tokens=0), 400),                                      # bad max_tokens
])
def test_parse_errors(body, status):
    with pytest.raises(GatewayError) as e:
        _gw().parse_request(body)
    assert e.value.status == status
    assert "error" in e.value.to_error()


def test_admit_builds_stream():
    gw = _gw()
    req = gw.parse_request(_body(max_tokens=5))
    s = gw.admit(req)
    assert s.id.startswith("chatcmpl-")
    assert len(s.prompt) >= 1 and s.max_new_tokens == 5


def test_format_response_shape_and_usage():
    gw = _gw()
    req = gw.parse_request(_body(max_tokens=3))
    s = gw.admit(req)
    s.generated = [101, 102, 103]  # pretend the scheduler decoded these
    resp = gw.format_response(req, s)
    assert resp["object"] == "chat.completion"
    assert resp["choices"][0]["message"]["role"] == "assistant"
    assert resp["choices"][0]["finish_reason"] == "stop"
    assert resp["usage"]["completion_tokens"] == 3
    assert resp["usage"]["total_tokens"] == len(s.prompt) + 3


def test_stream_chunks_end_with_stop():
    gw = _gw()
    req = gw.parse_request(_body())
    s = gw.admit(req)
    s.generated = [1, 2]
    chunks = list(gw.stream_chunks(req, s))
    assert len(chunks) == 3  # 2 content + 1 final
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_version_info():
    assert _gw().version_info()["service"] == "cairn"
