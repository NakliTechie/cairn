# demo/ — the rung-1 stack, live

`server.py` runs the whole no-GPU stack behind an OpenAI-compatible HTTP API: the
gateway (auth/validate/admit) → the multi-stream scheduler → the mock block pipeline →
recovery. Only the SGLang block forward is mocked; the control + scheduling + recovery
logic is the real thing.

```sh
uv run --python 3.9 --with pyyaml python demo/server.py        # http://127.0.0.1:8400

curl localhost:8400/version
curl localhost:8400/v1/health
curl -s localhost:8400/demo/scenario | python -m json.tool      # K streams + induced recovery

curl -s localhost:8400/v1/chat/completions \
  -H 'authorization: Bearer sk-cairn-demo' -H 'content-type: application/json' \
  -d '{"model":"gpt-oss-120b","messages":[{"role":"user","content":"hi"}],"max_tokens":12}'
```

- `/demo/scenario` decodes 6 streams through the N-stage split, kills a middle stage
  mid-decode, reassigns to the warm spare, replays KV, and resumes — returning the
  occupancy + recovery timeline. (Tokens are mock integers; the real content arrives
  with the SGLang forward at rung 2.)
- Verified by `scheduler/tests/test_demo_server.py` (real in-process HTTP round-trip).
