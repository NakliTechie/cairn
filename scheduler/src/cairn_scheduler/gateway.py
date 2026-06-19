"""Gateway — the Workers ingress logic (spec §7), in plain Python so the validation /
admission / response-shaping rules are testable and portable to a CF Worker.

One ingress (invariant #7): auth → validate → admit → route → stream tokens back. The
API is **OpenAI-compatible** — that is the agent face (handoff §12), no second surface.
This module is *not in the per-token hot path* (invariant #4): it shapes the request into
a `Stream` for the in-VPC scheduler and shapes the result back; the decode loop is the
scheduler's.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Iterator, List, Optional, Set

from .runtime import _mix
from .scheduler import Stream


class GatewayError(Exception):
    """An admission-time rejection, carrying an HTTP-ish status (OpenAI error shape)."""

    def __init__(self, status: int, message: str, code: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code

    def to_error(self) -> dict:
        return {"error": {"message": self.message, "type": self.code, "code": self.code}}


@dataclass
class ChatRequest:
    model: str
    messages: List[dict]
    max_tokens: int = 64
    stream: bool = False
    temperature: float = 0.0


def _prompt_to_hiddens(messages: List[dict]) -> List[int]:
    """Stand-in tokenizer (the real one lives in the fork): fold each message's text into
    a deterministic sequence of integer 'token' hiddens for the sim."""
    hiddens: List[int] = []
    for m in messages:
        text = str(m.get("content", ""))
        acc = _mix(0xA11CE, len(text) + 1)
        for ch in text:
            acc = _mix(acc, ord(ch))
            hiddens.append(acc & 0xFFFF)
    return hiddens or [_mix(0xA11CE, 1) & 0xFFFF]  # never-empty prompt


# Ingress limits — bound every request so one caller can't fan an unbounded decode or a
# giant prompt into the fleet (forward-pass H1/H2). Keep in sync with control/src/gateway.ts.
MAX_OUTPUT_TOKENS = 4096
MAX_MESSAGES = 256
MAX_PROMPT_CHARS = 128_000


class Gateway:
    def __init__(self, *, api_keys: Set[str], model_names: Set[str], default_version: str = "v1",
                 max_output_tokens: int = MAX_OUTPUT_TOKENS, max_messages: int = MAX_MESSAGES,
                 max_prompt_chars: int = MAX_PROMPT_CHARS) -> None:
        self.api_keys = set(api_keys)
        self.model_names = set(model_names)
        self.version = default_version  # surfaced at /version (handoff §14)
        self.max_output_tokens = max_output_tokens
        self.max_messages = max_messages
        self.max_prompt_chars = max_prompt_chars
        self._counter = 0

    # --- auth (one door, invariant #7) ---
    def authenticate(self, authorization: Optional[str]) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise GatewayError(401, "missing or malformed Authorization header", "authentication_error")
        key = authorization[len("Bearer "):].strip()
        # Constant-time compare (M8). `any` only short-circuits on a match (a valid key the
        # attacker already holds); every invalid key is checked against all entries.
        if not any(hmac.compare_digest(key, k) for k in self.api_keys):
            raise GatewayError(401, "invalid API key", "authentication_error")
        return key

    # --- validation ---
    def parse_request(self, body: dict) -> ChatRequest:
        if not isinstance(body, dict):
            raise GatewayError(400, "request body must be a JSON object")
        model = body.get("model")
        if not model:
            raise GatewayError(400, "missing required field: model")
        if model not in self.model_names:
            raise GatewayError(404, f"model '{model}' not found", "model_not_found")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise GatewayError(400, "messages must be a non-empty array")
        if len(messages) > self.max_messages:
            raise GatewayError(400, f"too many messages (max {self.max_messages})")
        total_chars = 0
        for m in messages:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise GatewayError(400, "each message needs 'role' and 'content'")
            total_chars += len(str(m.get("content", "")))
        if total_chars > self.max_prompt_chars:
            raise GatewayError(400, f"prompt too large (max {self.max_prompt_chars} chars)")
        max_tokens = body.get("max_tokens", 64)
        # `type(...) is int` rejects bool (isinstance(True, int) is True) — M14.
        if type(max_tokens) is not int or max_tokens < 1:
            raise GatewayError(400, "max_tokens must be a positive integer")
        if max_tokens > self.max_output_tokens:
            raise GatewayError(400, f"max_tokens exceeds ceiling ({self.max_output_tokens})")
        return ChatRequest(
            model=model, messages=messages, max_tokens=max_tokens,
            stream=bool(body.get("stream", False)), temperature=float(body.get("temperature", 0.0)),
        )

    def admit(self, req: ChatRequest) -> Stream:
        """Shape a validated request into a scheduler Stream (route-setup; the scheduler
        owns admission against K_max)."""
        self._counter += 1
        sid = f"chatcmpl-{self._counter}"
        return Stream(id=sid, prompt=_prompt_to_hiddens(req.messages), max_new_tokens=req.max_tokens)

    # --- response shaping (OpenAI-compatible) ---
    @staticmethod
    def _decode(token_ids: List[int]) -> str:
        # Stand-in detokeniser for the sim.
        return " ".join(str(t) for t in token_ids)

    def format_response(self, req: ChatRequest, stream: Stream) -> dict:
        prompt_tokens = len(stream.prompt)
        completion_tokens = len(stream.generated)
        return {
            "id": stream.id,
            "object": "chat.completion",
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": self._decode(stream.generated)},
                "finish_reason": "stop" if stream.done else "length",
            }],
            "usage": {  # metadata only — billing needs counts, not content (spec §8)
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def stream_chunks(self, req: ChatRequest, stream: Stream) -> Iterator[dict]:
        """SSE-style chunks (the real Worker writes these to the response body)."""
        for tok in stream.generated:
            yield {
                "id": stream.id, "object": "chat.completion.chunk", "model": req.model,
                "choices": [{"index": 0, "delta": {"content": self._decode([tok])}, "finish_reason": None}],
            }
        yield {
            "id": stream.id, "object": "chat.completion.chunk", "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    def version_info(self) -> dict:
        return {"service": "cairn", "version": self.version}
