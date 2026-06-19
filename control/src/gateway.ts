// Gateway logic (spec §7) — auth, OpenAI-compatible validation, response shaping.
// Pure functions so they unit-test in plain node; the Worker fetch handler (index.ts)
// wires them to Request/Response. Mirrors scheduler/.../gateway.py.

export class GatewayError extends Error {
  constructor(
    public status: number,
    message: string,
    public code: string = "invalid_request_error",
  ) {
    super(message);
  }
  toError() {
    return { error: { message: this.message, type: this.code, code: this.code } };
  }
}

export interface ChatMessage {
  role: string;
  content: string;
}

export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  max_tokens: number;
  stream: boolean;
  temperature: number;
}

// One door (invariant #7). Bearer-token auth against the configured key set.
export function authenticate(authorization: string | null, apiKeys: Set<string>): string {
  if (!authorization || !authorization.startsWith("Bearer ")) {
    throw new GatewayError(401, "missing or malformed Authorization header", "authentication_error");
  }
  const key = authorization.slice("Bearer ".length).trim();
  if (!apiKeys.has(key)) {
    throw new GatewayError(401, "invalid API key", "authentication_error");
  }
  return key;
}

// Ingress limits — bound every request (forward-pass H1/H2). Keep in sync with the Python
// gateway (scheduler/src/cairn_scheduler/gateway.py).
export const MAX_OUTPUT_TOKENS = 4096;
export const MAX_MESSAGES = 256;
export const MAX_PROMPT_CHARS = 128_000;

export function parseRequest(body: unknown, modelNames: Set<string>): ChatRequest {
  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    throw new GatewayError(400, "request body must be a JSON object");
  }
  const b = body as Record<string, unknown>;
  const model = b.model;
  if (typeof model !== "string" || !model) {
    throw new GatewayError(400, "missing required field: model");
  }
  if (!modelNames.has(model)) {
    throw new GatewayError(404, `model '${model}' not found`, "model_not_found");
  }
  const messages = b.messages;
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new GatewayError(400, "messages must be a non-empty array");
  }
  if (messages.length > MAX_MESSAGES) {
    throw new GatewayError(400, `too many messages (max ${MAX_MESSAGES})`);
  }
  let totalChars = 0;
  for (const m of messages) {
    if (typeof m !== "object" || m === null || !("role" in m) || !("content" in m)) {
      throw new GatewayError(400, "each message needs 'role' and 'content'");
    }
    totalChars += String((m as Record<string, unknown>).content ?? "").length;
  }
  if (totalChars > MAX_PROMPT_CHARS) {
    throw new GatewayError(400, `prompt too large (max ${MAX_PROMPT_CHARS} chars)`);
  }
  let maxTokens = 64;
  if (b.max_tokens !== undefined) {
    if (typeof b.max_tokens !== "number" || !Number.isInteger(b.max_tokens) || b.max_tokens < 1) {
      throw new GatewayError(400, "max_tokens must be a positive integer");
    }
    if (b.max_tokens > MAX_OUTPUT_TOKENS) {
      throw new GatewayError(400, `max_tokens exceeds ceiling (${MAX_OUTPUT_TOKENS})`);
    }
    maxTokens = b.max_tokens;
  }
  return {
    model,
    messages: messages as ChatMessage[],
    max_tokens: maxTokens,
    stream: Boolean(b.stream),
    temperature: typeof b.temperature === "number" ? b.temperature : 0.0,
  };
}

// OpenAI-compatible completion shape. `content` + token counts come from the in-VPC
// data plane; usage is metadata only (spec §8 — counts, not content).
export function formatResponse(
  id: string,
  req: ChatRequest,
  content: string,
  promptTokens: number,
  completionTokens: number,
  finishReason: "stop" | "length" = "stop",
) {
  return {
    id,
    object: "chat.completion",
    model: req.model,
    choices: [
      { index: 0, message: { role: "assistant", content }, finish_reason: finishReason },
    ],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
  };
}
