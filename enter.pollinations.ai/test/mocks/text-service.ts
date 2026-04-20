import { env } from "cloudflare:test";
import { Hono } from "hono";
import { stream } from "hono/streaming";
import type { MockAPI } from "./fetch.ts";
import { createHonoMockHandler } from "./fetch.ts";

type TextServiceState = {
    /** When true, returns JSON even for stream: true requests (simulates upstream bug) */
    forceNonStreaming: boolean;
};

type MockEmbeddingRequest = {
    model?: string;
    input?: unknown;
    dimensions?: unknown;
};

type MockContentPart =
    | {
          type: "image_url";
          image_url: { url: string };
      }
    | {
          type: "video_url";
          video_url: { url: string };
      };

const MAX_EMBEDDING_MEDIA_SIZE = 20 * 1024 * 1024;

export function createMockTextService(): MockAPI<TextServiceState> {
    const state: TextServiceState = { forceNonStreaming: false };

    const app = new Hono()
        .post("/openai", async (c) => {
            // Add realistic delay to simulate actual service response time
            await new Promise((resolve) => setTimeout(resolve, 100));

            const body = await c.req.json();
            const isStreaming =
                body.stream === true && !state.forceNonStreaming;

            if (isStreaming) {
                // streaming response in SSE format (match real text service headers)
                c.header("Content-Type", "text/event-stream; charset=utf-8");
                c.header("Cache-Control", "no-cache");
                c.header("Connection", "keep-alive");
                return stream(c, async (stream) => {
                    for await (const chunk of mockOpenAIStream(
                        "Hello, whats up?",
                    )) {
                        await stream.write(chunk);
                    }
                });
            }

            // set usage headers
            c.header("x-model-used", "gpt-5-nano-2025-08-07");
            c.header("x-usage-prompt-text-tokens", "1000");
            c.header("x-usage-prompt-cached-tokens", "0");
            c.header("x-usage-prompt-audio-tokens", "0");
            c.header("x-usage-prompt-image-tokens", "0");
            c.header("x-usage-completion-text-tokens", "500");
            c.header("x-usage-completion-reasoning-tokens", "0");
            c.header("x-usage-completion-audio-tokens", "0");
            c.header("x-usage-completion-image-tokens", "0");
            // regular response
            return c.json({
                id: `chatcmpl-mock-${Date.now()}`,
                object: "chat.completion",
                created: Math.floor(Date.now() / 1000),
                model: "gpt-5-nano-2025-08-07",
                choices: [
                    {
                        index: 0,
                        message: {
                            role: "assistant",
                            content: "Hi!",
                        },
                        finish_reason: "stop",
                    },
                ],
                usage: mockUsage,
            });
        })
        .post("/embeddings", async (c) => {
            const body = (await c.req.json()) as MockEmbeddingRequest;
            const embeddingError = getMockEmbeddingError(body.input);
            if (embeddingError) {
                return c.json({ error: embeddingError }, 400);
            }

            return createMockEmbeddingsResponse(body);
        })
        .post("/v1/embeddings", async (c) => {
            const body = (await c.req.json()) as MockEmbeddingRequest;
            const embeddingError = getMockEmbeddingError(body.input);
            if (embeddingError) {
                return c.json({ error: embeddingError }, 400);
            }

            return createMockEmbeddingsResponse(body);
        });

    const configuredHost = env.TEXT_SERVICE_URL
        ? new URL(env.TEXT_SERVICE_URL).host
        : "ec2-54-147-14-220.compute-1.amazonaws.com:16385";

    return {
        state,
        handlerMap: {
            [configuredHost]: createHonoMockHandler(app),
        },
        reset: () => {
            state.forceNonStreaming = false;
        },
    };
}

const mockUsage = {
    prompt_tokens: 1000,
    completion_tokens: 500,
    total_tokens: 1500,
    prompt_tokens_details: {
        cached_tokens: 0,
        audio_tokens: 0,
    },
    completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
    },
};

async function* mockOpenAIStream(
    message: string,
    delay: number = 0,
): AsyncIterable<string> {
    const parts = message.split(/(?= )/);
    for (const part of parts) {
        if (delay > 0) {
            await new Promise((resolve) => setTimeout(resolve, delay));
        }
        yield `data: ${JSON.stringify({
            id: `chatcmpl-chunk-mock-${Date.now()}`,
            object: "chat.completion.chunk",
            created: Date.now(),
            model: "gpt-5-nano-2025-08-07",
            choices: [
                {
                    index: 0,
                    delta: {
                        content: part,
                    },
                    finish_reason: null,
                },
            ],
        })}\n\n`;
    }
    yield `data: ${JSON.stringify({
        id: `chatcmpl-chunk-mock-${Date.now()}`,
        object: "chat.completion.chunk",
        created: Date.now(),
        model: "gpt-5-nano-2025-08-07",
        choices: [],
        usage: mockUsage,
    })}\n\n`;
    yield `data: [DONE]\n\n`;
}

function createMockEmbeddingsResponse(body: MockEmbeddingRequest): Response {
    const dimensions =
        typeof body.dimensions === "number" && body.dimensions > 0
            ? body.dimensions
            : 3_072;
    const items = normalizeEmbeddingInputs(body.input);
    const promptTokens = Math.max(1, items.length);

    return new Response(
        JSON.stringify({
            object: "list",
            data: items.map((_, index) => ({
                object: "embedding",
                embedding: Array.from(
                    { length: dimensions },
                    (_unused, offset) => (index + offset + 1) / dimensions,
                ),
                index,
            })),
            model: body.model ?? "gemini-embedding-2-preview",
            usage: {
                prompt_tokens: promptTokens,
                total_tokens: promptTokens,
            },
        }),
        {
            headers: {
                "Content-Type": "application/json",
                "x-model-used": body.model ?? "gemini-embedding-2-preview",
                "x-usage-prompt-text-tokens": String(promptTokens),
                "x-usage-prompt-cached-tokens": "0",
                "x-usage-prompt-audio-tokens": "0",
                "x-usage-prompt-image-tokens": "0",
                "x-usage-completion-text-tokens": "0",
                "x-usage-completion-reasoning-tokens": "0",
                "x-usage-completion-audio-tokens": "0",
                "x-usage-completion-image-tokens": "0",
            },
        },
    );
}

function normalizeEmbeddingInputs(input: unknown): unknown[] {
    if (input == null) {
        return [];
    }
    if (!Array.isArray(input)) {
        return [input];
    }
    if (input.length === 0) {
        return [];
    }
    if (typeof input[0] === "string") {
        return input;
    }
    return [input];
}

function getMockEmbeddingError(input: unknown): string | null {
    for (const part of getContentParts(input)) {
        const label = part.type === "video_url" ? "Video" : "Image";
        const url =
            part.type === "video_url" ? part.video_url.url : part.image_url.url;
        if (url.startsWith("data:")) {
            const byteLength = getDataUrlByteLength(url);
            if (byteLength == null) {
                return `Invalid ${label.toLowerCase()} data URL`;
            }
            if (byteLength > MAX_EMBEDDING_MEDIA_SIZE) {
                return `${label} too large: ${byteLength} bytes (max ${MAX_EMBEDDING_MEDIA_SIZE})`;
            }
            continue;
        }

        const parsedUrl = new URL(url);
        if (isPrivateIpv4Address(parsedUrl.hostname)) {
            return `Blocked request to private/internal URL: ${parsedUrl.hostname} -> ${parsedUrl.hostname}`;
        }
    }

    return null;
}

function getContentParts(input: unknown): MockContentPart[] {
    if (!input || typeof input !== "object") {
        return [];
    }

    if (!Array.isArray(input)) {
        return isMockContentPart(input) ? [input] : [];
    }

    return input.filter(isMockContentPart);
}

function isMockContentPart(value: unknown): value is MockContentPart {
    if (!value || typeof value !== "object" || !("type" in value)) {
        return false;
    }

    if (value.type === "image_url") {
        return (
            "image_url" in value &&
            !!value.image_url &&
            typeof value.image_url === "object" &&
            "url" in value.image_url &&
            typeof value.image_url.url === "string"
        );
    }

    if (value.type === "video_url") {
        return (
            "video_url" in value &&
            !!value.video_url &&
            typeof value.video_url === "object" &&
            "url" in value.video_url &&
            typeof value.video_url.url === "string"
        );
    }

    return false;
}

function getDataUrlByteLength(dataUrl: string): number | null {
    const [meta, payload] = dataUrl.split(",", 2);
    if (!meta?.startsWith("data:") || payload == null) {
        return null;
    }

    if (meta.includes(";base64")) {
        return Buffer.from(payload, "base64").byteLength;
    }

    try {
        return Buffer.byteLength(decodeURIComponent(payload), "utf8");
    } catch {
        return null;
    }
}

function isPrivateIpv4Address(address: string): boolean {
    const octets = address
        .split(".")
        .map((octet) => Number.parseInt(octet, 10));
    if (octets.length !== 4 || octets.some(Number.isNaN)) {
        return false;
    }
    const [a, b] = octets;
    return (
        a === 0 ||
        a === 10 ||
        a === 127 ||
        (a === 169 && b === 254) ||
        (a === 172 && b >= 16 && b <= 31) ||
        (a === 192 && b === 168)
    );
}
