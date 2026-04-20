import {
    createExecutionContext,
    env,
    SELF,
    waitOnExecutionContext,
} from "cloudflare:test";
import type { ModelName } from "@shared/registry/registry.ts";
import { describe, expect } from "vitest";
import worker from "@/index.ts";
import { test } from "../fixtures.ts";
import { assertTrackedBillingEvent } from "../helpers/billing-assertions.ts";

const EMBEDDINGS_ENDPOINT = "http://localhost:3000/api/generate/v1/embeddings";
const EMBEDDING_MODELS_ENDPOINT =
    "http://localhost:3000/api/generate/embeddings/models";
const TEST_EMBEDDING_MODEL = "gemini-embedding-2";
const TEST_EMBEDDING_INPUT = "Hello world";
const MAX_MEDIA_SIZE = 20 * 1024 * 1024;

function buildEmbeddingsBody(extra: Record<string, unknown> = {}) {
    return JSON.stringify({
        model: TEST_EMBEDDING_MODEL,
        input: TEST_EMBEDDING_INPUT,
        ...extra,
    });
}

describe("POST /generate/v1/embeddings (authenticated)", () => {
    test(
        "returns an OpenAI-compatible response and tracks billing",
        { timeout: 30000 },
        async ({ apiKey, mocks }) => {
            await mocks.enable("polar", "tinybird", "vcr");
            const ctx = createExecutionContext();
            const response = await worker.fetch(
                new Request(EMBEDDINGS_ENDPOINT, {
                    method: "POST",
                    headers: {
                        "content-type": "application/json",
                        authorization: `Bearer ${apiKey}`,
                    },
                    body: buildEmbeddingsBody(),
                }),
                env,
                ctx,
            );
            const body = await response.text();
            expect(
                response.status,
                `Expected 200 but got ${response.status}: ${body}`,
            ).toBe(200);

            const data = JSON.parse(body) as {
                object: string;
                data: { object: string; embedding: number[]; index: number }[];
                model: string;
                usage: { prompt_tokens: number; total_tokens: number };
            };
            expect(data.object).toBe("list");
            expect(data.data).toHaveLength(1);
            expect(data.data[0].object).toBe("embedding");
            expect(data.data[0].embedding).toBeInstanceOf(Array);
            expect(data.data[0].embedding.length).toBeGreaterThan(0);
            expect(data.data[0].index).toBe(0);
            expect(data.model).toBe(TEST_EMBEDDING_MODEL);
            expect(data.usage.prompt_tokens).toBeGreaterThan(0);
            expect(data.usage.total_tokens).toBe(data.usage.prompt_tokens);
            expect(response.headers.get("x-model-used")).toBe(
                TEST_EMBEDDING_MODEL,
            );

            await waitOnExecutionContext(ctx);

            const events = mocks.tinybird.state.events;
            expect(events).toHaveLength(1);
            expect(events[0].tokenCountPromptText).toBeGreaterThan(0);
            expect(events[0].tokenCountCompletionText).toBe(0);
            assertTrackedBillingEvent(
                events[0],
                TEST_EMBEDDING_MODEL as ModelName,
            );
        },
    );

    test(
        "supports custom dimensions",
        { timeout: 30000 },
        async ({ apiKey, mocks }) => {
            await mocks.enable("polar", "tinybird", "vcr");
            const response = await SELF.fetch(EMBEDDINGS_ENDPOINT, {
                method: "POST",
                headers: {
                    "content-type": "application/json",
                    authorization: `Bearer ${apiKey}`,
                },
                body: buildEmbeddingsBody({ dimensions: 768 }),
            });
            const body = await response.text();
            expect(
                response.status,
                `Expected 200 but got ${response.status}: ${body}`,
            ).toBe(200);

            const data = JSON.parse(body) as {
                data: { embedding: number[] }[];
            };
            expect(data.data[0].embedding).toHaveLength(768);
        },
    );

    test(
        "supports batch input",
        { timeout: 30000 },
        async ({ apiKey, mocks }) => {
            await mocks.enable("polar", "tinybird", "vcr");
            const response = await SELF.fetch(EMBEDDINGS_ENDPOINT, {
                method: "POST",
                headers: {
                    "content-type": "application/json",
                    authorization: `Bearer ${apiKey}`,
                },
                body: buildEmbeddingsBody({ input: ["Hello", "World"] }),
            });
            const body = await response.text();
            expect(
                response.status,
                `Expected 200 but got ${response.status}: ${body}`,
            ).toBe(200);

            const data = JSON.parse(body) as {
                data: { index: number }[];
            };
            expect(data.data).toHaveLength(2);
            expect(data.data[0].index).toBe(0);
            expect(data.data[1].index).toBe(1);
        },
    );

    test(
        "rejects models that do not support embeddings",
        { timeout: 10000 },
        async ({ apiKey, mocks }) => {
            await mocks.enable("polar", "tinybird");
            const response = await SELF.fetch(EMBEDDINGS_ENDPOINT, {
                method: "POST",
                headers: {
                    "content-type": "application/json",
                    authorization: `Bearer ${apiKey}`,
                },
                body: buildEmbeddingsBody({ model: "flux" }),
            });
            const body = await response.text();
            expect(response.status).toBe(400);

            const error = JSON.parse(body) as {
                error: { message: string };
            };
            expect(error.error.message).toContain(
                "does not support embeddings",
            );
            expect(error.error.message).toContain("flux");
        },
    );

    test(
        "blocks private media URLs",
        { timeout: 10000 },
        async ({ apiKey, mocks }) => {
            await mocks.enable("polar", "tinybird", "text");
            const response = await SELF.fetch(EMBEDDINGS_ENDPOINT, {
                method: "POST",
                headers: {
                    "content-type": "application/json",
                    authorization: `Bearer ${apiKey}`,
                },
                body: buildEmbeddingsBody({
                    input: [
                        {
                            type: "image_url",
                            image_url: { url: "http://127.0.0.1/test.png" },
                        },
                    ],
                }),
            });
            const body = await response.text();
            expect(response.status).toBe(400);

            const error = JSON.parse(body) as {
                error: { message: string };
            };
            expect(error.error.message).toContain(
                "Blocked request to private/internal URL",
            );
            expect(error.error.message).toContain("127.0.0.1");
        },
    );

    test(
        "rejects oversized data URLs",
        { timeout: 10000 },
        async ({ apiKey, mocks }) => {
            await mocks.enable("polar", "tinybird", "text");
            const oversizedDataUrl = `data:image/png,${"a".repeat(MAX_MEDIA_SIZE + 1)}`;
            const response = await SELF.fetch(EMBEDDINGS_ENDPOINT, {
                method: "POST",
                headers: {
                    "content-type": "application/json",
                    authorization: `Bearer ${apiKey}`,
                },
                body: buildEmbeddingsBody({
                    input: [
                        {
                            type: "image_url",
                            image_url: { url: oversizedDataUrl },
                        },
                    ],
                }),
            });
            const body = await response.text();
            expect(response.status).toBe(400);

            const error = JSON.parse(body) as {
                error: { message: string };
            };
            expect(error.error.message).toContain("Image too large");
        },
    );
});

describe("POST /generate/v1/embeddings (unauthenticated)", () => {
    test(
        "rejects unauthenticated requests",
        { timeout: 10000 },
        async ({ mocks }) => {
            await mocks.enable("polar", "tinybird");
            const response = await SELF.fetch(EMBEDDINGS_ENDPOINT, {
                method: "POST",
                headers: {
                    "content-type": "application/json",
                },
                body: buildEmbeddingsBody(),
            });
            expect(response.status).toBe(401);
        },
    );
});

describe("GET /embeddings/models", () => {
    test(
        "returns the public embeddings model list",
        { timeout: 10000 },
        async ({ mocks }) => {
            await mocks.enable("polar", "tinybird");
            const response = await SELF.fetch(EMBEDDING_MODELS_ENDPOINT, {
                method: "GET",
            });
            expect(response.status).toBe(200);

            const data = (await response.json()) as {
                object: string;
                data: { id: string; object: string }[];
            };
            expect(data.object).toBe("list");
            expect(data.data.length).toBeGreaterThan(0);
            expect(data.data[0].id).toBe(TEST_EMBEDDING_MODEL);
            expect(data.data[0].object).toBe("model");
        },
    );
});
