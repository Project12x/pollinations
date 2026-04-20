import { lookup } from "node:dns/promises";
import { isIP } from "node:net";
import type { Usage } from "../shared/registry/registry.ts";
import { buildUsageHeaders } from "../shared/registry/usage-headers.ts";
import googleCloudAuth from "./auth/googleCloudAuth.ts";

const GOOGLE_PROJECT_ID = process.env.GOOGLE_PROJECT_ID;
const VERTEX_REGION = "us-central1";

const MAX_MEDIA_SIZE = 20 * 1024 * 1024; // 20MB max per media item

export class EmbeddingInputError extends Error {
    readonly status = 400;
}

/**
 * Block internal/metadata URLs to prevent SSRF.
 * Throws if the URL points to a private or internal network address.
 */
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

function isPrivateIpv6Address(address: string): boolean {
    const normalized = address.toLowerCase().split("%", 2)[0];
    if (normalized === "::1") {
        return true;
    }
    if (normalized.startsWith("::ffff:")) {
        return isPrivateIpv4Address(normalized.slice(7));
    }
    if (normalized.startsWith("fc") || normalized.startsWith("fd")) {
        return true;
    }
    if (normalized.startsWith("fe")) {
        const firstHextet = Number.parseInt(normalized.slice(0, 4), 16);
        return (
            !Number.isNaN(firstHextet) &&
            firstHextet >= 0xfe80 &&
            firstHextet <= 0xfebf
        );
    }
    return false;
}

function isPrivateIpAddress(address: string): boolean {
    const version = isIP(address);
    if (version === 4) {
        return isPrivateIpv4Address(address);
    }
    if (version === 6) {
        return isPrivateIpv6Address(address);
    }
    return false;
}

function assertMediaSize(label: string, byteLength: number): void {
    if (byteLength > MAX_MEDIA_SIZE) {
        throw new EmbeddingInputError(
            `${label} too large: ${byteLength} bytes (max ${MAX_MEDIA_SIZE})`,
        );
    }
}

function parseDataUrl(
    dataUrl: string,
    label: string,
): { mimeType: string; data: string } {
    const [meta, payload] = dataUrl.split(",", 2);
    if (!meta?.startsWith("data:") || !payload) {
        throw new EmbeddingInputError(
            `Invalid ${label.toLowerCase()} data URL`,
        );
    }

    const mimeType =
        meta.slice(5).split(";", 1)[0] || "application/octet-stream";
    const isBase64 = meta.includes(";base64");
    let buffer: Buffer;
    if (isBase64) {
        buffer = Buffer.from(payload, "base64");
    } else {
        try {
            buffer = Buffer.from(decodeURIComponent(payload), "utf8");
        } catch {
            throw new EmbeddingInputError(
                `Invalid ${label.toLowerCase()} data URL`,
            );
        }
    }

    assertMediaSize(label, buffer.byteLength);
    return { mimeType, data: buffer.toString("base64") };
}

function assertBase64MediaSize(label: string, data: string): void {
    const buffer = Buffer.from(data.replace(/\s+/g, ""), "base64");
    assertMediaSize(label, buffer.byteLength);
}

async function assertPublicUrl(url: string): Promise<URL> {
    const parsed = new URL(url);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        throw new EmbeddingInputError(
            `Unsupported media URL protocol: ${parsed.protocol}`,
        );
    }

    const addresses =
        isIP(parsed.hostname) > 0
            ? [parsed.hostname]
            : (
                  await lookup(parsed.hostname, {
                      all: true,
                      verbatim: true,
                  })
              ).map(({ address }) => address);

    if (addresses.length === 0) {
        throw new EmbeddingInputError(
            `Failed to resolve public media URL: ${parsed.hostname}`,
        );
    }

    for (const address of addresses) {
        if (isPrivateIpAddress(address)) {
            throw new EmbeddingInputError(
                `Blocked request to private/internal URL: ${parsed.hostname} -> ${address}`,
            );
        }
    }
    return parsed;
}

/**
 * Fetch a media URL with SSRF protection and size limits.
 * Returns the raw buffer and content-type header.
 */
async function fetchMedia(
    url: string,
    label: string,
): Promise<{ buffer: ArrayBuffer; contentType: string }> {
    const parsedUrl = await assertPublicUrl(url);
    const response = await fetch(parsedUrl, {
        signal: AbortSignal.timeout(30_000),
    });
    const cl = parseInt(response.headers.get("content-length") || "0", 10);
    if (Number.isFinite(cl)) {
        assertMediaSize(label, cl);
    }
    const buffer = await response.arrayBuffer();
    assertMediaSize(label, buffer.byteLength);
    const contentType =
        response.headers.get("content-type") || "application/octet-stream";
    return { buffer, contentType };
}

// Gemini embedding task types (passed through if provided)
type GeminiTaskType =
    | "SEMANTIC_SIMILARITY"
    | "CLASSIFICATION"
    | "CLUSTERING"
    | "RETRIEVAL_DOCUMENT"
    | "RETRIEVAL_QUERY"
    | "CODE_RETRIEVAL_QUERY"
    | "QUESTION_ANSWERING"
    | "FACT_VERIFICATION";

// --- OpenAI-compatible request types ---

interface TextInput {
    type: "text";
    text: string;
}

interface ImageUrlInput {
    type: "image_url";
    image_url: { url: string };
}

interface AudioInput {
    type: "input_audio";
    input_audio: { data: string; format: string };
}

interface VideoUrlInput {
    type: "video_url";
    video_url: { url: string; mime_type?: string };
}

type ContentPart = TextInput | ImageUrlInput | AudioInput | VideoUrlInput;

interface EmbeddingRequest {
    model: string;
    input: string | string[] | ContentPart | ContentPart[];
    dimensions?: number;
    task_type?: GeminiTaskType;
}

// --- Gemini API types ---

interface GeminiPart {
    text?: string;
    inline_data?: { mime_type: string; data: string };
}

type GeminiModality = "TEXT" | "IMAGE" | "AUDIO" | "VIDEO";

interface ModalityTokenCount {
    modality?: GeminiModality;
    tokenCount?: number;
}

interface GeminiEmbedResponse {
    embedding: { values: number[] };
    usageMetadata?: {
        promptTokenCount?: number;
        totalTokenCount?: number;
        promptTokensDetails?: ModalityTokenCount[];
    };
}

const MODALITY_TO_USAGE_KEY: Record<GeminiModality, keyof Usage> = {
    TEXT: "promptTextTokens",
    IMAGE: "promptImageTokens",
    AUDIO: "promptAudioTokens",
    VIDEO: "promptVideoTokens",
};

/**
 * Split Gemini's per-modality token breakdown into our Usage shape.
 * Falls back to billing the full promptTokenCount as text if no modality
 * details are returned (older API versions, pure-text requests).
 */
function extractModalityUsage(result: GeminiEmbedResponse): Usage {
    const details = result.usageMetadata?.promptTokensDetails;
    if (details && details.length > 0) {
        const usage: Usage = {};
        for (const { modality, tokenCount } of details) {
            if (!modality || !tokenCount) continue;
            const key = MODALITY_TO_USAGE_KEY[modality];
            if (key) {
                usage[key] = (usage[key] ?? 0) + tokenCount;
            }
        }
        return usage;
    }
    return { promptTextTokens: result.usageMetadata?.promptTokenCount ?? 0 };
}

// --- Transform: OpenAI input → Gemini parts ---

async function inputToGeminiParts(
    input: string | ContentPart | ContentPart[],
): Promise<GeminiPart[]> {
    const parts: GeminiPart[] = [];

    if (typeof input === "string") {
        parts.push({ text: input });
        return parts;
    }

    const items = Array.isArray(input) ? input : [input];

    for (const part of items) {
        if (typeof part === "string") {
            parts.push({ text: part });
        } else if (part.type === "text") {
            parts.push({ text: part.text });
        } else if (part.type === "image_url") {
            const { url } = part.image_url;
            if (url.startsWith("data:")) {
                const { mimeType, data } = parseDataUrl(url, "Image");
                parts.push({ inline_data: { mime_type: mimeType, data } });
            } else {
                const { buffer, contentType } = await fetchMedia(url, "Image");
                const base64 = Buffer.from(buffer).toString("base64");
                parts.push({
                    inline_data: { mime_type: contentType, data: base64 },
                });
            }
        } else if (part.type === "input_audio") {
            assertBase64MediaSize("Audio", part.input_audio.data);
            const mimeType = `audio/${part.input_audio.format || "mp3"}`;
            parts.push({
                inline_data: {
                    mime_type: mimeType,
                    data: part.input_audio.data,
                },
            });
        } else if (part.type === "video_url") {
            const { url, mime_type } = part.video_url;
            if (url.startsWith("data:")) {
                const parsedVideo = parseDataUrl(url, "Video");
                parts.push({
                    inline_data: {
                        mime_type: mime_type || parsedVideo.mimeType,
                        data: parsedVideo.data,
                    },
                });
            } else {
                const { buffer, contentType } = await fetchMedia(url, "Video");
                const base64 = Buffer.from(buffer).toString("base64");
                parts.push({
                    inline_data: {
                        mime_type: mime_type || contentType,
                        data: base64,
                    },
                });
            }
        }
    }

    return parts;
}

// --- Call Gemini embedContent API via Vertex AI v1beta1 ---

async function callGeminiEmbed(
    modelId: string,
    parts: GeminiPart[],
    taskType?: GeminiTaskType,
    outputDimensionality?: number,
): Promise<GeminiEmbedResponse> {
    if (!GOOGLE_PROJECT_ID) {
        throw new Error("GOOGLE_PROJECT_ID not configured");
    }

    const accessToken = await googleCloudAuth.getAccessToken();
    if (!accessToken) {
        throw new Error(
            "Google Cloud authentication failed — missing or invalid credentials",
        );
    }

    const url = `https://${VERTEX_REGION}-aiplatform.googleapis.com/v1beta1/projects/${GOOGLE_PROJECT_ID}/locations/${VERTEX_REGION}/publishers/google/models/${modelId}:embedContent`;

    const body = {
        content: { parts },
        embedContentConfig: {
            ...(taskType && { taskType }),
            ...(outputDimensionality && { outputDimensionality }),
        },
    };

    const response = await fetch(url, {
        method: "POST",
        headers: {
            Authorization: `Bearer ${accessToken}`,
            "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(30_000),
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
            `Gemini embedding API error (${response.status}): ${errorText}`,
        );
    }

    return response.json() as Promise<GeminiEmbedResponse>;
}

// --- Normalize input to array of individual embedding inputs ---

function normalizeInputs(
    input: EmbeddingRequest["input"],
): (string | ContentPart[])[] {
    if (typeof input === "string") {
        return [input];
    }

    if (!Array.isArray(input)) {
        // Single ContentPart → wrap as single multimodal embedding
        return [[input]];
    }

    if (input.length === 0) {
        return [];
    }

    // Array of strings → each string is a separate embedding
    if (typeof input[0] === "string") {
        return input as string[];
    }

    // Array of ContentParts → single multimodal embedding
    return [input as ContentPart[]];
}

// --- Main handler ---

export async function generateEmbeddings(
    request: EmbeddingRequest,
): Promise<Response> {
    const { model, input, dimensions, task_type } = request;
    const modelId = model;

    const inputs = normalizeInputs(input);

    if (inputs.length === 0) {
        return new Response(
            JSON.stringify({
                object: "list",
                data: [],
                model: modelId,
                usage: { prompt_tokens: 0, total_tokens: 0 },
            }),
            { headers: { "Content-Type": "application/json" } },
        );
    }

    // Process in chunks to avoid saturating Vertex AI with concurrent requests
    const EMBED_CONCURRENCY = 10;
    const results: {
        object: "embedding";
        embedding: number[];
        index: number;
        usage: Usage;
    }[] = [];
    for (let i = 0; i < inputs.length; i += EMBED_CONCURRENCY) {
        const chunk = inputs.slice(i, i + EMBED_CONCURRENCY);
        const chunkResults = await Promise.all(
            chunk.map(async (singleInput, j) => {
                const parts = await inputToGeminiParts(singleInput);
                const result = await callGeminiEmbed(
                    modelId,
                    parts,
                    task_type,
                    dimensions,
                );
                return {
                    object: "embedding" as const,
                    embedding: result.embedding.values,
                    index: i + j,
                    usage: extractModalityUsage(result),
                };
            }),
        );
        results.push(...chunkResults);
    }

    const embeddings = results.map(({ object, embedding, index }) => ({
        object,
        embedding,
        index,
    }));

    const aggregatedUsage: Usage = {};
    for (const r of results) {
        for (const [key, value] of Object.entries(r.usage) as [
            keyof Usage,
            number | undefined,
        ][]) {
            if (value) {
                aggregatedUsage[key] = (aggregatedUsage[key] ?? 0) + value;
            }
        }
    }
    const promptTokens = Object.values(aggregatedUsage).reduce(
        (s, v) => s + (v ?? 0),
        0,
    );

    const usageHeaders = buildUsageHeaders(modelId, aggregatedUsage);

    const responseBody = {
        object: "list",
        data: embeddings,
        model: modelId,
        usage: {
            prompt_tokens: promptTokens,
            total_tokens: promptTokens,
        },
    };

    return new Response(JSON.stringify(responseBody), {
        headers: {
            "Content-Type": "application/json",
            ...usageHeaders,
        },
    });
}
