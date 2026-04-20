import path from "node:path";
import {
    defineWorkersConfig,
    readD1Migrations,
} from "@cloudflare/vitest-pool-workers/config";
import { loadEnv } from "vite";
import viteConfig from "./vite.config";

export default defineWorkersConfig(async ({ mode }) => {
    const migrationsPath = path.join(__dirname, "drizzle");
    const migrations = await readD1Migrations(migrationsPath);
    const env = loadEnv(mode, process.cwd(), "");

    return {
        ...viteConfig,
        test: {
            globalSetup: ["./test/setup/snapshot-server.ts"],
            setupFiles: [
                "./test/setup/apply-migrations.ts",
                "./test/setup/rejection-handler.ts",
            ],
            reporters: ["default"],
            teardownTimeout: 5000,
            poolOptions: {
                workers: {
                    singleWorker: true,
                    wrangler: {
                        configPath: "./wrangler.toml",
                        environment: env.TEST_ENV || "test",
                    },
                    miniflare: {
                        bindings: {
                            TEST_MIGRATIONS: migrations,
                            TEST_VCR_MODE:
                                env.TEST_VCR_MODE || "replay-or-record",
                            ...(env.TEXT_SERVICE_URL
                                ? { TEXT_SERVICE_URL: env.TEXT_SERVICE_URL }
                                : {}),
                            ...(env.IMAGE_SERVICE_URL
                                ? { IMAGE_SERVICE_URL: env.IMAGE_SERVICE_URL }
                                : {}),
                        },
                    },
                },
            },
            deps: {
                optimizer: {
                    ssr: {
                        enabled: true,
                        include: [
                            "@polar-sh/sdk",
                            "better-auth",
                            "kysely",
                            "drizzle-orm",
                            "hono-openapi",
                        ],
                    },
                },
            },
        },
    };
});
