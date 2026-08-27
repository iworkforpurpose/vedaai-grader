import type { NextConfig } from "next";

/**
 * Next serves the whole origin and proxies the API beside it.
 *
 * Both processes live in one container, so `/api/*` is rewritten to the FastAPI
 * worker on loopback. The browser therefore only ever talks to the origin it
 * loaded from, which means no CORS to configure and no second hostname to keep
 * in sync — and, because this proxy is a Node process rather than a serverless
 * function, none of the request-body caps that made uploading a scanned answer
 * sheet awkward.
 *
 * The rewrite streams, which the progress endpoint depends on: server-sent events
 * through a buffering proxy arrive all at once at the end, which is the same as
 * not having them.
 */
const INTERNAL_API_BASE = process.env.INTERNAL_API_BASE ?? "http://127.0.0.1:8000";

const config: NextConfig = {
  reactStrictMode: true,
  // Traces the minimal set of files the server actually needs, so the container
  // carries a runtime rather than a node_modules tree.
  output: "standalone",
  typescript: { ignoreBuildErrors: false },
  eslint: { ignoreDuringBuilds: true },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${INTERNAL_API_BASE}/:path*` }];
  },
};

export default config;
