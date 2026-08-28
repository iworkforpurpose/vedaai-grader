/**
 * Client for the grader API.
 *
 * Both processes run in one container behind one origin, and Next proxies
 * `/api/*` to the FastAPI worker on loopback. That removes CORS from the picture
 * entirely — the browser only ever talks to the origin it loaded from — and it
 * removes the request-body cap that would otherwise apply, because the proxy is a
 * Node process rather than a serverless function.
 *
 * Two bases, because the same code runs in two places and only one of them can
 * resolve a relative URL:
 *
 *   - the browser fetches `/api/...`, same origin, no host needed;
 *   - a server component runs inside the container, where a relative URL has no
 *     base at all, so it addresses the worker on loopback directly and skips its
 *     own proxy on the way.
 */

/**
 * Browser-facing base. Relative, so it follows whatever origin served the page.
 *
 * Deliberately the only base in this module. Its server-side counterpart lives in
 * `api.server.ts`, because anything here reaches the browser's bundle — including
 * a default nothing on the client ever reads.
 */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

export interface Health {
  status: string;
  version: string;
  /** Render DPI the geometry is relative to. Must match the frontend's assumption. */
  render_dpi: number;
  hgbench_scale: number;
  contract_model_count: number;
  /** The largest document the service will accept, in bytes. */
  max_upload_bytes: number;
  /** Whether a submission survives a restart of the service. */
  submissions_durable: boolean;
}

/** The DPI the frontend expects. Cross-checked against the API at runtime.
 *
 * Normalized coordinates mean the browser never divides by DPI, so a mismatch
 * would not throw — it would silently shift every highlight. Comparing the two
 * values turns that into something visible.
 */
export const EXPECTED_RENDER_DPI = 200;
