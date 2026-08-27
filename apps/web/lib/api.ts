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

/** Browser-facing base. Relative, so it follows whatever origin served the page. */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

/**
 * Base for fetches made on the server.
 *
 * Loopback rather than the public hostname on purpose: a server component
 * reaching its own load balancer to talk to a process in the same container
 * would depend on DNS, TLS and the balancer's health — three things that can
 * fail while the worker beside it is perfectly fine.
 */
export const INTERNAL_API_BASE =
  process.env.INTERNAL_API_BASE ?? "http://127.0.0.1:8000";

/** The base to use from wherever this is running. */
export function apiBase(): string {
  return typeof window === "undefined" ? INTERNAL_API_BASE : API_BASE;
}

export interface Health {
  status: string;
  version: string;
  /** Render DPI the geometry is relative to. Must match the frontend's assumption. */
  render_dpi: number;
  hgbench_scale: number;
  contract_model_count: number;
}

/** The DPI the frontend expects. Cross-checked against the API at runtime.
 *
 * Normalized coordinates mean the browser never divides by DPI, so a mismatch
 * would not throw — it would silently shift every highlight. Comparing the two
 * values turns that into something visible.
 */
export const EXPECTED_RENDER_DPI = 200;

export class ApiUnavailableError extends Error {
  constructor(cause: unknown) {
    super(
      `Cannot reach the grader API at ${apiBase()}. Start it with \`pnpm --filter @vedaai/api dev\`.`,
    );
    this.name = "ApiUnavailableError";
    this.cause = cause;
  }
}

export async function fetchHealth(): Promise<Health> {
  let response: Response;
  try {
    response = await fetch(`${apiBase()}/health`, { cache: "no-store" });
  } catch (cause) {
    throw new ApiUnavailableError(cause);
  }
  if (!response.ok) {
    throw new Error(`Health check failed with HTTP ${response.status}`);
  }
  return (await response.json()) as Health;
}
