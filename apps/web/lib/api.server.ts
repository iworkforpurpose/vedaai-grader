import "server-only";

import type { Health } from "./api";

/**
 * The base for fetches made on the server, and the helpers that use it.
 *
 * Split out of `lib/api.ts` because that module is imported by client components,
 * and a `NEXT_PUBLIC_*`-adjacent default in a client-imported module ends up in
 * the browser's JavaScript whether or not anything reads it. The loopback literal
 * below was in two client chunks for exactly that reason.
 *
 * That mattered more than tidiness. It made a real fault undetectable: a stray
 * `apps/web/.env.local` in the build context inlined
 * `NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000`, the deployed page told the browser
 * to fetch from its own loopback, and every upload failed with "cannot reach the
 * grader service". A build-time check for loopback URLs in the client bundle is
 * the obvious guard, and it could not work while this file put one there
 * legitimately.
 *
 * `server-only` makes an accidental client import a build error rather than a
 * silent regression.
 */

/**
 * Loopback rather than the public hostname on purpose: a server component
 * reaching its own load balancer to talk to a process in the same container would
 * depend on DNS, TLS and the balancer's health — three things that can fail while
 * the worker beside it is perfectly fine.
 */
export const INTERNAL_API_BASE =
  process.env.INTERNAL_API_BASE ?? "http://127.0.0.1:8000";

export class ApiUnavailableError extends Error {
  constructor(cause: unknown) {
    super(
      `Cannot reach the grader API at ${INTERNAL_API_BASE}. ` +
        "Start it with `pnpm --filter @vedaai/api dev`.",
    );
    this.name = "ApiUnavailableError";
    this.cause = cause;
  }
}

export async function fetchHealth(): Promise<Health> {
  let response: Response;
  try {
    response = await fetch(`${INTERNAL_API_BASE}/health`, { cache: "no-store" });
  } catch (cause) {
    throw new ApiUnavailableError(cause);
  }
  if (!response.ok) {
    throw new Error(`Health check failed with HTTP ${response.status}`);
  }
  return (await response.json()) as Health;
}
