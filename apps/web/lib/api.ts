/**
 * Client for the grader API.
 *
 * Page images and the progress stream are served by the FastAPI worker rather
 * than proxied through Next, and uploads go straight to object storage via a
 * presigned URL — a Vercel function caps its request body at 4.5 MB, which a
 * scanned answer sheet routinely exceeds.
 */

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

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
      `Cannot reach the grader API at ${API_BASE}. Start it with \`pnpm --filter @vedaai/api dev\`.`,
    );
    this.name = "ApiUnavailableError";
    this.cause = cause;
  }
}

export async function fetchHealth(): Promise<Health> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  } catch (cause) {
    throw new ApiUnavailableError(cause);
  }
  if (!response.ok) {
    throw new Error(`Health check failed with HTTP ${response.status}`);
  }
  return (await response.json()) as Health;
}
