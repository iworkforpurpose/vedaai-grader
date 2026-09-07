import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * Tells the API who a request came from.
 *
 * The worker listens on loopback inside this container, so every request reaches
 * it from the proxy and the peer address is always identical. Without the real
 * caller forwarded explicitly, one person's retry loop would exhaust the rate
 * limit for everybody.
 *
 * The header can be trusted because it is overwritten here on every request,
 * so nothing outside this container can set it.
 */
export function middleware(request: NextRequest): NextResponse {
  const client =
    request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "unknown";

  const headers = new Headers(request.headers);
  headers.set("x-client-key", client);

  return NextResponse.next({ request: { headers } });
}

export const config = {
  matcher: ["/((?!_next/static|_next/image).*)"],
};
