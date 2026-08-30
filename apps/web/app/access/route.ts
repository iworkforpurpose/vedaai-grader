import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * Exchange the access code for a session cookie.
 *
 * Lives at `/access` rather than under `/api`, because everything under `/api`
 * is rewritten to the grading worker on loopback. Next does check filesystem
 * routes before those rewrites, so a handler there would in fact win — but it
 * would win by a rule nobody reading either file would notice, and the day the
 * rewrite changes shape it stops winning silently.
 */

const COOKIE = "vedaai_access";

/** A week. Long enough not to nag a tester, short enough to expire if a link leaks. */
const LIFETIME = 7 * 24 * 60 * 60;

async function tokenFor(code: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(code),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const signature = await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode("vedaai-grader-access-v1"),
  );
  return Array.from(new Uint8Array(signature))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function sameSecret(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let difference = 0;
  for (let i = 0; i < a.length; i += 1) {
    difference |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return difference === 0;
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  const expected = process.env.ACCESS_CODE?.trim();
  if (!expected) {
    // No code configured means the origin is open and there is nothing to
    // exchange. Saying so beats setting a cookie that proves nothing.
    return NextResponse.json({ detail: "No access code is configured." }, { status: 404 });
  }

  let offered = "";
  try {
    offered = String(((await request.json()) as { code?: unknown }).code ?? "");
  } catch {
    return NextResponse.json({ detail: "Send a code." }, { status: 400 });
  }

  if (!sameSecret(offered, expected)) {
    // No detail about which part was wrong, and no distinction from a missing
    // code. Anything more helpful is only helpful to somebody guessing.
    return NextResponse.json({ detail: "That code was not accepted." }, { status: 401 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set({
    name: COOKIE,
    value: await tokenFor(expected),
    httpOnly: true,
    sameSite: "lax",
    // Set over TLS in the deployment; left off on a laptop, where the cookie
    // would otherwise never be stored and the gate could not be tested at all.
    secure: request.nextUrl.protocol === "https:",
    path: "/",
    maxAge: LIFETIME,
  });
  return response;
}
