import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * One passcode in front of the whole origin.
 *
 * The service answers a public URL and asks nobody who they are. Every page,
 * every submission and every stored script is readable by anyone who finds the
 * address — and a submission is a real student's handwriting, which is not the
 * kind of thing to leave open while a link is passed around.
 *
 * The gate sits here rather than in the API because Next owns the origin and
 * rewrites `/api/*` to the worker on loopback, so middleware is the single place
 * both are behind. Putting it in the API would leave the pages open; putting it
 * in a layout would leave the API open.
 *
 * What it is: a shared passcode, appropriate to a link shared with a handful of
 * people who are helping test. What it is not: accounts, or any way of telling
 * those people apart. When submissions need to belong to somebody, this is the
 * seam that grows.
 *
 * Set `ACCESS_CODE` to switch it on. Unset, the origin is open — which is right
 * for a laptop and wrong for anything with a public address, so the deployment
 * always sets it.
 */

const COOKIE = "vedaai_access";

/**
 * Paths that have to work before anyone can unlock anything.
 *
 * `/brand` is here for a reason worth remembering. Those files are a logo, a
 * crest and an avatar — public brand furniture, nothing anybody needs a password
 * to see — but that is not why gating them broke the page.
 *
 * Next's image optimizer fetches the source itself, server-side, from its own
 * origin, and that request carries no cookie. Behind the gate it got a redirect
 * to the unlock screen, could not read an image out of it, and answered 400. So
 * every `<Image>` on the site went blank while the answer-sheet pages — plain
 * `<img>` fetched by the browser, which does send the cookie — kept working. The
 * symptom pointed at image handling and the cause was authentication.
 */
const ALWAYS_OPEN = [
  "/unlock",
  "/access",
  "/api/health",
  "/_next",
  "/brand",
  "/favicon.ico",
  "/icon.png",
];

/**
 * The cookie value that proves knowledge of the passcode.
 *
 * An HMAC of a fixed string keyed by the code, rather than the code itself: a
 * cookie is readable by anything with the browser open, and one holding the
 * passcode verbatim hands it to whatever reads it. Keying by the code also means
 * changing the code invalidates every session, which is what makes rotating it
 * an actual revocation rather than a suggestion.
 */
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

/**
 * Comparison that does not finish early on the first wrong character.
 *
 * The timing of a short-circuiting compare leaks how much of a secret is right,
 * which is enough to recover it one character at a time. The cost of avoiding
 * that is nothing.
 */
function sameSecret(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let difference = 0;
  for (let i = 0; i < a.length; i += 1) {
    difference |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return difference === 0;
}

export async function middleware(request: NextRequest) {
  const code = process.env.ACCESS_CODE?.trim();

  /*
   * Who the API should count a request against.
   *
   * The worker sits on loopback inside this container, so every request reaches
   * it from the proxy and the peer address is always the same. Passing the real
   * caller explicitly is what keeps one person's loop from being everybody's
   * refusal — and the header is trustworthy precisely because nothing outside
   * this container can set it: it is overwritten here on every request.
   */
  const client =
    request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "unknown";
  const headers = new Headers(request.headers);
  headers.set("x-client-key", client);

  const onward = () => NextResponse.next({ request: { headers } });

  if (!code) return onward();

  const path = request.nextUrl.pathname;
  if (ALWAYS_OPEN.some((open) => path === open || path.startsWith(`${open}/`))) {
    return onward();
  }

  const presented = request.cookies.get(COOKIE)?.value ?? "";
  if (presented && sameSecret(presented, await tokenFor(code))) return onward();

  /*
   * An unauthenticated API call gets a status, not a redirect. A fetch following
   * a 307 to an HTML page reports "unexpected token < in JSON", which sends
   * whoever debugs it looking at the parser rather than at the session that
   * quietly expired.
   */
  if (path.startsWith("/api/")) {
    return NextResponse.json(
      { detail: "This service is not open. Enter the access code first." },
      { status: 401 },
    );
  }

  const unlock = request.nextUrl.clone();
  unlock.pathname = "/unlock";
  // Carried so unlocking returns to the page that was asked for, which matters
  // for a shared link to one submission.
  unlock.searchParams.set("next", `${path}${request.nextUrl.search}`);
  return NextResponse.redirect(unlock);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image).*)"],
};
