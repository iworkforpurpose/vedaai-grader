"""Let the corpus scripts through the access gate.

The deployed origin is behind a shared passcode, so every script that talks to it
needs a session the same way a browser does. Rather than thread a header through
each call site, this exchanges the code once and installs an opener that carries
the cookie — after which plain `urllib.request.urlopen` works everywhere it
already did.

Reads `ACCESS_CODE` from the environment, and does nothing when it is unset,
which is the case on a laptop and the case for any deployment that has not
switched the gate on.
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import urllib.error
import urllib.request


def unlock(base: str) -> bool:
    """Open a session against `base`. True if one was needed and obtained."""
    code = os.getenv("ACCESS_CODE", "").strip()
    if not code:
        return False

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    # Installed globally so the scripts that already call `urlopen` directly pick
    # the session up without knowing this exists.
    urllib.request.install_opener(opener)

    request = urllib.request.Request(
        f"{base.rstrip('/')}/access",
        method="POST",
        data=json.dumps({"code": code}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener.open(request, timeout=30) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # No gate on this deployment. Nothing to unlock and nothing wrong.
            return False
        raise SystemExit(
            f"  the access code was refused by {base} ({exc.code}). "
            "Check ACCESS_CODE matches the deployment."
        ) from exc
    return True
