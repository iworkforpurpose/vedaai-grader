"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

/**
 * The access code prompt.
 *
 * One field, because there is one credential. No mention of what is behind the
 * door and no distinction between a wrong code and a missing one — both are
 * "that code was not accepted", since telling a stranger which half they got
 * right is the only thing a more helpful message would achieve.
 */

/** Where to land after unlocking, with anything that could send you elsewhere removed. */
function safeDestination(next: string | undefined): string {
  // A path, not a URL. Without this, `?next=https://elsewhere.example` turns the
  // unlock form into an open redirect that borrows this origin's credibility.
  if (!next || !next.startsWith("/") || next.startsWith("//")) return "/";
  return next;
}

export function UnlockForm({ next }: { next?: string }): React.JSX.Element {
  const [code, setCode] = useState("");
  const [refused, setRefused] = useState(false);
  const [checking, setChecking] = useState(false);
  const router = useRouter();

  async function submit(event: React.FormEvent): Promise<void> {
    event.preventDefault();
    if (!code.trim() || checking) return;

    setChecking(true);
    setRefused(false);
    try {
      const response = await fetch("/access", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      if (!response.ok) {
        setRefused(true);
        setCode("");
        return;
      }
      // Replace rather than push, so the back button does not return to a form
      // that is now pointless.
      router.replace(safeDestination(next));
      router.refresh();
    } catch {
      setRefused(true);
    } finally {
      setChecking(false);
    }
  }

  return (
    <main className="unlock">
      <form className="unlock-card" onSubmit={submit}>
        <h1 className="unlock-title">Access code</h1>
        <p className="unlock-note">This service is not open to the public.</p>

        <input
          className="unlock-input"
          type="password"
          value={code}
          onChange={(event) => setCode(event.target.value)}
          placeholder="Enter code"
          aria-label="Access code"
          aria-invalid={refused}
          autoFocus
          autoComplete="off"
        />

        {/*
          Announced politely rather than assertively: the message replaces itself
          on each attempt, and a screen reader interrupting mid-word to repeat it
          is worse than hearing it a moment later.
        */}
        <p className="unlock-error" role="status" aria-live="polite">
          {refused ? "That code was not accepted." : " "}
        </p>

        <button className="cta unlock-submit" type="submit" disabled={!code.trim() || checking}>
          {checking ? "Checking…" : "Continue"}
        </button>
      </form>
    </main>
  );
}
