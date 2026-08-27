"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { API_BASE } from "@/lib/api";

/**
 * Upload both documents and go to the review surface.
 *
 * Files post directly to the pipeline service rather than through a Next route
 * handler. That is not incidental: a serverless function caps its request body
 * at 4.5 MB, and a scanned answer sheet routinely exceeds it. Posting straight
 * to the worker sidesteps the cap entirely.
 */
export function UploadForm(): React.JSX.Element {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    setBusy(true);

    const form = new FormData(event.currentTarget);
    try {
      const response = await fetch(`${API_BASE}/submissions`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        // The API's 422 detail says what was wrong with the file. Surfacing it
        // verbatim is more useful than a generic failure, because it is
        // actionable: wrong format, too many pages, empty file.
        const body = (await response.json().catch(() => null)) as { detail?: string } | null;
        setError(body?.detail ?? `Upload failed with HTTP ${response.status}`);
        return;
      }

      const submission = (await response.json()) as { submission_id: string };
      router.push(`/review/${submission.submission_id}`);
    } catch {
      setError(
        `Cannot reach the pipeline service at ${API_BASE}. Start it with \`pnpm --filter @vedaai/api dev\`.`,
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      onSubmit={onSubmit}
      className="upload-form"
      style={{ display: "flex", flexDirection: "column", gap: "var(--sp-4)" }}
    >
      <Field
        name="question_paper"
        label="Question paper"
        hint="PDF or images. Typed papers are read directly from the PDF, which is exact and needs no OCR."
      />
      <Field
        name="answer_sheet"
        label="Answer sheet"
        hint="One student's handwritten script, as a PDF or photos."
      />

      <div style={{ display: "flex", alignItems: "center", gap: "var(--sp-4)" }}>
        <button
          className="process-button"
          type="submit"
          disabled={busy}
          style={{
            fontFamily: "var(--font-ui)",
            fontSize: "var(--fs-base)",
            padding: "var(--sp-2) var(--sp-5)",
            borderRadius: "var(--radius)",
            border: "1px solid var(--accent)",
            background: busy ? "var(--surface-2)" : "var(--accent)",
            color: busy ? "var(--text-muted)" : "var(--accent-contrast)",
            cursor: busy ? "progress" : "pointer",
          }}
        >
          {busy ? "Processing…" : "Process"}
        </button>
        {busy && (
          <span style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
            Rendering and transcribing pages.
          </span>
        )}
      </div>

      {error !== null && (
        <p
          role="alert"
          style={{
            margin: 0,
            padding: "var(--sp-3) var(--sp-4)",
            borderRadius: "var(--radius)",
            border: "1px solid var(--border)",
            borderLeft: "3px solid var(--status-missing)",
            background: "var(--surface)",
            color: "var(--text)",
          }}
        >
          {error}
        </p>
      )}
    </form>
  );
}

function Field({
  name,
  label,
  hint,
}: {
  name: string;
  label: string;
  hint: string;
}): React.JSX.Element {
  // Associated by id rather than by nesting, and that is not a style preference.
  //
  // A file input inside its own label is activated twice by one click: once
  // directly, and again when the click reaches the label and the label forwards
  // activation back to its control. The picker opens, and a second picker opens
  // on top of it — so choosing a file appears to do nothing, and the dialogs
  // stack. Sixteen of them were queued on a single visit.
  //
  // Nesting is legal and works for a checkbox or a text field, where a second
  // activation is harmless. It is specifically the file input, whose activation
  // opens a modal dialog, that cannot survive it.
  const id = `upload-${name}`;
  return (
    <div className="upload-field" style={{ display: "flex", flexDirection: "column", gap: "var(--sp-1)" }}>
      <label htmlFor={id} style={{ fontWeight: 600 }}>
        {label}
      </label>
      <span id={`${id}-hint`} style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)" }}>
        {hint}
      </span>
      <input
        className="file-input"
        id={id}
        type="file"
        name={name}
        required
        aria-describedby={`${id}-hint`}
        accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.webp"
        style={{
          marginTop: "var(--sp-1)",
          padding: "var(--sp-2)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          background: "var(--surface)",
          fontSize: "var(--fs-sm)",
        }}
      />
    </div>
  );
}
