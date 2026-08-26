import Link from "next/link";
import { DebugReview } from "@/components/DebugReview";
import { API_BASE } from "@/lib/api";
import type { LineIndex, Submission } from "@/lib/contracts";

/**
 * Geometry inspection surface for one submission.
 *
 * Phase 1's deliverable. The product's review surface replaces this in Phase 7,
 * but the overlay stays reachable: when a highlight lands in the wrong place,
 * the first question is always whether the underlying boxes were right, and
 * this is what answers it.
 */

export const dynamic = "force-dynamic";

async function getJson<T>(path: string): Promise<T | null> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  // A 404 on a line index is expected, not exceptional: a document without a
  // configured transcription engine simply has none.
  if (!response.ok) return null;
  return (await response.json()) as T;
}

export default async function ReviewPage({
  params,
}: {
  params: Promise<{ id: string }>;
}): Promise<React.JSX.Element> {
  const { id } = await params;

  const submission = await getJson<Submission>(`/submissions/${id}`);
  if (submission === null) {
    return (
      <main style={{ maxWidth: 720, margin: "0 auto", padding: "var(--sp-7) var(--sp-5)" }}>
        <h1 style={{ fontSize: "var(--fs-xl)" }}>Submission not found</h1>
        <p style={{ color: "var(--text-2)" }}>
          Submissions are held in memory, so restarting the pipeline service clears them.
          Re-upload to start again.
        </p>
        <Link href="/">Back to upload</Link>
      </main>
    );
  }

  const [questionLines, answerLines] = await Promise.all([
    getJson<LineIndex>(`/submissions/${id}/lines/question_paper`),
    getJson<LineIndex>(`/submissions/${id}/lines/answer_sheet`),
  ]);

  return (
    <main
      style={{
        maxWidth: 1100,
        margin: "0 auto",
        padding: "var(--sp-6) var(--sp-5) var(--sp-7)",
        display: "flex",
        flexDirection: "column",
        gap: "var(--sp-5)",
      }}
    >
      <header style={{ display: "flex", flexDirection: "column", gap: "var(--sp-1)" }}>
        <p
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "var(--fs-xs)",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            color: "var(--text-muted)",
            margin: 0,
          }}
        >
          <Link href="/">Upload</Link> · geometry inspector · {submission.submission_id}
        </p>
        <h1 style={{ fontSize: "var(--fs-xl)", margin: 0, letterSpacing: "-0.02em" }}>
          {submission.question_paper_file?.filename ?? "Question paper"}
          {" · "}
          {submission.answer_sheet_file?.filename ?? "Answer sheet"}
        </h1>
        <p style={{ margin: 0, color: "var(--text-2)" }}>
          Every box below came from the transcription layer. If they sit correctly on the words,
          highlight geometry is sound and any later mis-highlight is a mapping fault.
        </p>
      </header>

      <DebugReview
        submission={submission}
        questionLines={questionLines}
        answerLines={answerLines}
      />
    </main>
  );
}
