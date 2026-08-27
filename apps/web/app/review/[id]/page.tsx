import Link from "next/link";
import { ReviewSurface } from "@/components/ReviewSurface";
import { INTERNAL_API_BASE } from "@/lib/api";
import type { Submission } from "@/lib/contracts";

/**
 * The teacher's review surface.
 *
 * Server-rendered so the first paint already carries the questions and their
 * statuses; interaction is handled by the client component below. The geometry
 * inspector lives at ./inspect.
 */

export const dynamic = "force-dynamic";

export default async function ReviewPage({
  params,
}: {
  params: Promise<{ id: string }>;
}): Promise<React.JSX.Element> {
  const { id } = await params;

  const response = await fetch(`${INTERNAL_API_BASE}/submissions/${id}`, { cache: "no-store" });
  if (!response.ok) {
    return (
      <main style={{ maxWidth: 640, margin: "0 auto", padding: "var(--sp-7) var(--sp-5)" }}>
        <h1 style={{ fontSize: "var(--fs-xl)" }}>Submission not found</h1>
        <p style={{ color: "var(--text-2)" }}>
          Submissions are held in memory, so restarting the pipeline service clears them.
          Upload the paper and answer sheet again to start over.
        </p>
        <Link href="/">Back to upload</Link>
      </main>
    );
  }

  const submission = (await response.json()) as Submission;
  return <ReviewSurface initial={submission} />;
}
