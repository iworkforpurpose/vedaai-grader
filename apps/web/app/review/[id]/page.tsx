import { notFound } from "next/navigation";
import { AppShell } from "@/components/AppShell";
import { MapSurface } from "@/components/MapSurface";
import type { Submission } from "@/lib/contracts";
import { INTERNAL_API_BASE } from "@/lib/api";

/**
 * The mapping screen for one submission.
 *
 * Fetched on the server so the first paint already has the questions and the
 * mapping — a client fetch would show an empty split pane first, and the whole
 * point of the screen is what is in it.
 */

export const dynamic = "force-dynamic";

export default async function ReviewPage({
  params,
}: {
  params: Promise<{ id: string }>;
}): Promise<React.JSX.Element> {
  const { id } = await params;
  const response = await fetch(`${INTERNAL_API_BASE}/submissions/${id}`, {
    cache: "no-store",
  });
  if (response.status === 404) notFound();
  if (!response.ok) {
    throw new Error(`Could not load submission ${id}: HTTP ${response.status}`);
  }
  const submission = (await response.json()) as Submission;

  return (
    <AppShell crumb="Exams" collapsedRail>
      <MapSurface initial={submission} />
    </AppShell>
  );
}
