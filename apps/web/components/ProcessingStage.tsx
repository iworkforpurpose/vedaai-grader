"use client";

import { LoadingStage } from "./LoadingStage";
import { PIPELINE_PHASES } from "./LoadingPhases";
import { progressCaption, useProgress } from "@/lib/progress";

/**
 * The waiting screen for a submission the service is still working on.
 *
 * Prefers what the worker says about itself. The timed captions are still here
 * and still correct in their ordering, but they are a guess at the position, and
 * a guess is only worth showing while there is nothing better — before the first
 * event arrives, or if the stream cannot be opened at all.
 *
 * Switching between the two mid-wait is deliberate rather than a compromise. The
 * first event usually lands within a moment of the screen appearing, so what a
 * reader sees is the timed line briefly and then the real one, which is the right
 * way round: the fallback covers the gap instead of replacing the truth.
 */
export function ProcessingStage({
  submissionId,
}: {
  submissionId: string;
}): React.JSX.Element {
  const progress = useProgress(submissionId);
  const caption = progressCaption(progress);

  if (caption) return <LoadingStage detail={caption} />;
  return <LoadingStage phases={PIPELINE_PHASES} />;
}
