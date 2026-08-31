import { AppShell } from "@/components/AppShell";
import { ReviewSkeleton } from "@/components/Skeleton";

/**
 * Shown while a submission is fetched on the server.
 *
 * `collapsedRail` matches the route it becomes. Without it the rail would stand
 * full width and then snap to icons the moment the content arrived — an animation
 * on the one part of the screen that never changed, at the exact moment the
 * reader is deciding where to look.
 */
export default function Loading(): React.JSX.Element {
  return (
    <AppShell crumb="Exams" collapsedRail>
      <ReviewSkeleton />
    </AppShell>
  );
}
