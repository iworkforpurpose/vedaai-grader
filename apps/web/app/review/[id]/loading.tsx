import { ReviewSkeleton, ShellSkeleton } from "@/components/Skeleton";

/**
 * Shown while a submission is fetched on the server.
 *
 * The rail is collapsed to match the route it becomes, or it would stand full
 * width and snap to icons the instant the content arrived.
 */
export default function Loading(): React.JSX.Element {
  return (
    <ShellSkeleton collapsed>
      <ReviewSkeleton />
    </ShellSkeleton>
  );
}
