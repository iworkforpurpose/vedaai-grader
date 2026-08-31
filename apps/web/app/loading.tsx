import { AppShell } from "@/components/AppShell";
import { UploadSkeleton } from "@/components/Skeleton";

/**
 * Shown while the upload route renders on the server.
 *
 * The shell is the real one. The rail and the top bar do not depend on anything
 * being fetched, so drawing placeholder versions of them would invent a second
 * layout to keep in step with the first — and the seam between the two is exactly
 * the misalignment that makes a skeleton look fake. Only the content is blocks.
 */
export default function Loading(): React.JSX.Element {
  return (
    <AppShell crumb="Exams">
      <UploadSkeleton />
    </AppShell>
  );
}
