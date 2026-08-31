import { UploadSkeleton } from "@/components/Skeleton";

/**
 * Shown while the upload route is being rendered on the server.
 *
 * App Router serves this for both entrances: a hard refresh, and a client-side
 * navigation back from a review. The route pairs it with a floor in `pacing`, so
 * it is always on screen long enough to be read rather than seen.
 */
export default function Loading(): React.JSX.Element {
  return <UploadSkeleton />;
}
