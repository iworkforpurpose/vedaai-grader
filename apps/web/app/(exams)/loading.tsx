import { ShellSkeleton, UploadSkeleton } from "@/components/Skeleton";

/**
 * Shown while the upload route renders on the server.
 *
 * It lives in a route group rather than at `app/loading.tsx`, and that placement
 * is the whole point. A loading file at the root is the Suspense boundary for
 * every segment beneath it, so refreshing a review streamed this upload skeleton
 * first and the review skeleton second — a flash of the wrong layout at the exact
 * moment the skeleton exists to stop one. The group scopes it to the screen it
 * describes without changing the URL.
 */
export default function Loading(): React.JSX.Element {
  return (
    <ShellSkeleton>
      <UploadSkeleton />
    </ShellSkeleton>
  );
}
