import { ReviewSkeleton } from "@/components/Skeleton";

/**
 * Shown while a submission is being fetched on the server.
 *
 * This is the one that earns its keep: the review route fetches the whole
 * submission before it can render anything, and the screen it becomes is the
 * densest in the app. Arriving into the right shape matters more here than
 * anywhere else.
 */
export default function Loading(): React.JSX.Element {
  return <ReviewSkeleton />;
}
