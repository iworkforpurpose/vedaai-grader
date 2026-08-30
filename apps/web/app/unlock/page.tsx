import { UnlockForm } from "@/components/UnlockForm";

/**
 * Where the middleware sends anyone without a session.
 *
 * Deliberately says nothing about what is behind it. A page announcing whose
 * scripts are stored here, to somebody who cannot get in, is an invitation
 * rather than a door.
 */

export const dynamic = "force-dynamic";

export default async function UnlockPage({
  searchParams,
}: {
  searchParams: Promise<{ next?: string }>;
}): Promise<React.JSX.Element> {
  const { next } = await searchParams;
  return <UnlockForm next={next} />;
}
