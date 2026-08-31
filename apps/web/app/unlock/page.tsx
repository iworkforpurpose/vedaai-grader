import type { Metadata } from "next";
import { UnlockForm } from "@/components/UnlockForm";

/**
 * Where the middleware sends anyone without a session.
 *
 * Deliberately says nothing about what is behind it. A page announcing whose
 * scripts are stored here, to somebody who cannot get in, is an invitation
 * rather than a door.
 */

/*
 * Its own metadata, because the root layout's was undoing the point of this page.
 *
 * That description names the whole product -- question papers, handwritten answer
 * sheets, what gets mapped to what -- and it was being served in the head of every
 * route, this one included. A door that says nothing in its body and everything in
 * its markup is not saying nothing.
 */
export const metadata: Metadata = {
  title: "Access code · VedaAI",
  description: "Enter an access code to continue.",
  robots: { index: false, follow: false },
};

export const dynamic = "force-dynamic";

export default async function UnlockPage({
  searchParams,
}: {
  searchParams: Promise<{ next?: string }>;
}): Promise<React.JSX.Element> {
  const { next } = await searchParams;
  return <UnlockForm next={next} />;
}
