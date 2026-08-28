"use client";

import { useState } from "react";
import { AppShell } from "./AppShell";
import { UploadForm } from "./UploadForm";

/**
 * The Exams tab: upload, then the waiting screen while the pipeline runs.
 *
 * A thin client wrapper exists only to hold one boolean. The page itself stays a
 * server component, and the rail needs to collapse for the loading frame the way
 * it does for the mapping frame — which the server cannot know, because it
 * depends on a request that has not been made yet.
 */
export function ExamsScreen({
  maxUploadBytes,
}: {
  /** The service's upload limit, fetched by the page on the server. */
  maxUploadBytes?: number;
}): React.JSX.Element {
  const [working, setWorking] = useState(false);

  return (
    <AppShell crumb="Exams" collapsedRail={working}>
      <UploadForm onWorkingChange={setWorking} maxUploadBytes={maxUploadBytes} />
    </AppShell>
  );
}
