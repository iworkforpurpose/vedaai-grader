import { ExamsScreen } from "@/components/ExamsScreen";
import { fetchHealth } from "@/lib/api.server";

/**
 * The Exams tab — the only reachable screen, as scoped.
 *
 * The service check that used to sit here has moved to the geometry inspector.
 * It compares the API's render DPI against the frontend's assumption, which is
 * worth doing — a mismatch silently offsets every highlight rather than throwing
 * — but it is a diagnostic, and the design has no panel for one.
 */

export const dynamic = "force-dynamic";

export default async function ExamsPage(): Promise<React.JSX.Element> {
  // The upload limit is the service's to state, so it is read from the service.
  //
  // Swallowed on failure on purpose: an upload screen that will not render
  // because a health check timed out is worse than one whose hint omits a size,
  // and the upload itself does not depend on this answer.
  let maxUploadBytes: number | undefined;
  try {
    maxUploadBytes = (await fetchHealth()).max_upload_bytes;
  } catch {
    maxUploadBytes = undefined;
  }

  return <ExamsScreen maxUploadBytes={maxUploadBytes} />;
}
