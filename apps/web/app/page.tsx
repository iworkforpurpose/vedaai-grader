import { ExamsScreen } from "@/components/ExamsScreen";

/**
 * The Exams tab — the only reachable screen, as scoped.
 *
 * The service check that used to sit here has moved to the geometry inspector.
 * It compares the API's render DPI against the frontend's assumption, which is
 * worth doing — a mismatch silently offsets every highlight rather than throwing
 * — but it is a diagnostic, and the design has no panel for one.
 */

export const dynamic = "force-dynamic";

export default function ExamsPage(): React.JSX.Element {
  return <ExamsScreen />;
}
