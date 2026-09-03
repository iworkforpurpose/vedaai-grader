import type { Notice } from "@/lib/review";

/**
 * What the pipeline knows went wrong, where the person reading the marks is.
 *
 * The API produces a careful vocabulary for its own limits and the review screen
 * rendered none of it. `submission.warnings` was read in two places: a debug
 * route nothing links to, and the failed-submission branch — which shows
 * `warnings[0]`, and only when the run failed outright. A run that completed
 * while silently degraded looked exactly like one that worked.
 *
 * Observed on a live run: four warnings on the payload, including "answers were
 * not marked: the provider is out of credit" and "answers were matched by wording
 * rather than by meaning", against a screen that said "3 of 6 answered · rubric
 * only" and offered no way to find out why.
 *
 * Placed above the question list rather than beside the questions it affects,
 * because these are facts about the whole run. A caveat attached to one question
 * reads as a fault in that answer.
 */
export function RunNotices({ notices }: { notices: readonly Notice[] }): React.JSX.Element | null {
  if (notices.length === 0) return null;

  const blocking = notices.some((n) => n.severity === "blocking");

  return (
    <div
      className="notices"
      /*
       * `alert` when a stage did not run, `status` otherwise.
       *
       * A blocking notice changes what the numbers below it mean — nothing was
       * marked, or nothing matched — so it interrupts. A degraded one is worth
       * hearing at the next pause rather than over whatever is being read.
       */
      role={blocking ? "alert" : "status"}
      aria-live={blocking ? "assertive" : "polite"}
    >
      {notices.map((notice, i) => (
        <p key={i} className="notice" data-severity={notice.severity}>
          <span className="notice-mark" aria-hidden="true">
            {notice.severity === "informational" ? "·" : "!"}
          </span>
          {notice.text}
        </p>
      ))}
    </div>
  );
}
