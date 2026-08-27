"use client";

import type { QuestionGrade, RubricPoint } from "@/lib/contracts";

/**
 * The marks proposed for one question, and the evidence behind each.
 *
 * Every rubric point that awards marks lists the lines that earned them, and
 * clicking a point highlights those lines on the sheet. That is the whole reason
 * grading is offered here at all: a mark a teacher can check in two seconds is
 * worth having, and one they would have to re-mark from scratch is not.
 *
 * Points with no citation are shown as unjudged rather than as failures. The
 * difference matters — "the student did not do this" and "nobody has decided yet"
 * are the same zero in a total and completely different to the person marking.
 */
export function GradePanel({
  grade,
  onCite,
  citedPointId,
}: {
  grade: QuestionGrade;
  onCite: (point: RubricPoint) => void;
  citedPointId: string | null;
}): React.JSX.Element {
  const judged = grade.rubric_points.some((point) => point.cited_line_ids.length > 0);

  return (
    <section
      style={{
        borderTop: "1px solid var(--border)",
        padding: "var(--sp-3) var(--sp-4)",
        background: "var(--surface-2)",
        fontSize: "var(--fs-sm)",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: "var(--sp-3)",
          marginBottom: "var(--sp-2)",
        }}
      >
        <strong>
          {judged ? (
            <>
              {grade.marks_awarded} / {grade.marks_available}
            </>
          ) : (
            <>Rubric · {grade.marks_available} marks</>
          )}
        </strong>
        {judged && grade.needs_review && (
          <span style={{ color: "var(--status-review)" }}>
            {grade.graded_on_partial_text
              ? "marked from text we could not read reliably"
              : "worth checking"}
          </span>
        )}
      </header>

      {!judged && (
          <p style={{ margin: "0 0 var(--sp-2)", color: "var(--text-muted)" }}>
            Nothing has been marked. Every point below is yours to decide.
          </p>
        )}

      <ol style={{ listStyle: "none", margin: 0, padding: 0 }}>
        {grade.rubric_points.map((point) => {
          const hasEvidence = point.cited_line_ids.length > 0;
          return (
            <li key={point.point_id} style={{ marginBottom: "var(--sp-2)" }}>
              <button
                type="button"
                onClick={() => onCite(point)}
                disabled={!hasEvidence}
                title={
                  hasEvidence
                    ? "Show the writing this mark rests on"
                    : "No line was cited for this point"
                }
                style={{
                  font: "inherit",
                  textAlign: "left",
                  width: "100%",
                  display: "flex",
                  gap: "var(--sp-2)",
                  alignItems: "baseline",
                  padding: "var(--sp-2)",
                  border: "1px solid",
                  borderColor:
                    citedPointId === point.point_id ? "var(--accent)" : "var(--border)",
                  borderRadius: "var(--radius-sm)",
                  background: "var(--surface)",
                  cursor: hasEvidence ? "pointer" : "default",
                  color: "inherit",
                }}
              >
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    minWidth: "3.5em",
                    color: point.marks_awarded > 0 ? "var(--status-ok)" : "var(--text-muted)",
                  }}
                >
                  {point.marks_awarded}/{point.marks_available}
                </span>
                <span style={{ flex: 1 }}>
                  {point.criterion}
                  {point.comment && (
                    <em
                      style={{
                        display: "block",
                        color: "var(--text-muted)",
                        fontStyle: "normal",
                        fontSize: "var(--fs-xs)",
                        marginTop: 2,
                      }}
                    >
                      {point.comment}
                    </em>
                  )}
                </span>
                {hasEvidence && (
                  <span style={{ color: "var(--accent)", fontSize: "var(--fs-xs)" }}>
                    show evidence
                  </span>
                )}
              </button>
            </li>
          );
        })}
      </ol>

      {grade.feedback && (
        <p style={{ margin: "var(--sp-2) 0 0", color: "var(--text-2)" }}>{grade.feedback}</p>
      )}
    </section>
  );
}
