/**
 * Pure logic for the review surface.
 *
 * No React and no DOM, so it can be tested directly. Everything here answers a
 * question the interface has to get right regardless of how it looks, which is
 * also what makes the later visual pass a restyle rather than a rewrite.
 *
 * The status vocabulary is the part worth care. A teacher reads this list and
 * acts on it, and the four absence states mean genuinely different things: one
 * says the student left it blank, the others say we could not tell. Collapsing
 * them into "missing" would throw away the distinction the pipeline works hardest
 * to preserve.
 */

import { mergeOverlapping } from "./geometry";
import type {
  AnswerBlock,
  AnswerStatus,
  InkRegion,
  Mapping,
  MappingResult,
  PageBox,
  Question,
  QuestionGrade,
  Submission,
} from "./contracts";

export interface StatusPresentation {
  label: string;
  /** One line explaining what the teacher should do about it. */
  hint: string;
  colour: string;
  /** Whether this state asks for the teacher's attention. */
  needsAttention: boolean;
}

/**
 * How each answer status is shown.
 *
 * Only `unanswered` claims the student left something blank. The wording of the
 * others is deliberately non-committal — "could not read", "not found" — because
 * that is what they mean, and a teacher who reads "missing" will not go looking.
 */
export const STATUS: Record<AnswerStatus, StatusPresentation> = {
  answered: {
    label: "Answered",
    hint: "An answer was found and located on the sheet.",
    colour: "var(--status-answered)",
    needsAttention: false,
  },
  unanswered: {
    label: "Not answered",
    hint: "No writing was found for this question.",
    colour: "var(--status-unanswered)",
    needsAttention: false,
  },
  ocr_failed: {
    label: "Could not read",
    hint: "There is writing here, but it could not be transcribed. Read it yourself.",
    colour: "var(--status-review)",
    needsAttention: true,
  },
  not_required: {
    label: "Not required",
    hint: "The paper allowed this to be skipped, and it was.",
    colour: "var(--status-not-required)",
    needsAttention: false,
  },
  pages_missing: {
    label: "Page may be missing",
    hint: "An answer continues past the last uploaded page. Check for a missing page.",
    colour: "var(--status-missing)",
    needsAttention: true,
  },
  uncertain: {
    label: "Not found",
    hint: "Writing that may answer this is on the sheet, but could not be placed here.",
    colour: "var(--status-review)",
    needsAttention: true,
  },
};

export interface QuestionRow {
  question: Question;
  mapping: Mapping | undefined;
  status: AnswerStatus;
  presentation: StatusPresentation;
  hasHighlight: boolean;
}

/**
 * The question list, in printed order.
 *
 * Ordered by `print_order` rather than by label, because labels cannot order
 * anything reliably — they restart per section and mix romans with letters.
 */
export function buildRows(submission: Submission): QuestionRow[] {
  const mappings = new Map<string, Mapping>();
  for (const mapping of submission.mapping?.mappings ?? []) {
    mappings.set(mapping.qid, mapping);
  }

  const questions = [...(submission.questions?.questions ?? [])].sort(
    (a, b) => a.print_order - b.print_order,
  );

  return questions.map((question) => {
    const mapping = mappings.get(question.qid);
    const status: AnswerStatus = mapping?.status ?? "unanswered";
    return {
      question,
      mapping,
      status,
      presentation: STATUS[status],
      hasHighlight: (mapping?.highlight?.boxes.length ?? 0) > 0,
    };
  });
}

export interface ReviewSummary {
  total: number;
  answered: number;
  notAnswered: number;
  needsAttention: number;
  /** True when unplaced writing means no absence can be claimed. */
  absenceSuppressed: boolean;
  orphanCount: number;
}

/**
 * The headline a teacher reads first.
 *
 * "Which questions were left unanswered" is the product's stated purpose, so the
 * count is prominent — but it is suppressed entirely when unplaced writing means
 * the number would be a guess. A wrong count here is worse than no count.
 */
export function summarize(submission: Submission, rows: QuestionRow[]): ReviewSummary {
  const result: MappingResult | null = submission.mapping;
  // Stems are excluded from every count. "6 of 22 answered" is wrong if some of
  // those 22 are headings that could not be answered — the denominator has to be
  // the number of questions the paper actually asked.
  const asked = rows.filter((r) => !r.question.is_stem);
  return {
    total: asked.length,
    answered: asked.filter((r) => r.status === "answered" || r.status === "ocr_failed").length,
    notAnswered: asked.filter((r) => r.status === "unanswered").length,
    needsAttention: asked.filter((r) => r.presentation.needsAttention).length,
    absenceSuppressed: result?.absence_claims_suppressed ?? false,
    orphanCount: result?.orphans.length ?? 0,
  };
}

/** Boxes for one question's answer, grouped by the page they sit on. */
export function highlightByPage(mapping: Mapping | undefined): Map<number, PageBox[]> {
  const out = new Map<number, PageBox[]>();
  // Merged before grouping, so a band nested inside another is drawn once. A
  // region marked twice reads as two claims about the same writing, and a teacher
  // cannot tell which rectangle is the one the mapper means.
  for (const box of mergeOverlapping(mapping?.highlight?.boxes ?? [])) {
    const bucket = out.get(box.page);
    if (bucket) bucket.push(box);
    else out.set(box.page, [box]);
  }
  return out;
}

/**
 * Where the unplaced writing is, per page.
 *
 * Orphans are the case the brief asks about directly — writing that matches no
 * question — and they matter more than they look. An orphan usually means one of
 * two things, and the second is the important one: either the student answered
 * something the paper does not contain, or our own extraction missed a question.
 * Either way it is writing a teacher must be able to see and place by hand.
 */
export function orphanHighlightByPage(
  // The whole result, not one question's mapping: orphans belong to nothing, which
  // is what makes them orphans.
  mapping: MappingResult | undefined,
): Map<number, PageBox[]> {
  const out = new Map<number, PageBox[]>();
  for (const orphan of mapping?.orphans ?? []) {
    for (const box of orphan.highlight?.boxes ?? []) {
      const bucket = out.get(box.page);
      if (bucket) bucket.push(box);
      else out.set(box.page, [box]);
    }
  }
  return out;
}

/**
 * The blocks a teacher could move away from a question, newest arrangement first.
 *
 * Separate from `blocksOf` because the question being *corrected* needs its blocks
 * individually addressable — a wrong mapping is often one block of several, and
 * moving the whole answer would trade one error for another.
 */
export function movableBlocks(
  submission: Submission,
  qid: string,
): AnswerBlock[] {
  const mapping = submission.mapping?.mappings.find((m) => m.qid === qid);
  return blocksOf(submission, mapping);
}

/**
 * Whether a question's mapping was set by hand rather than by the aligner.
 *
 * Shown, because a teacher returning to a script needs to know which decisions
 * were theirs. Re-running the marking will not undo an override, and it would be
 * unreasonable to expect anyone to remember what they moved.
 */
export function isTeacherPlaced(submission: Submission, qid: string): boolean {
  return submission.mapping?.mappings.find((m) => m.qid === qid)?.teacher_override ?? false;
}

/**
 * Which question a point on the sheet belongs to.
 *
 * Reverse lookup: click a region and jump to its question. The smallest
 * containing box wins, so a tightly-bounded answer beats a large one that merely
 * encloses the same point — otherwise a page-spanning answer would swallow every
 * click on the pages it covers.
 */
export function questionAtPoint(
  rows: QuestionRow[],
  page: number,
  x: number,
  y: number,
): QuestionRow | null {
  let best: QuestionRow | null = null;
  let bestArea = Infinity;

  for (const row of rows) {
    for (const pageBox of row.mapping?.highlight?.boxes ?? []) {
      if (pageBox.page !== page) continue;
      const b = pageBox.box;
      if (x < b.x0 || x > b.x1 || y < b.y0 || y > b.y1) continue;
      const area = (b.x1 - b.x0) * (b.y1 - b.y0);
      if (area < bestArea) {
        bestArea = area;
        best = row;
      }
    }
  }
  return best;
}

/** Ink regions the recognizer never accounted for, grouped by page.
 *
 * Shown as a distinct layer because these are the regions most likely to hold an
 * answer the system failed to place — the first place a teacher should look when
 * a question reads "not found".
 */
export function untranscribedInkByPage(submission: Submission): Map<number, InkRegion[]> {
  const out = new Map<number, InkRegion[]>();
  for (const region of submission.ink_regions) {
    if (!region.is_orphan_ink) continue;
    const bucket = out.get(region.page);
    if (bucket) bucket.push(region);
    else out.set(region.page, [region]);
  }
  return out;
}

/**
 * Apply a teacher's reassignment locally.
 *
 * Returns a new submission with the block moved from whichever question held it
 * to the target. Local so the interface responds immediately; the server is told
 * separately, and its reply is authoritative.
 *
 * Reassignment is core rather than a nicety. Gradescope — the established tool
 * for this exact task — does not place answer regions automatically at all: it
 * has students mark their own, or requires a pre-printed template, and gives
 * instructors an explicit tool to correct regions. Manual correction is the norm
 * in this problem, not an admission of failure.
 */
export function applyReassignment(
  submission: Submission,
  blockId: string,
  toQid: string,
): Submission {
  if (!submission.mapping) return submission;

  const position = new Map(submission.blocks.map((b, i) => [b.block_id, i]));
  const byId = new Map(submission.blocks.map((b) => [b.block_id, b]));

  const mappings = submission.mapping.mappings.map((mapping) => {
    if (mapping.qid === toQid) {
      // Added to what the question already holds, mirroring the server. Replacing
      // would make a split answer unrepairable: restoring one block would
      // displace the other, indefinitely.
      const merged = [...new Set([...mapping.block_ids, blockId])].sort(
        (a, b) => (position.get(a) ?? 0) - (position.get(b) ?? 0),
      );
      const owned = merged.map((id) => byId.get(id)).filter((b) => b !== undefined);
      const boxes = owned.flatMap((b) => b.geometry);
      const pages = [...new Set(boxes.map((box) => box.page))].sort((a, b) => a - b);
      return {
        ...mapping,
        status: "answered" as AnswerStatus,
        block_ids: merged,
        highlight: boxes.length
          ? {
              boxes,
              derived_from: "ocr_lines",
              pages,
              spans_pages: pages.length > 1,
            }
          : mapping.highlight,
        teacher_override: true,
      };
    }
    if (mapping.block_ids.includes(blockId)) {
      const remaining = mapping.block_ids.filter((id) => id !== blockId);
      return {
        ...mapping,
        block_ids: remaining,
        // Losing its only block does not make a question blank — the teacher
        // moved the answer elsewhere, which says nothing about whether this
        // question was attempted.
        status: remaining.length ? mapping.status : ("uncertain" as AnswerStatus),
        highlight: remaining.length ? mapping.highlight : null,
      };
    }
    return mapping;
  });

  // Orphans are recomputed rather than filtered, matching the server. A block
  // displaced from the target question must reappear here, or the teacher loses
  // sight of real writing whenever the request fails and this state is all
  // they have.
  const owned = new Set(mappings.flatMap((m) => m.block_ids));
  const orphans = submission.blocks
    .filter((b) => !owned.has(b.block_id))
    .map((b) => {
      const existing = submission.mapping?.orphans.find((o) => o.block_id === b.block_id);
      return (
        existing ?? {
          block_id: b.block_id,
          text_preview: b.text.slice(0, 160),
          // The aligner's guess is not recomputed client-side; the server's
          // response carries it a moment later.
          best_guess_qid: null,
          best_guess_score: null,
          highlight: {
            boxes: b.geometry,
            derived_from: "ocr_lines" as const,
            pages: b.pages_spanned,
            spans_pages: b.pages_spanned.length > 1,
          },
        }
      );
    });

  return {
    ...submission,
    mapping: {
      ...submission.mapping,
      mappings,
      orphans,
    },
  };
}


/**
 * The blocks of writing a question currently owns, in document order.
 *
 * Needed because reassignment moves a *block*, not a question: the teacher's
 * commonest correction is "this writing belongs to a different question", and
 * naming which piece of writing moves requires knowing what the question holds.
 */
export function blocksOf(submission: Submission, mapping: Mapping | undefined): AnswerBlock[] {
  if (!mapping) return [];
  const byId = new Map(submission.blocks.map((block) => [block.block_id, block]));
  return mapping.block_ids
    .map((id) => byId.get(id))
    .filter((block): block is AnswerBlock => block !== undefined);
}

/** A short, human-readable stand-in for a block of writing. */
export function blockPreview(block: AnswerBlock): string {
  const text = block.text.trim().replace(/\s+/g, " ");
  if (text) return text.length > 60 ? `${text.slice(0, 60)}…` : text;
  // A block with no readable text is still real writing — a diagram, or
  // handwriting the recognizer failed on. Name it by where it is.
  const page = block.pages_spanned[0];
  return page === undefined ? "writing with no readable text" : `writing on page ${page + 1}`;
}


/** The grade for one question, if the submission has been marked. */
export function gradeFor(submission: Submission, qid: string): QuestionGrade | undefined {
  return submission.grades?.grades.find((grade) => grade.qid === qid);
}

/**
 * Where the lines a rubric point cites actually are.
 *
 * This is what makes a mark checkable rather than merely plausible. The model
 * named lines and never coordinates; resolving those names here is what turns a
 * citation into something the teacher can look at, and a citation that cannot be
 * resolved shows as nothing rather than as a rectangle over the wrong place.
 */
export function citationHighlight(
  submission: Submission,
  lineIds: readonly string[],
): Map<number, PageBox[]> {
  const wanted = new Set(lineIds);
  const grouped = new Map<number, PageBox[]>();

  const cited: PageBox[] = [];
  for (const line of submission.answer_sheet_lines?.lines ?? []) {
    if (!wanted.has(line.line_id)) continue;
    cited.push({ page: line.page, box: line.box });
  }

  // Cited lines are consecutive far more often than not, and one box per line is
  // the shape the mapper rejected as unreadable. Same merge, same reason.
  for (const box of mergeOverlapping(cited)) {
    const existing = grouped.get(box.page);
    if (existing) existing.push(box);
    else grouped.set(box.page, [box]);
  }
  return grouped;
}

export interface MarkSummary {
  awarded: number;
  available: number;
  /** Questions whose marks a teacher should look at before accepting them. */
  needsReview: number;
  /** True once marking has run at all. */
  marked: boolean;
  /** True when marking produced a rubric but proposed no marks. */
  rubricOnly: boolean;
}

/**
 * The marking summary for the header.
 *
 * ``rubricOnly`` is reported separately from a genuine zero. A script that was
 * never marked and a script that scored nothing look identical in a total, and
 * only one of them is a result.
 */
export function summarizeMarks(submission: Submission): MarkSummary {
  const grades = submission.grades;
  if (!grades) {
    return { awarded: 0, available: 0, needsReview: 0, marked: false, rubricOnly: false };
  }

  // Same correction as scoreTone: the grades say whether they were judged, so the
  // summary line no longer claims "rubric only" for a script marked all zeros.
  const judged = grades.grades.some((grade) => grade.judged);
  return {
    awarded: grades.total_awarded,
    available: grades.total_available,
    needsReview: grades.review_count,
    marked: true,
    rubricOnly: !judged,
  };
}


export type ScoreTone = "pass" | "partial" | "zero" | "none";

/**
 * Which of the frame's three score pills a grade should wear.
 *
 * The design defines exactly three — full marks, partial, nothing — and a fourth
 * state the design does not show but the product has: not marked at all. That one
 * gets the neutral chip rather than a zero, because "nobody has marked this" and
 * "the student scored nothing" are the same number and completely different
 * facts.
 */
export function scoreTone(grade: QuestionGrade | undefined): ScoreTone {
  if (!grade || grade.marks_available <= 0) return "none";
  /*
   * Read from the grade, not inferred from its citations.
   *
   * This used to treat "no point cites a line" as "nobody marked it", which is
   * wrong for exactly the case that matters: a marker who awards zero cites
   * nothing, because citations are evidence *for* marks and there are none. So a
   * decided 0 out of 4 wore the neutral chip and was indistinguishable from a
   * question nobody had looked at — which is how "the second question does not get
   * scored" was reported, when in fact it had been scored zero every time.
   */
  if (!grade.judged) return "none";
  if (grade.marks_awarded <= 0) return "zero";
  if (grade.marks_awarded >= grade.marks_available) return "pass";
  return "partial";
}

/** The pill's text, or null when there is nothing to show. */
export function scoreLabel(grade: QuestionGrade | undefined): string | null {
  if (!grade || grade.marks_available <= 0) return null;
  const trim = (n: number) => (Number.isInteger(n) ? String(n) : n.toFixed(1));
  return `${trim(grade.marks_awarded)} / ${trim(grade.marks_available)}`;
}

/**
 * The feedback worth showing for one question.
 *
 * Prefers the model's note to the student, then the first rubric-point comment —
 * which is where a skip reason lands, and a teacher needs to see "nothing was
 * written for this" as much as they need a mark.
 */
export function feedbackFor(grade: QuestionGrade | undefined): string | null {
  if (!grade) return null;
  if (grade.feedback) return grade.feedback;
  const comment = grade.rubric_points.find((point) => point.comment)?.comment;
  return comment ?? null;
}


/**
 * Split a printed label into the badge and the sub-part column.
 *
 * The frame puts the question number in a circle and any sub-part letter in its
 * own column beside it, so labels stay aligned down the list however deep they
 * nest. Three shapes occur in real papers and all three appear in the data:
 *
 *   "4."        -> badge 4
 *   "11 (a)"    -> badge 11, sub a
 *   "(i)"       -> badge 3, sub i   — a sub-part printed on its own
 *
 * The third is why this takes a path as well as a label. A comprehension paper
 * prints its sub-parts as a bare "(i)" under a stem, so the label carries no
 * number and the circle showed a lone "i" — which tells a teacher scanning the
 * list nothing about which question it belongs to. The number is not invented:
 * extraction already worked out that this is part of question 3 and wrote it
 * into the path. This is where that becomes visible.
 *
 * What the paper printed always wins for the number. The path fills a gap and
 * never overrules, because the number a teacher reads back to a student has to
 * be the one on the page in front of them.
 */
export function splitLabel(
  labelRaw: string,
  path: readonly string[] = [],
): { badge: string; sub: string | null } {
  const raw = labelRaw.trim();

  const numbered = /^(\d+)\s*[.)]?\s*(?:\(\s*([A-Za-z]+)\s*\)|([A-Za-z]+)[.)])?\s*$/.exec(raw);
  if (numbered) {
    const letter = numbered[2] ?? numbered[3];
    return { badge: numbered[1] ?? "?", sub: letter ? `${letter}.` : null };
  }

  // A bare sub-part: "(i)", "(a)", "ii." — nothing printed says which question
  // it belongs to, so the path is asked.
  const bare = /^\(?\s*([A-Za-z0-9]{1,4})\s*\)?\s*[.)]?$/.exec(raw);
  if (bare) {
    const token = bare[1] ?? "?";
    const parent = path.length > 1 ? path[0] : undefined;
    return parent ? { badge: parent, sub: `${token}.` } : { badge: token, sub: null };
  }

  // Anything else: keep it short enough to fit the circle rather than clip it.
  return { badge: raw.replace(/[^A-Za-z0-9]/g, "").slice(0, 3) || "?", sub: null };
}
