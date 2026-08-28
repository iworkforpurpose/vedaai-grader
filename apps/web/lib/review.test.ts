import { describe, expect, it } from "vitest";
import type {
  AnswerStatus,
  Mapping,
  Question,
  QuestionGrade,
  Submission,
} from "./contracts";
import {
  applyReassignment,
  blockPreview,
  blocksOf,
  buildRows,
  citationHighlight,
  gradeFor,
  highlightByPage,
  isTeacherPlaced,
  movableBlocks,
  orphanHighlightByPage,
  questionAtPoint,
  scoreTone,
  STATUS,
  summarize,
  summarizeMarks,
  untranscribedInkByPage,
} from "./review";

function question(qid: string, label: string, order: number, text = "Some question"): Question {
  return {
    qid,
    label_raw: label,
    text,
    path: qid.split("/").slice(1),
    print_order: order,
    section_id: null,
    stem_ref: null,
    choice_group: null,
    marks: 2,
    line_ids: [],
    geometry: [],
    extraction_confidence: 1,
    depth: 1,
    is_subpart: false,
    parent_qid: null,
  } as unknown as Question;
}

function mapping(
  qid: string,
  status: AnswerStatus,
  boxes: { page: number; x0: number; y0: number; x1: number; y1: number }[] = [],
  blockIds: string[] = [],
): Mapping {
  return {
    qid,
    status,
    block_ids: blockIds,
    start_line_id: null,
    end_line_id: null,
    highlight: boxes.length
      ? {
          boxes: boxes.map((b) => ({
            page: b.page,
            box: { x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1 },
          })),
          derived_from: "ocr_lines",
          pages: [...new Set(boxes.map((b) => b.page))],
          spans_pages: new Set(boxes.map((b) => b.page)).size > 1,
        }
      : null,
    confidence: 0.8,
    evidence: {
      label_agreement: 0,
      semantic_similarity: null,
      order_prior: 0,
      length_plausibility: 0,
      signals: [],
      total_score: 0,
    },
    anchor_id: null,
    shares_block_with: [],
    teacher_override: false,
    needs_review: false,
  } as unknown as Mapping;
}

function submission(overrides: Partial<Submission> = {}): Submission {
  return {
    submission_id: "s1",
    status: "complete",
    question_paper_file: null,
    answer_sheet_file: null,
    pages: [],
    question_paper_lines: null,
    answer_sheet_lines: null,
    ink_regions: [],
    questions: { questions: [], sections: [], stems: [], choice_groups: [], gaps: [], total_marks: null },
    blocks: [],
    anchors: [],
    mapping: null,
    grades: null,
    warnings: [],
    error: null,
    answer_sheet_page_count: 0,
    question_count: 0,
    ...overrides,
  } as unknown as Submission;
}

describe("status vocabulary", () => {
  it("only 'unanswered' claims the student left something blank", () => {
    // The distinction the pipeline works hardest to preserve. A teacher acts on
    // "not answered" without re-reading, so every other absence state has to read
    // as uncertainty rather than as a finding.
    expect(STATUS.unanswered.label).toBe("Not answered");
    expect(STATUS.uncertain.label).toBe("Not found");
    expect(STATUS.ocr_failed.label).toBe("Could not read");
    expect(STATUS.pages_missing.label).toBe("Page may be missing");
  });

  it("flags the states that need a teacher's attention", () => {
    expect(STATUS.ocr_failed.needsAttention).toBe(true);
    expect(STATUS.uncertain.needsAttention).toBe(true);
    expect(STATUS.pages_missing.needsAttention).toBe(true);
    expect(STATUS.answered.needsAttention).toBe(false);
    // A legitimately skipped optional question is not a problem to investigate.
    expect(STATUS.not_required.needsAttention).toBe(false);
  });
});

describe("buildRows", () => {
  it("lists questions in printed order, not label order", () => {
    // Labels restart per section and mix romans with letters, so they cannot
    // order anything; print_order is the authority.
    const sub = submission({
      questions: {
        questions: [question("B/1", "1.", 3), question("A/2", "2.", 1)],
        sections: [],
        stems: [],
        choice_groups: [],
        gaps: [],
        total_marks: null,
      },
    } as unknown as Partial<Submission>);

    expect(buildRows(sub).map((r) => r.question.qid)).toEqual(["A/2", "B/1"]);
  });

  it("shows the label exactly as the paper printed it", () => {
    const sub = submission({
      questions: {
        questions: [question("A/11/a", "11 (a)", 0)],
        sections: [],
        stems: [],
        choice_groups: [],
        gaps: [],
        total_marks: null,
      },
    } as unknown as Partial<Submission>);

    expect(buildRows(sub)[0]!.question.label_raw).toBe("11 (a)");
  });

  it("defaults to unanswered when no mapping exists yet", () => {
    const sub = submission({
      questions: {
        questions: [question("A/1", "1.", 0)],
        sections: [],
        stems: [],
        choice_groups: [],
        gaps: [],
        total_marks: null,
      },
    } as unknown as Partial<Submission>);

    expect(buildRows(sub)[0]!.status).toBe("unanswered");
  });
});

describe("summarize", () => {
  const base = submission({
    questions: {
      questions: [question("A/1", "1.", 0), question("A/2", "2.", 1), question("A/3", "3.", 2)],
      sections: [],
      stems: [],
      choice_groups: [],
      gaps: [],
      total_marks: null,
    },
  } as unknown as Partial<Submission>);

  it("counts answered, unanswered and needs-checking separately", () => {
    const sub = {
      ...base,
      mapping: {
        mappings: [
          mapping("A/1", "answered", [{ page: 0, x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.2 }]),
          mapping("A/2", "unanswered"),
          mapping("A/3", "uncertain"),
        ],
        orphans: [],
        unassigned_ink_ratio: 0,
        absence_claims_suppressed: false,
      },
    } as unknown as Submission;

    const summary = summarize(sub, buildRows(sub));
    expect(summary.answered).toBe(1);
    expect(summary.notAnswered).toBe(1);
    expect(summary.needsAttention).toBe(1);
  });

  it("reports when the unanswered count must be withheld", () => {
    // Unplaced writing means the count would be a guess, and a wrong count here
    // is worse than no count.
    const sub = {
      ...base,
      mapping: {
        mappings: [mapping("A/1", "uncertain")],
        orphans: [],
        unassigned_ink_ratio: 0.4,
        absence_claims_suppressed: true,
      },
    } as unknown as Submission;

    expect(summarize(sub, buildRows(sub)).absenceSuppressed).toBe(true);
  });
});

describe("highlightByPage", () => {
  it("groups a page-spanning highlight by page", () => {
    const grouped = highlightByPage(
      mapping("A/1", "answered", [
        { page: 0, x0: 0.1, y0: 0.8, x1: 0.9, y1: 0.95 },
        { page: 1, x0: 0.1, y0: 0.05, x1: 0.9, y1: 0.3 },
      ]),
    );
    expect([...grouped.keys()]).toEqual([0, 1]);
  });

  it("returns nothing for a question with no highlight", () => {
    expect(highlightByPage(mapping("A/1", "unanswered")).size).toBe(0);
    expect(highlightByPage(undefined).size).toBe(0);
  });
});

describe("questionAtPoint", () => {
  const sub = {
    ...submission({
      questions: {
        questions: [question("A/1", "1.", 0), question("A/2", "2.", 1)],
        sections: [],
        stems: [],
        choice_groups: [],
        gaps: [],
        total_marks: null,
      },
    } as unknown as Partial<Submission>),
    mapping: {
      mappings: [
        // A large region, and a small one inside it.
        mapping("A/1", "answered", [{ page: 0, x0: 0.0, y0: 0.0, x1: 1.0, y1: 1.0 }]),
        mapping("A/2", "answered", [{ page: 0, x0: 0.4, y0: 0.4, x1: 0.6, y1: 0.6 }]),
      ],
      orphans: [],
      unassigned_ink_ratio: 0,
      absence_claims_suppressed: false,
    },
  } as unknown as Submission;

  it("prefers the smallest containing region", () => {
    // Otherwise a page-spanning answer swallows every click on the pages it
    // covers, and the reverse lookup becomes useless exactly where the page is
    // busiest.
    const rows = buildRows(sub);
    expect(questionAtPoint(rows, 0, 0.5, 0.5)?.question.qid).toBe("A/2");
  });

  it("falls back to the enclosing region outside the small one", () => {
    const rows = buildRows(sub);
    expect(questionAtPoint(rows, 0, 0.05, 0.05)?.question.qid).toBe("A/1");
  });

  it("returns null on a page with no mapped answers", () => {
    expect(questionAtPoint(buildRows(sub), 3, 0.5, 0.5)).toBeNull();
  });
});

describe("untranscribedInkByPage", () => {
  it("surfaces only ink the recognizer never accounted for", () => {
    // The first place to look when a question reads "not found".
    const sub = submission({
      ink_regions: [
        { region_id: "ink:1", page: 0, is_orphan_ink: true, box: { x0: 0, y0: 0, x1: 0.2, y1: 0.2 } },
        { region_id: "ink:2", page: 0, is_orphan_ink: false, box: { x0: 0, y0: 0.3, x1: 0.2, y1: 0.4 } },
      ],
    } as unknown as Partial<Submission>);

    const grouped = untranscribedInkByPage(sub);
    expect(grouped.get(0)?.map((r) => r.region_id)).toEqual(["ink:1"]);
  });
});

describe("applyReassignment", () => {
  const sub = {
    ...submission({
      questions: {
        questions: [question("A/1", "1.", 0), question("A/2", "2.", 1)],
        sections: [],
        stems: [],
        choice_groups: [],
        gaps: [],
        total_marks: null,
      },
      blocks: [
        {
          block_id: "blk:000",
          line_ids: ["as:0001"],
          ink_region_ids: [],
          text: "some answer",
          geometry: [{ page: 0, box: { x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.2 } }],
          pages_spanned: [0],
          has_continuation_marker: false,
          is_text_free: false,
          spans_pages: false,
        },
      ],
    } as unknown as Partial<Submission>),
    mapping: {
      mappings: [
        mapping("A/1", "answered", [{ page: 0, x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.2 }], ["blk:000"]),
        mapping("A/2", "unanswered"),
      ],
      orphans: [],
      unassigned_ink_ratio: 0,
      absence_claims_suppressed: false,
    },
  } as unknown as Submission;

  it("moves the block to the chosen question and records the override", () => {
    const next = applyReassignment(sub, "blk:000", "A/2");
    const target = next.mapping!.mappings.find((m) => m.qid === "A/2")!;

    expect(target.status).toBe("answered");
    expect(target.block_ids).toEqual(["blk:000"]);
    expect(target.teacher_override).toBe(true);
  });

  it("does not declare the losing question blank", () => {
    // The teacher moved an answer, which says nothing about whether the original
    // question was attempted. Asserting a blank on the strength of a correction
    // elsewhere is exactly the unfounded absence claim to avoid.
    const next = applyReassignment(sub, "blk:000", "A/2");
    const loser = next.mapping!.mappings.find((m) => m.qid === "A/1")!;

    expect(loser.block_ids).toEqual([]);
    expect(loser.status).not.toBe("unanswered");
    expect(loser.status).toBe("uncertain");
  });

  it("is a no-op when there is no mapping to change", () => {
    const bare = submission();
    expect(applyReassignment(bare, "blk:000", "A/1")).toBe(bare);
  });

  it("keeps writing displaced from the target reachable as an orphan", () => {
    // If this state is all the teacher has — the request failed — every block
    // must still be visible somewhere. Writing that belongs to no question and
    // appears in no orphan list has vanished from the interface.
    const next = applyReassignment(sub, "blk:000", "A/2");
    const reachable = new Set([
      ...next.mapping!.mappings.flatMap((m) => m.block_ids),
      ...next.mapping!.orphans.map((o) => o.block_id),
    ]);
    expect(reachable.has("blk:000")).toBe(true);
  });
});

describe("appending to an answered question", () => {
  // The client mirrors the server here on purpose: replacing would make a split
  // answer unrepairable, and a divergence would show as the highlight jumping
  // when the server's response arrives.
  const twoBlocks = {
    ...submission({
      questions: {
        questions: [question("A/1", "1.", 0), question("A/2", "2.", 1)],
        sections: [],
        stems: [],
        choice_groups: [],
        gaps: [],
        total_marks: null,
      },
      blocks: [
        {
          block_id: "blk:000",
          line_ids: ["as:0001"],
          ink_region_ids: [],
          text: "first half",
          geometry: [{ page: 0, box: { x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.2 } }],
          pages_spanned: [0],
          has_continuation_marker: false,
          is_text_free: false,
          spans_pages: false,
        },
        {
          block_id: "blk:001",
          line_ids: ["as:0002"],
          ink_region_ids: [],
          text: "second half",
          geometry: [{ page: 1, box: { x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.2 } }],
          pages_spanned: [1],
          has_continuation_marker: false,
          is_text_free: false,
          spans_pages: false,
        },
      ],
    } as unknown as Partial<Submission>),
    mapping: {
      mappings: [
        mapping("A/1", "answered", [{ page: 0, x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.2 }], ["blk:000"]),
        mapping("A/2", "answered", [{ page: 1, x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.2 }], ["blk:001"]),
      ],
      orphans: [],
      unassigned_ink_ratio: 0,
      absence_claims_suppressed: false,
    },
  } as unknown as Submission;

  it("holds both blocks and unions their highlights", () => {
    const next = applyReassignment(twoBlocks, "blk:001", "A/1");
    const target = next.mapping!.mappings.find((m) => m.qid === "A/1")!;

    expect(target.block_ids).toEqual(["blk:000", "blk:001"]);
    expect(target.highlight!.boxes).toHaveLength(2);
    expect(target.highlight!.spans_pages).toBe(true);
    expect(target.highlight!.pages).toEqual([0, 1]);
  });

  it("orders merged blocks by position on the sheet, not click order", () => {
    // start_line_id and end_line_id name a span, so the order the teacher
    // happened to click in must not decide which end is which.
    const next = applyReassignment(twoBlocks, "blk:001", "A/1");
    expect(next.mapping!.mappings.find((m) => m.qid === "A/1")!.block_ids).toEqual([
      "blk:000",
      "blk:001",
    ]);
  });
});

describe("blocksOf", () => {
  const sub = {
    ...submission({
      questions: {
        questions: [question("A/1", "1.", 0)],
        sections: [],
        stems: [],
        choice_groups: [],
        gaps: [],
        total_marks: null,
      },
      blocks: [
        {
          block_id: "blk:000",
          line_ids: ["as:0001"],
          ink_region_ids: [],
          text: "  a written   answer  ",
          geometry: [{ page: 0, box: { x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.2 } }],
          pages_spanned: [0],
          has_continuation_marker: false,
          is_text_free: false,
          spans_pages: false,
        },
        {
          block_id: "blk:001",
          line_ids: [],
          ink_region_ids: ["ink:1"],
          text: "",
          geometry: [{ page: 2, box: { x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.2 } }],
          pages_spanned: [2],
          has_continuation_marker: false,
          is_text_free: true,
          spans_pages: false,
        },
      ],
    } as unknown as Partial<Submission>),
    mapping: {
      mappings: [mapping("A/1", "answered", [], ["blk:001", "blk:000"])],
      orphans: [],
      unassigned_ink_ratio: 0,
      absence_claims_suppressed: false,
    },
  } as unknown as Submission;

  it("resolves the blocks a question owns", () => {
    const rows = buildRows(sub);
    expect(blocksOf(sub, rows[0]!.mapping).map((b) => b.block_id)).toEqual([
      "blk:001",
      "blk:000",
    ]);
  });

  it("ignores block ids with no matching block", () => {
    const rows = buildRows(sub);
    const stale = { ...rows[0]!.mapping!, block_ids: ["blk:000", "blk:missing"] };
    expect(blocksOf(sub, stale).map((b) => b.block_id)).toEqual(["blk:000"]);
  });

  it("names a block with no readable text by where it is", () => {
    // A diagram, or handwriting the recognizer failed on, still needs a label
    // the teacher can act on.
    const textFree = sub.blocks.find((b) => b.block_id === "blk:001")!;
    expect(blockPreview(textFree)).toBe("writing on page 3");
    expect(blockPreview(sub.blocks[0]!)).toBe("a written answer");
  });
});

describe("citationHighlight", () => {
  // The mechanism that makes a mark checkable. The model named lines and never
  // coordinates; resolving those names is what turns a citation into something
  // the teacher can look at.
  const sub = submission({
    answer_sheet_lines: {
      kind: "answer_sheet",
      engine: "paddle_ocr_vl",
      lines: [
        {
          line_id: "as:0001",
          page: 0,
          box: { x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.14 },
          text: "first",
        },
        {
          line_id: "as:0002",
          page: 1,
          box: { x0: 0.1, y0: 0.2, x1: 0.9, y1: 0.24 },
          text: "second",
        },
      ],
    },
  } as unknown as Partial<Submission>);

  it("resolves cited line ids to their boxes, grouped by page", () => {
    const grouped = citationHighlight(sub, ["as:0001", "as:0002"]);
    expect([...grouped.keys()].sort()).toEqual([0, 1]);
    expect(grouped.get(0)![0]!.box.y0).toBeCloseTo(0.1);
  });

  it("shows nothing for a citation that does not resolve", () => {
    // Better than a rectangle over the wrong place: an unresolvable citation is
    // exactly the case the validation exists to catch, and drawing a guess would
    // hide it.
    expect(citationHighlight(sub, ["as:9999"]).size).toBe(0);
  });

  it("returns nothing when the sheet was never transcribed", () => {
    expect(citationHighlight(submission(), ["as:0001"]).size).toBe(0);
  });
});

describe("summarizeMarks", () => {
  function grade(
    qid: string,
    awarded: number,
    available: number,
    cited: string[],
    judged = true,
  ) {
    return {
      qid,
      judged,
      marks_available: available,
      marks_awarded: awarded,
      rubric_points: [
        {
          point_id: `${qid}#1`,
          criterion: "something",
          marks_available: available,
          marks_awarded: awarded,
          satisfied: awarded > 0,
          cited_line_ids: cited,
          comment: null,
        },
      ],
      feedback: null,
      confidence: 0.8,
      graded_on_partial_text: false,
      fraction: available ? awarded / available : null,
      needs_review: false,
    };
  }

  it("reports nothing marked before marking has run", () => {
    expect(summarizeMarks(submission()).marked).toBe(false);
  });

  it("separates a rubric with no marks from a genuine zero", () => {
    /*
     * A script that was never marked and one that scored nothing look identical
     * in a total, and only one of them is a result.
     *
     * The intent was right and the fixture was wrong. It told the two apart by
     * giving the scored zero a citation, which is the one thing a real zero never
     * has — citations evidence marks that were awarded. So this passed while a
     * marker's 0 out of 4 was being displayed as an unmarked question, because the
     * fixture had been built to agree with the implementation rather than with a
     * real payload. The discriminator is now the flag the grade actually carries.
     */
    const rubricOnly = submission({
      grades: {
        grades: [grade("A/1", 0, 2, [], false)],
        overall_feedback: null,
        weak_topics: [],
        committed: false,
        total_awarded: 0,
        total_available: 2,
        review_count: 0,
      },
    } as unknown as Partial<Submission>);
    expect(summarizeMarks(rubricOnly).rubricOnly).toBe(true);

    const scoredZero = submission({
      grades: {
        // A judged zero: no citation, because nothing was awarded to evidence.
        grades: [grade("A/1", 0, 2, [], true)],
        overall_feedback: null,
        weak_topics: [],
        committed: false,
        total_awarded: 0,
        total_available: 2,
        review_count: 0,
      },
    } as unknown as Partial<Submission>);
    expect(summarizeMarks(scoredZero).rubricOnly).toBe(false);
    expect(summarizeMarks(scoredZero).marked).toBe(true);
  });
});

describe("gradeFor", () => {
  it("finds the grade for one question", () => {
    const sub = submission({
      grades: {
        grades: [{ qid: "A/2", marks_awarded: 1, marks_available: 2, rubric_points: [] }],
        overall_feedback: null,
        weak_topics: [],
        committed: false,
        total_awarded: 1,
        total_available: 2,
        review_count: 0,
      },
    } as unknown as Partial<Submission>);
    expect(gradeFor(sub, "A/2")?.marks_awarded).toBe(1);
    expect(gradeFor(sub, "A/1")).toBeUndefined();
  });

  it("is undefined before marking has run", () => {
    expect(gradeFor(submission(), "A/1")).toBeUndefined();
  });
});

describe("stems", () => {
  // "2. Answer the following:" is a heading. It is kept as a question because the
  // paper printed it and the teacher expects to see it, but it asks nothing.
  const sub = {
    ...submission({
      questions: {
        questions: [
          { ...question("A/1", "1.", 0), is_stem: false },
          { ...question("A/2", "2.", 1, "Answer the following:"), is_stem: true },
          { ...question("A/2/i", "(i)", 2), is_stem: false },
        ],
        sections: [],
        stems: [],
        choice_groups: [],
        gaps: [],
        total_marks: null,
      },
    } as unknown as Partial<Submission>),
    mapping: {
      mappings: [
        mapping("A/1", "answered", [{ page: 0, x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.2 }]),
        mapping("A/2", "not_required"),
        mapping("A/2/i", "unanswered"),
      ],
      orphans: [],
      unassigned_ink_ratio: 0,
      absence_claims_suppressed: false,
    },
  } as unknown as Submission;

  it("is not counted as a question that could be answered", () => {
    // Otherwise "1 of 3 answered" reports a denominator the paper never asked.
    const summary = summarize(sub, buildRows(sub));
    expect(summary.total).toBe(2);
    expect(summary.answered).toBe(1);
    expect(summary.notAnswered).toBe(1);
  });

  it("is still listed, so the paper's numbering is preserved", () => {
    expect(buildRows(sub).map((r) => r.question.label_raw)).toEqual(["1.", "2.", "(i)"]);
  });
});


describe("scoreTone and a decided zero", () => {
  const grade = (over: Partial<QuestionGrade>): QuestionGrade =>
    ({
      qid: "A/2",
      marks_available: 4,
      marks_awarded: 0,
      rubric_points: [],
      feedback: null,
      graded_by: "openai:gpt-4o-mini",
      judged: false,
      confidence: 0,
      graded_on_partial_text: false,
      fraction: 0,
      ...over,
    }) as QuestionGrade;

  it("shows a marker's zero as a zero", () => {
    // The bug this guards: a zero cites nothing, because citations evidence marks
    // that were given. Inferring "judged" from citations made this read "none" and
    // look identical to an unmarked question.
    expect(scoreTone(grade({ judged: true }))).toBe("zero");
  });

  it("shows an unjudged question as unscored, whatever its marks say", () => {
    expect(scoreTone(grade({ judged: false }))).toBe("none");
  });

  it("still distinguishes partial from full", () => {
    expect(scoreTone(grade({ judged: true, marks_awarded: 1 }))).toBe("partial");
    expect(scoreTone(grade({ judged: true, marks_awarded: 4 }))).toBe("pass");
  });

  it("treats a question worth no marks as unscored even when judged", () => {
    expect(scoreTone(grade({ judged: true, marks_available: 0 }))).toBe("none");
  });
});


describe("reassignment helpers", () => {
  const block = (id: string, text: string, page: number) => ({
    block_id: id,
    text,
    line_ids: [`as:${id}`],
    geometry: [{ page, box: { x0: 0.1, y0: 0.1, x1: 0.9, y1: 0.2 } }],
    pages_spanned: [page],
    start_line_id: `as:${id}`,
    end_line_id: `as:${id}`,
  });

  const withOrphans = () =>
    submission({
      blocks: [block("blk:000", "Pandas are specialists.", 0), block("blk:001", "Stray note.", 1)],
      mapping: {
        mappings: [
          {
            qid: "A/1",
            status: "answered",
            block_ids: ["blk:000"],
            highlight: null,
            teacher_override: true,
            confidence: 0.9,
            evidence: [],
          },
        ],
        orphans: [
          {
            block_id: "blk:001",
            text_preview: "Stray note.",
            best_guess_qid: null,
            best_guess_score: null,
            highlight: {
              boxes: [{ page: 1, box: { x0: 0.1, y0: 0.3, x1: 0.9, y1: 0.4 } }],
              derived_from: "ocr_lines",
              pages: [1],
              spans_pages: false,
            },
          },
        ],
        unassigned_ink_ratio: 0,
        absence_claims_suppressed: false,
      },
    } as unknown as Partial<Submission>);

  it("groups unplaced writing by the page it is on", () => {
    const byPage = orphanHighlightByPage(withOrphans().mapping ?? undefined);
    expect([...byPage.keys()]).toEqual([1]);
    expect(byPage.get(1)).toHaveLength(1);
  });

  it("has nothing to show when every block found a question", () => {
    expect(orphanHighlightByPage(undefined).size).toBe(0);
  });

  it("lists a question's blocks individually, so one of several can be moved", () => {
    // A wrong mapping is often one block out of several. Moving the whole answer
    // would trade one error for another.
    const blocks = movableBlocks(withOrphans(), "A/1");
    expect(blocks.map((b) => b.block_id)).toEqual(["blk:000"]);
  });

  it("reports a mapping the teacher set by hand", () => {
    expect(isTeacherPlaced(withOrphans(), "A/1")).toBe(true);
    expect(isTeacherPlaced(withOrphans(), "A/2")).toBe(false);
  });
});
