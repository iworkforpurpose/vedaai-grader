"use client";

import { useMemo, useState } from "react";
import type { DocumentKind, InkRegion, LineIndex, Page, Submission } from "@/lib/contracts";
import { DebugOverlay } from "./DebugOverlay";
import { InkOverlay, summarizeInk } from "./InkOverlay";
import { PageCanvas } from "./PageCanvas";

/**
 * The geometry inspector.
 *
 * Its job is to make the transcription layer falsifiable by eye before anything
 * is built on top of it. Every highlight this product will draw comes from the
 * boxes shown here, so this is where a geometry fault is caught — cheaply, and
 * before it has been mistaken for a mapping fault three phases later.
 *
 * Pages render in one continuous scroll rather than a pager. That is the layout
 * the review surface needs anyway, because an answer spanning a page boundary
 * then reads as a single continuous region instead of two disconnected ones.
 */
export function DebugReview({
  submission,
  questionLines,
  answerLines,
  inkRegions,
}: {
  submission: Submission;
  questionLines: LineIndex | null;
  answerLines: LineIndex | null;
  inkRegions: readonly InkRegion[];
}): React.JSX.Element {
  const [kind, setKind] = useState<DocumentKind>("question_paper");
  const [showWords, setShowWords] = useState(false);
  const [showText, setShowText] = useState(true);
  const [showInk, setShowInk] = useState(false);
  const [showNoise, setShowNoise] = useState(false);
  const [orphansOnly, setOrphansOnly] = useState(false);

  const index = kind === "question_paper" ? questionLines : answerLines;
  const pages = useMemo(
    () => submission.pages.filter((p) => p.kind === kind).sort((a, b) => a.index - b.index),
    [submission.pages, kind],
  );

  const linesByPage = useMemo(() => {
    const out = new Map<number, LineIndex["lines"]>();
    for (const line of index?.lines ?? []) {
      const bucket = out.get(line.page);
      if (bucket) bucket.push(line);
      else out.set(line.page, [line]);
    }
    return out;
  }, [index]);

  const lowConfidence = (index?.lines ?? []).filter((l) => l.is_low_confidence).length;

  // Ink exists only for the answer sheet; a printed paper has no student marking.
  const inkForKind = kind === "answer_sheet" ? inkRegions : [];
  const inkByPage = useMemo(() => {
    const out = new Map<number, InkRegion[]>();
    for (const region of inkForKind) {
      const bucket = out.get(region.page);
      if (bucket) bucket.push(region);
      else out.set(region.page, [region]);
    }
    return out;
  }, [inkForKind]);
  const inkSummary = useMemo(() => summarizeInk(inkForKind), [inkForKind]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-5)" }}>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "var(--sp-4)",
          alignItems: "center",
          padding: "var(--sp-3) var(--sp-4)",
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          position: "sticky",
          top: 0,
          zIndex: 2,
        }}
      >
        <SegmentedControl
          value={kind}
          onChange={setKind}
          options={[
            { value: "question_paper", label: "Question paper" },
            { value: "answer_sheet", label: "Answer sheet" },
          ]}
        />
        <Toggle checked={showText} onChange={setShowText} label="Line IDs" />
        <Toggle checked={showWords} onChange={setShowWords} label="Word boxes" />
        {kind === "answer_sheet" && (
          <>
            <Toggle checked={showInk} onChange={setShowInk} label="Ink" />
            {showInk && (
              <>
                <Toggle
                  checked={orphansOnly}
                  onChange={setOrphansOnly}
                  label="Untranscribed only"
                />
                <Toggle checked={showNoise} onChange={setShowNoise} label="Not-the-page" />
              </>
            )}
          </>
        )}

        <span
          style={{
            marginLeft: "auto",
            fontFamily: "var(--font-mono)",
            fontSize: "var(--fs-xs)",
            color: "var(--text-muted)",
          }}
        >
          {pages.length} pages · {index?.lines.length ?? 0} lines
          {index ? ` · ${index.engine}` : ""}
          {lowConfidence > 0 ? ` · ${lowConfidence} low-confidence` : ""}
          {showInk && inkForKind.length > 0
            ? ` · ink: ${inkSummary.substantive} regions, ${inkSummary.orphans} untranscribed`
            : ""}
        </span>
      </div>

      {submission.warnings.length > 0 && (
        <ul
          style={{
            margin: 0,
            padding: "var(--sp-3) var(--sp-4) var(--sp-3) var(--sp-6)",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderLeft: "3px solid var(--status-review)",
            borderRadius: "var(--radius)",
            fontSize: "var(--fs-sm)",
            color: "var(--text-2)",
          }}
        >
          {submission.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}

      {index === null ? (
        <p style={{ color: "var(--text-muted)" }}>
          No transcription for this document, so there are no boxes to draw. Page images are
          still shown below.
        </p>
      ) : null}

      <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-5)" }}>
        {pages.map((page: Page) => (
          <PageCanvas
            key={page.image_key}
            page={page}
            label={`Page ${page.index + 1} · ${page.width}×${page.height}px @ ${page.dpi} DPI`}
          >
            <DebugOverlay
              lines={linesByPage.get(page.index) ?? []}
              showWords={showWords}
              showText={showText}
            />
            {showInk && (
              <InkOverlay
                regions={inkByPage.get(page.index) ?? []}
                showNoise={showNoise}
                orphansOnly={orphansOnly}
              />
            )}
          </PageCanvas>
        ))}
      </div>
    </div>
  );
}

function SegmentedControl<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (next: T) => void;
  options: readonly { value: T; label: string }[];
}): React.JSX.Element {
  return (
    <div style={{ display: "flex", border: "1px solid var(--border)", borderRadius: "var(--radius)" }}>
      {options.map((option, i) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            style={{
              fontFamily: "var(--font-ui)",
              fontSize: "var(--fs-sm)",
              padding: "var(--sp-1) var(--sp-3)",
              border: "none",
              borderLeft: i === 0 ? "none" : "1px solid var(--border)",
              background: active ? "var(--accent)" : "transparent",
              color: active ? "var(--accent-contrast)" : "var(--text-2)",
              cursor: "pointer",
            }}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
}): React.JSX.Element {
  return (
    <label
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--sp-2)",
        fontSize: "var(--fs-sm)",
        cursor: "pointer",
      }}
    >
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}
