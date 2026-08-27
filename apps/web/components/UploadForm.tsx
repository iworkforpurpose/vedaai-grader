"use client";

import Image from "next/image";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { API_BASE } from "@/lib/api";
import {
  ArrowRightIcon,
  ClockGlyph,
  CloudGlyph,
  GearGlyph,
  GridGlyph,
  UploadIcon,
} from "./icons";

/**
 * The upload screen.
 *
 * Both documents post straight to the pipeline service. The action stays disabled
 * until both are chosen — which is what the frame's empty state shows, and also
 * what the pipeline requires: a question paper with no answer sheet has nothing to
 * map, and saying so before the click beats a validation error after it.
 *
 * Each drop zone accepts a drop as well as a click. The design draws a dashed
 * rectangle, the conventional signal for a drop target, so accepting only clicks
 * would be a promise the interface makes and does not keep.
 */

type Slot = "question_paper" | "answer_sheet";

const ACCEPT = ".pdf,.png,.jpg,.jpeg,.tif,.tiff,.webp";

export function UploadForm(): React.JSX.Element {
  const router = useRouter();
  const [files, setFiles] = useState<Record<Slot, File | null>>({
    question_paper: null,
    answer_sheet: null,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ready = files.question_paper !== null && files.answer_sheet !== null;

  async function onSubmit(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!ready) return;

    setError(null);
    setBusy(true);

    const body = new FormData();
    body.append("question_paper", files.question_paper as File);
    body.append("answer_sheet", files.answer_sheet as File);

    try {
      const response = await fetch(`${API_BASE}/submissions`, { method: "POST", body });
      if (!response.ok) {
        // The API's 422 detail names what was wrong with the file — wrong format,
        // too many pages, empty upload — and that is actionable, so it is shown
        // verbatim rather than replaced with a generic failure.
        const detail = (await response.json().catch(() => null)) as { detail?: string } | null;
        setError(detail?.detail ?? `Upload failed with HTTP ${response.status}`);
        return;
      }
      const submission = (await response.json()) as { submission_id: string };
      router.push(`/review/${submission.submission_id}`);
    } catch {
      setError(`Cannot reach the grader service at ${API_BASE}. Check that it is running.`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="upload" onSubmit={onSubmit}>
      <div className="upload-heading">
        <h1 className="upload-title">
          Upload <em>Question Paper &amp; Answer Sheets</em>
        </h1>
        <p className="upload-subtitle">Upload both files to get started</p>
      </div>

      <Hero />

      <div className="dropzones">
        <DropZone
          slot="question_paper"
          label="Question Paper"
          file={files.question_paper}
          onPick={(file) => setFiles((current) => ({ ...current, question_paper: file }))}
        />
        <DropZone
          slot="answer_sheet"
          label="Answer Sheet"
          file={files.answer_sheet}
          onPick={(file) => setFiles((current) => ({ ...current, answer_sheet: file }))}
        />
      </div>

      <div className="actions">
        <button type="submit" className="cta" disabled={!ready || busy}>
          {busy ? "Mapping…" : "Start Mapping"}
          {!busy && <ArrowRightIcon size={20} />}
        </button>

        <p className="cta-caption">
          {busy
            ? "Rendering pages and reading the handwriting. About 15 seconds a page."
            : "Once both files are uploaded, you’ll be able to map answers with questions"}
        </p>

        {error !== null && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
      </div>
    </form>
  );
}

function DropZone({
  slot,
  label,
  file,
  onPick,
}: {
  slot: Slot;
  label: string;
  file: File | null;
  onPick: (file: File | null) => void;
}): React.JSX.Element {
  const [dragging, setDragging] = useState(false);
  const id = `pick-${slot}`;

  return (
    <label
      className="dropzone"
      htmlFor={id}
      data-filled={file !== null}
      data-dragging={dragging}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        const dropped = event.dataTransfer.files?.[0];
        if (dropped) onPick(dropped);
      }}
    >
      {/* Associated by id rather than nested. A file input inside its own label is
          activated twice by one click — the picker opens, and a second opens on
          top of it. */}
      <input
        id={id}
        type="file"
        name={slot}
        accept={ACCEPT}
        onChange={(event) => onPick(event.target.files?.[0] ?? null)}
      />

      <span className="dropzone-icon">
        <UploadIcon size={20} />
      </span>
      <span className="dropzone-label">
        Upload <em>{label}</em>
      </span>
      {file === null ? (
        <span className="dropzone-hint">Max 10MB</span>
      ) : (
        <span className="dropzone-file">{file.name}</span>
      )}
    </label>
  );
}

/**
 * The illustration: the teacher portrait ringed by four badges.
 *
 * The portrait is the real exported asset. Its geometry follows the file — a
 * 138px outer circle at 10% accent, a 108px inner at 26%, the portrait at 79 of
 * 138, and four 13px badges at the designed angles — so the ring proportions hold
 * as the whole thing scales with the viewport.
 */
function Hero(): React.JSX.Element {
  return (
    <div className="hero" aria-hidden>
      <span className="hero-portrait">
        <Image
          src="/brand/teacher.png"
          alt=""
          width={79}
          height={97}
          priority
          sizes="79px"
        />
      </span>
      <span className="hero-badge" data-at="tr">
        <ClockGlyph />
      </span>
      <span className="hero-badge" data-at="tl">
        <GridGlyph />
      </span>
      <span className="hero-badge" data-at="br">
        <CloudGlyph />
      </span>
      <span className="hero-badge" data-at="bl">
        <GearGlyph />
      </span>
    </div>
  );
}
