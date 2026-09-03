"use client";

import Image from "next/image";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { API_BASE } from "@/lib/api";
import { crossFade } from "@/lib/transitions";
import { TypedPhrase } from "./TypedPhrase";
import { LoadingStage } from "./LoadingStage";
import { PIPELINE_PHASES } from "./LoadingPhases";
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

export function UploadForm({
  onWorkingChange,
  maxUploadBytes,
}: {
  /** Reported upward so the shell can collapse the rail, as the frame shows it. */
  onWorkingChange?: (working: boolean) => void;
  /**
   * The service's own limit, read from its health payload.
   *
   * A prop rather than a constant here because the number belongs to the code
   * that enforces it. The label used to read "Max 10MB", which was the cap of a
   * host the uploads no longer pass through, and understating a limit by four
   * times stops someone uploading a file that would have worked.
   *
   * Undefined when the health check did not answer. The hint then describes what
   * the input accepts and says nothing about size, which is better than naming a
   * limit nobody confirmed.
   */
  maxUploadBytes?: number;
}): React.JSX.Element {
  const router = useRouter();
  const [files, setFiles] = useState<Record<Slot, File | null>>({
    question_paper: null,
    answer_sheet: null,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ready = files.question_paper !== null && files.answer_sheet !== null;

  /**
   * Refuse an oversized file here rather than after uploading it.
   *
   * Without this the browser sends the whole thing to object storage, the service
   * reads it back, and only then reports the size — so the person waits out an
   * upload that was always going to be rejected, and on a phone pays for it.
   */
  function pick(slot: Slot, file: File | null): void {
    if (file && tooLarge(file, maxUploadBytes)) {
      setError(
        `${file.name} is ${(file.size / 1e6).toFixed(1)} MB, above the ` +
          `${Math.floor((maxUploadBytes ?? 0) / 1e6)} MB limit.`,
      );
      return;
    }
    setError(null);
    setFiles((current) => ({ ...current, [slot]: file }));
  }

  async function onSubmit(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!ready) return;

    setError(null);
    // The swap this exists for: the form is replaced by the waiting screen in one
    // frame, and without a cross-fade that is the most abrupt moment in the app.
    crossFade(() => {
      setBusy(true);
      onWorkingChange?.(true);
    });

    // Whether this call ends by navigating away rather than returning to the form.
    let leaving = false;

    try {
      const body = await sendDocuments(
        files.question_paper as File,
        files.answer_sheet as File,
      );
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
      // Stay on the waiting screen. We are leaving for the review route, and
      // clearing `busy` here would put the form back for the moment the
      // navigation takes.
      //
      // That moment used to be invisible because the upload response only arrived
      // once the whole pipeline had finished, by which point nobody was looking at
      // this component. Now that it returns in about a second, dropping back to
      // the form is a visible flash of the screen the reader just left.
      leaving = true;
      router.push(`/review/${submission.submission_id}`);
    } catch (cause) {
      setError(
        cause instanceof Error && cause.message.startsWith("Uploading")
          ? cause.message
          : `Cannot reach the grader service at ${API_BASE}. Check that it is running.`,
      );
    } finally {
      if (!leaving) {
        crossFade(() => {
          setBusy(false);
          onWorkingChange?.(false);
        });
      }
    }
  }

  /*
   * The waiting screen, while the request is in flight.
   *
   * This is where the loading frame belongs, and it was unreachable. The review
   * screen renders it when a submission reads `processing`, but the POST only
   * resolves once ingest has finished — so by the time the router navigates the
   * status is already `complete`, and the frame could never appear. All a teacher
   * saw for the better part of a minute was a button that said "Mapping…", which
   * is indistinguishable from a click that did nothing.
   *
   * Rendered from inside this component rather than swapped in by the parent, so
   * the component that owns the request stays mounted: unmounting it mid-flight
   * would drop the error path, and a failed upload would hang on this screen
   * forever instead of coming back and saying why.
   */
  if (busy) {
    return <LoadingStage phases={PIPELINE_PHASES} />;
  }

  return (
    <form className="upload" onSubmit={onSubmit}>
      <div className="upload-heading">
        <h1 className="upload-title">
          Upload{" "}
          <em>
            <TypedPhrase text="Question Paper & Answer Sheets" />
          </em>
        </h1>
        <p className="upload-subtitle">Upload both files to get started</p>
      </div>

      <Hero />

      <div className="dropzones">
        <DropZone
          slot="question_paper"
          label="Question Paper"
          file={files.question_paper}
          hint={sizeHint(maxUploadBytes)}
          onPick={(file) => pick("question_paper", file)}
        />
        <DropZone
          slot="answer_sheet"
          label="Answer Sheet"
          file={files.answer_sheet}
          hint={sizeHint(maxUploadBytes)}
          onPick={(file) => pick("answer_sheet", file)}
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

/**
 * Get both documents to the service, past it where possible.
 *
 * Asks where to put them first. When object storage is configured the browser posts
 * each file straight there and this returns the keys — the service never carries
 * the bytes, so whatever limit its host places on request bodies stops applying.
 * That matters concretely: the service accepts documents up to 40 MB, and an API
 * gateway in front of it caps requests at 10 MB. Routing uploads through the host
 * would let the host decide the product's limit.
 *
 * When there is no bucket — a laptop, no AWS — the same files go in the request as
 * before. Not a fallback so much as the other environment: one path each.
 */
async function sendDocuments(paper: File, sheet: File): Promise<FormData> {
  const body = new FormData();

  const plan = await fetch(`${API_BASE}/uploads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question_paper_name: paper.name,
      answer_sheet_name: sheet.name,
    }),
  });

  // A failure here is not fatal: posting the files still works, and is what a
  // deployment without object storage does anyway.
  const mode = plan.ok
    ? ((await plan.json()) as {
        mode: string;
        slots?: Record<
          string,
          { key: string; url: string; fields: Record<string, string> }
        >;
      })
    : { mode: "direct" as const };

  if (mode.mode !== "s3" || !mode.slots) {
    body.append("question_paper", paper);
    body.append("answer_sheet", sheet);
    return body;
  }

  // Both at once. They are independent objects and the sheet is much the larger of
  // the two, so waiting for the paper first would add its latency for nothing.
  await Promise.all([
    send(mode.slots.question_paper!, paper),
    send(mode.slots.answer_sheet!, sheet),
  ]);

  body.append("question_paper_key", mode.slots.question_paper!.key);
  body.append("answer_sheet_key", mode.slots.answer_sheet!.key);
  return body;
}

/**
 * Post one file to its signed destination.
 *
 * A signed form rather than a signed PUT, because only a POST policy can carry a
 * size condition. A signed PUT authorises an object of any size, and the URL was
 * being handed out by an endpoint that had no rate limit either — so the service's
 * own 40 MB cap, which does not run until the renderer, was the only thing between
 * a caller and an arbitrarily large object in the operator's bucket.
 *
 * The signed fields must precede the file in the form. S3 reads the policy as it
 * streams and rejects the request the moment it sees a field it was not signed
 * for; `file` last is not a style choice.
 */
async function send(
  slot: { url: string; fields: Record<string, string> },
  file: File,
): Promise<void> {
  const form = new FormData();
  for (const [name, value] of Object.entries(slot.fields)) form.append(name, value);
  form.append("file", file);

  const response = await fetch(slot.url, { method: "POST", body: form });
  if (!response.ok) {
    // Named as an upload failure rather than folded into "cannot reach the
    // service", because the destination is a different host and the fix is
    // different — most often the bucket's CORS rules. A 403 here with a
    // well-formed request is usually the size condition doing its job.
    throw new Error(`Uploading ${file.name} failed with HTTP ${response.status}`);
  }
}

/**
 * What the dropzone says about size, and what counts as too large.
 *
 * Decimal megabytes, matching how the service states the limit in its own rejection
 * message and how a teacher's operating system reports the file's size. The service
 * keeps the limit a round decimal number for that reason.
 */
function sizeHint(maxBytes: number | undefined): string {
  if (!maxBytes || maxBytes <= 0) return "PDF or image";
  return `Max ${Math.floor(maxBytes / 1e6)}MB`;
}

function tooLarge(file: File, maxBytes: number | undefined): boolean {
  return Boolean(maxBytes && maxBytes > 0 && file.size > maxBytes);
}

function DropZone({
  slot,
  label,
  file,
  onPick,
  hint,
}: {
  slot: Slot;
  label: string;
  file: File | null;
  onPick: (file: File | null) => void;
  hint: string;
}): React.JSX.Element {
  const [dragging, setDragging] = useState(false);
  const id = `pick-${slot}`;

  return (
    <div
      className="dropzone-slot"
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
      {/*
       * The input is a SIBLING of the label, not a child.
       *
       * `htmlFor` on a label whose own subtree contains the input activates it
       * twice for one click: once because the click landed on the control's
       * label, and again when the label forwards activation to its target. For a
       * checkbox that is harmless. For a file input it opens the picker, then
       * opens a second picker on top of it, and choosing a file appears to do
       * nothing — which is exactly how this presented.
       *
       * This bug was found and fixed once already, then reintroduced here by
       * wrapping the card in the label for convenience. Sibling placement makes
       * it structurally impossible rather than a thing to remember.
       */}
      <input
        className="dropzone-input"
        id={id}
        type="file"
        name={slot}
        accept={ACCEPT}
        onChange={(event) => onPick(event.target.files?.[0] ?? null)}
      />

      <label
        className="dropzone"
        htmlFor={id}
        data-filled={file !== null}
        data-dragging={dragging}
      >
      <span className="dropzone-icon">
        <UploadIcon size={20} />
      </span>
      <span className="dropzone-label">
        Upload <em>{label}</em>
      </span>
      {file === null ? (
        <span className="dropzone-hint">{hint}</span>
      ) : (
        <span className="dropzone-file">{file.name}</span>
      )}
      </label>
    </div>
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
