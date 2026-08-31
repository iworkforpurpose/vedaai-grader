"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "./api";
import type { ProgressEvent } from "./contracts";

/** What the waiting screen knows about the job, once the stream has said anything. */
export interface Progress {
  /** The service's own words for what it is doing. Null until the first event. */
  message: string | null;
  stage: ProgressEvent["stage"] | null;
  pagesDone: number | null;
  pagesTotal: number | null;
}

const NOTHING_YET: Progress = {
  message: null,
  stage: null,
  pagesDone: null,
  pagesTotal: null,
};

/**
 * Follow a submission's progress stream.
 *
 * The service has published this all along and nothing in the browser was
 * listening, so the waiting screen was guessing on a timer: it named the right
 * stages in the right order and had no idea which one was actually running. On a
 * long paper it would sit on the last line while the pipeline was still in the
 * middle. This replaces the guess with what the worker says about itself.
 *
 * The endpoint replays from the beginning on connect, so arriving late — or
 * reconnecting — still shows what has happened rather than resuming blind.
 *
 * Every failure returns null rather than throwing. No EventSource, a refused
 * connection, a malformed frame: the caller falls back to the timed captions,
 * which is a worse waiting screen and still a waiting screen. Nothing here is
 * load-bearing enough to break a page over.
 */
export function useProgress(submissionId: string | null): Progress {
  const [progress, setProgress] = useState<Progress>(NOTHING_YET);

  useEffect(() => {
    if (!submissionId || typeof EventSource === "undefined") return;

    const source = new EventSource(`${API_BASE}/submissions/${submissionId}/events`);

    source.onmessage = (event: MessageEvent<string>) => {
      try {
        const data = JSON.parse(event.data) as ProgressEvent;
        setProgress({
          message: data.message,
          stage: data.stage,
          pagesDone: data.pages_done,
          pagesTotal: data.pages_total,
        });
      } catch {
        // One unreadable frame is not worth tearing the stream down for; the next
        // one is usually along in a moment, and the caption simply does not move.
      }
    };

    // The server closes the stream when the job ends, which arrives here as an
    // error. There is nothing to retry at that point -- the poll in MapSurface is
    // what notices the submission finished.
    source.onerror = () => source.close();

    return () => source.close();
  }, [submissionId]);

  return progress;
}

/**
 * The line to show under the loader.
 *
 * Page counts are appended rather than replacing the message, because "Reading
 * the handwriting" and "page 2 of 5" answer different questions -- what is
 * happening, and how much of it is left -- and the second is only sometimes
 * known.
 */
export function progressCaption(progress: Progress): string | null {
  const { message, pagesDone, pagesTotal } = progress;
  if (!message) return null;

  const counted = pagesDone !== null && pagesTotal !== null && pagesTotal > 0;
  if (!counted) return message;

  // Several stages name the page in their own words -- "answers.pdf: page 2" --
  // and appending the counter to those produces "page 2 · page 2 of 5". The
  // service is better placed than this is to say which page it means, so where it
  // has already said so, it wins.
  if (/page\s+\d+/i.test(message)) return message;

  return `${message} · page ${pagesDone} of ${pagesTotal}`;
}
