import { describe, expect, it } from "vitest";
import { progressCaption } from "./progress";

const at = (message: string | null, pagesDone: number | null, pagesTotal: number | null) => ({
  message,
  stage: null,
  pagesDone,
  pagesTotal,
});

describe("progressCaption", () => {
  it("says nothing before the first event", () => {
    expect(progressCaption(at(null, null, null))).toBeNull();
  });

  it("appends the page counter when the message has no page in it", () => {
    expect(progressCaption(at("Rendering answers.pdf", 0, 3))).toBe(
      "Rendering answers.pdf · page 0 of 3",
    );
  });

  it("leaves a message that already names its page alone", () => {
    // The real shape of a transcribing event, which would otherwise read
    // "page 1 · page 1 of 2".
    expect(progressCaption(at("theory_a_in_order.pdf: page 1", 1, 2))).toBe(
      "theory_a_in_order.pdf: page 1",
    );
  });

  it("passes a message through when there is no count", () => {
    expect(progressCaption(at("Matching answers to questions", null, null))).toBe(
      "Matching answers to questions",
    );
  });
});
