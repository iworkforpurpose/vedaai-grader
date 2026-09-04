import { describe, expect, it } from "vitest";

import { slotIsSigned, usableUploadPlan } from "./uploads";

const signed = {
  key: "abc/question_paper.pdf",
  url: "https://bucket.s3.amazonaws.com/",
  fields: { key: "abc/question_paper.pdf", policy: "p", "x-amz-signature": "s" },
};

describe("usableUploadPlan", () => {
  it("accepts a plan whose slots are fully signed", () => {
    expect(
      usableUploadPlan({
        mode: "s3",
        slots: { question_paper: signed, answer_sheet: signed },
      }),
    ).toBe(true);
  });

  it("rejects a slot that has a destination but no signed policy", () => {
    /*
     * The bug this file exists for. The service returned `url` and `key` and
     * omitted `fields`, which looks like a present slot and is not a usable one:
     * the signature and conditions travel as form fields, so the bucket refuses
     * the post. The browser never got that far - it threw on
     * `Object.entries(undefined)` before sending anything, and the first screen
     * reported it could not reach a service that was answering normally.
     */
    const unsigned = { key: signed.key, url: signed.url };
    expect(
      usableUploadPlan({
        mode: "s3",
        slots: { question_paper: unsigned, answer_sheet: signed },
      }),
    ).toBe(false);
  });

  it("rejects an empty policy, which signs nothing", () => {
    expect(slotIsSigned({ ...signed, fields: {} })).toBe(false);
  });

  it("rejects a plan missing one of the two slots", () => {
    expect(usableUploadPlan({ mode: "s3", slots: { question_paper: signed } })).toBe(
      false,
    );
  });

  it("treats a direct plan as unusable for the bucket path", () => {
    // Not a failure. It is the other supported environment, and the caller sends
    // the files through the service instead.
    expect(usableUploadPlan({ mode: "direct" })).toBe(false);
  });

  it("survives a response that is not a plan at all", () => {
    expect(usableUploadPlan(null)).toBe(false);
    expect(usableUploadPlan(undefined)).toBe(false);
    expect(usableUploadPlan({} as never)).toBe(false);
  });
});
