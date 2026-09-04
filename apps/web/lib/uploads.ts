/**
 * Whether the service's upload plan can actually be used.
 *
 * There are two supported ways to get a file to this service, and they are not a
 * primary and a fallback. Where object storage is configured the browser posts
 * straight to the bucket with a signed form; where it is not, the files go
 * through the service itself. The second is what every deployment without a
 * bucket does, and it is why this check can be quiet rather than loud.
 *
 * The reason it exists at all is that the two halves deploy separately. The
 * service can be new and the browser old, or the reverse, and a slot that has a
 * URL but no signed policy looks present without being usable. The browser met
 * exactly that as `Object.entries(undefined)` inside the upload: it threw before
 * issuing a single request, the caller's catch declined it for not being an
 * upload failure, and the first screen of the product reported that it could not
 * reach a service that was answering normally.
 *
 * So the shape is checked rather than trusted, and an unusable plan takes the
 * other path instead of ending the submission.
 */

export type UploadSlot = {
  key?: string;
  url?: string;
  fields?: Record<string, string>;
};

export type UploadPlan = {
  mode?: string;
  slots?: Record<string, UploadSlot | undefined>;
};

/** The slots a submission needs, both of them, every time. */
export const REQUIRED_SLOTS = ["question_paper", "answer_sheet"] as const;

/**
 * A signed POST is a destination plus a policy, and the policy travels as form
 * fields - the signature, the conditions and the key are all in there. A slot
 * carrying only a URL is one the bucket will refuse, so it is not usable and
 * saying so early is the difference between the other path and no path.
 */
export function slotIsSigned(slot: UploadSlot | undefined): boolean {
  if (!slot) return false;
  if (!slot.url || !slot.key) return false;
  return Boolean(slot.fields) && Object.keys(slot.fields ?? {}).length > 0;
}

export function usableUploadPlan(plan: UploadPlan | null | undefined): boolean {
  if (!plan || plan.mode !== "s3" || !plan.slots) return false;
  return REQUIRED_SLOTS.every((kind) => slotIsSigned(plan.slots?.[kind]));
}
