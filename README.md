# Answer sheet grader

Upload a printed question paper and one student's handwritten answer sheet. Every
question is extracted in printed order, every answer is located on the page, and
clicking a question highlights the exact region of the sheet that answers it.

Answers given out of order, questions left unanswered, writing that answers nothing
on the paper, and answers spanning several pages are all handled explicitly rather
than assumed away. Marking is included, and every mark cites the lines it rests on.

## The one design decision everything else follows from

**The model never emits coordinates. It emits line ids, and code computes the
geometry.**

This is not a stylistic preference. HG-Bench (arXiv 2606.25491), the published
benchmark for exactly this task, finds no zero-shot vision-language model exceeding
**55.22%** at locating a complete answer, and **48.22%** at step level. OCRBench v2
measures fine-grained localization IoU frequently below 0.2. Asking a model for
bounding boxes caps accuracy on the primary thing this product is judged by.

So geometry comes from two sources that do not depend on a model reading anything
correctly:

1. **Transcription line boxes**, for answers made of text.
2. **An ink mask** built with OpenCV, for answers that are drawings, and for telling
   a blank region apart from one the recognizer failed on.

The useful consequence: highlighting survives transcription failure. A line the
recognizer misreads still has a box.

## Coordinates

One contract, defined once in `packages/contracts` and generated into both
languages: normalized `[0,1]` floats, origin top-left, `page` 0-indexed, relative to
the page **as rendered**. `BBox` is a pydantic model whose validators reject anything
outside that, so invalid geometry fails at construction in a test rather than as a
misplaced rectangle in a browser.

Coordinate-convention drift across a language boundary is the bug class this project
was most likely to ship. Pinning it in one place and generating both sides is the
cheapest defence available.

## Absence is four different claims

A false "unanswered" is the worst error this product can make, because it is the one
claim a teacher acts on without checking. So absence is never a single threshold:

| Status | What it means | Evidence |
|---|---|---|
| `unanswered` | The student left it blank | Ink below threshold **and** no transcribed lines |
| `ocr_failed` | There is writing here we could not read | Ink present, no lines — says "check page N", never "unanswered" |
| `not_required` | Correctly skipped | An unsatisfied OR-group or optional section |
| `pages_missing` | A continuation runs past the last page | Marker or printed pagination exceeds the upload |

Plus a global check: if a substantial amount of ink is assigned to nothing, *every*
absence claim on that submission is downgraded to uncertain. Per-question evidence
can look fine while the page plainly has writing on it.

## Mapping

Two sequences — questions in printed order, answer blocks in document order — and a
monotone Needleman–Wunsch alignment over them, with moves for match, unanswered,
orphan, continuation, and one block shared by several sub-parts.

Out-of-order answers cannot be represented inside a monotone alignment, so labels a
student wrote in the margin become segment boundaries. Those labels are treated as
**hypotheses, not ground truth**: an anchor is confirmed only if it agrees
semantically with the question it claims, or is order-consistent with its
neighbours. Otherwise it contributes a score the alignment can outvote. A student
writing the wrong question number should not be able to hijack the mapping.

## Measured

On a synthetic golden set that generates the graded edge cases in volume:

| | |
|---|---|
| Question extraction F1 | 100% |
| Printed-order accuracy (Kendall τ) | +1.000 |
| Answer placement accuracy | 97.7% |
| Highlight covers the answer region (IoU) | 0.730 |
| Highlight is ink rather than paper (IoU) | 0.524 |
| Written labels reaching their own line | 100% |
| Blanks not called blank | 2.3% |
| **False "unanswered" rate** | **0.0%** |

Placement and highlight quality are reported apart because they move for unrelated
reasons: redrawing a highlight cannot send an answer to a different question, and
it swings the combined figure by twenty-five points. The two highlight numbers pull
against each other by construction — a band that reads as one region necessarily
includes the space between its lines, so covering the region better means covering
less ink. This project has now been misled by each of them in turn.

Measured with the scorer the deployed service uses, which the harness now names in
its own output. It did not always: this package's environment does not install the
embedding client, so every mapping figure published before this line was written
was measured by word overlap while the service scored by meaning. The two disagree
by three points and in opposite directions on individual cases — a fix built and
verified against the wrong one cost three points of accuracy and was reverted. Run
`pnpm turbo eval` and read the `answer scorer` line before believing any of this.

**These figures replace earlier ones, and two of them are lower.** Mapping accuracy
was quoted at 92.7% and highlight IoU at 0.744. Neither was measured against what
the product does.

Ground truth stored an answer as one rectangle per page, and the highlight was
drawn the same way, so a box around four spread-out lines scored a perfect 1.000
while covering sixty per cent blank paper. The metric was rewarding the fault a
teacher would complain about, and a highlight tightened onto the ink would have
scored *worse* — which is the shape of measurement error that hides a defect
instead of finding it. Truth now records the lines as well as the region, and both
are reported: against the writing because that is what a teacher sees, against the
region because HG-Bench is defined that way and its baselines are the only external
comparison there is.

The published numbers were also produced against real papers rather than only
generated ones. A user's Class 9 mathematics paper extracted one question of nine
while the harness reported 100% extraction F1, because the golden set is generated
by the same code that parses it and therefore only ever contained label styles the
parser already handled.

Four cases were added for the styles it could not previously express — a label
printed as a heading with the question below it, a section the paper numbers
`T1`..`Tn`, a lettered instruction block, and numbers written in the margin. The
check that they are worth having is that removing the fixes now moves the numbers:
extraction F1 falls from 100% to 88.2% and mapping from 81.1% to 72.0%. Before, the
same removal changed nothing at all.

**One fault remains outside what the synthetic set can reach, and it is worth
naming.** Synthetic pages are read from their PDF text layer rather than through
recognition — deliberately, so that a mapping regression is not confused with a
recognition one. But a question number written in the margin only becomes a
separate line, sitting a fraction below the line it labels, when a recognizer
reports it that way. That is how the reading-order fault arose and why no synthetic
case reproduces it: reverting that fix leaves every figure here unchanged. The
corpus of real documents is the harness for that class, which is why it exists.

Transcription engines, on five real handwritten pages with ground-truth
transcriptions:

| Engine | Character error rate |
|---|---|
| AWS Textract | **0.027** |
| PaddleOCR (local) | 0.132 |

Read those honestly. The mapping and highlight figures are **synthetic** — they
measure the algorithm, which is geometric and structural, and handwriting realism
does not affect them. They were also measured with the local recognizer, so
Textract's five-fold lower error rate makes them a floor rather than a ceiling.
Recall on real pages is the binding ceiling on everything downstream and needs
ground-truth boxes that nobody has drawn; that work is not done, and no figure here
should be read as if it were.

A second set of numbers is kept precisely because the synthetic ones could not see
what went wrong. Seven documents that have been through the pipeline — including
the ones that failed — are re-run after every change and inspected in a browser
(`tooling/scripts/rerun_corpus.py`, `inspect_corpus.py`, `score_mapping.py`). On the
three whose correct mapping is unambiguous, blocks landing on the question they
actually answer went from 3/8 to 7/8. Label binding across the scripts that write
question numbers went from 32% to 100%, and the share of a highlight that is
actually ink from 56% to 84%.

Tests: 448 on the API, 60 on the web app, 94 on the eval harness.

## Running it

Needs Node 20, pnpm, Python 3.13 and uv.

```bash
pnpm install
cp .env.example .env          # then fill in what you want to use
pnpm turbo codegen            # contracts -> TypeScript
pnpm dev                      # web on :3000, API on :8000
```

Everything works with no keys at all: the question paper is read from its PDF text
layer, and marking degrades to a rubric a teacher fills in rather than failing.
Handwriting recognition needs either `OCR_ENGINE=textract` with AWS credentials, or
`uv sync --extra ocr-local` for the local model. Marking needs `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`.

```bash
pnpm turbo test               # unit tests, both languages
pnpm turbo eval               # extraction, mapping, IoU against the golden set
python tooling/scripts/audit_ui.py <submission-id>   # layout faults across 6 viewports
```

## Deployment

**Live: https://wvqyfdkpl1.execute-api.ap-south-1.amazonaws.com**

One Fargate task in `ap-south-1` running both processes behind one origin: Next
serves the browser and proxies `/api/*` to the FastAPI worker on loopback. No CORS
and no second hostname. Only the browser-facing port is published, so the worker is
unreachable from outside by construction rather than by a security group rule.

In front of it sits an API Gateway HTTP API, for one reason: the task's public IP
changes on every deploy and serves plain HTTP, so the address could not be given to
anyone and student work travelled unencrypted. The gateway is a fixed hostname with
an AWS-managed certificate, no load balancer and no domain — a certificate-bearing
ALB is around ₹1,500 a month, which is not a sensible shape for a test deployment.
The conventional production answer is CloudFront in front of an ALB, and the code
path is identical; `deploy/README.md` records the trade-off.

Two of the gateway's quotas shaped the design, and neither can be raised:

- **A 30-second integration timeout.** Survivable only because ingest moved off the
  request — an upload now answers in about a second and the browser polls. The same
  upload took 30.7s a day earlier, so this would have been impossible.
- **A 10 MB request body**, against documents this service accepts up to 40 MB. So
  uploads do not pass through the gateway at all: the browser asks the service for
  a presigned URL and sends the file straight to object storage, then names the
  keys. The service never carries the bytes, which is also why every host's body
  cap stops being a design constraint rather than being traded against.

See [`deploy/README.md`](deploy/README.md) for what the two IAM roles may do, and
for the scoped `iam:PassRole` the operator needs — the obvious way to fix the error
you get without it is privilege escalation to every role in the account.

Submissions live in a DynamoDB table, compressed, spilling to object storage past
the 400 KB item limit — a measured two-page submission is 140 KiB and the page cap
is sixty. They were held in memory until real testers were invited, on the
reasoning that the brief permitted it; what the brief permits and what a tester
will forgive are different questions, and every push was discarding work in
progress.

The table also carries the expiry, set to seven days to match the lifecycle rule on
the rendered pages. A record outliving its page images would open to a review with
every page blank, which reads as the pipeline losing the work rather than as a link
expiring.

Writes are conditional on the version that was read, so a lost update is a 409 and
a reload rather than a silently discarded correction. Unreachable with one task, and
that is the point — it is a property of the deployment, not of the code, and it is
the assumption a second task would quietly invalidate.

## Deliberate non-goals

Named rather than half-built:

- **MCQ tick detection.** The answer boxes on AQA and Edexcel papers are vector
  graphics, invisible to text recognition.
- **Bilingual deduplication.** CBSE prints English and Hindi, so every question
  appears twice.
- **Handwritten mathematics to LaTeX.**
- **Step-level rubric citation.** HG-Bench puts step-level grounding *below*
  whole-answer grounding, so it is the harder half of the problem dressed as a
  flourish.

## Known limitations

- **Marking varies by about one mark between runs.** Temperature 0 and a fixed seed
  narrowed it; hosted models are not bit-reproducible. Settling it needs
  self-consistency over several samples.
- **Progress events do not survive a restart.** The live stream a browser watches
  during a run is per process; the result survives, the running commentary does
  not. A page open across a restart falls back to polling, which is the path it
  already uses.
- **Orphaned writing has never been seen in the wild.** The unplaced-answer card is
  built and tested, but no sample produces one — not even a programming answer
  sheet paired with a prose paper, which maps everything wrongly rather than
  refusing to map it. So it has only ever been exercised against an injected
  payload.
- **Real-page recall is unmeasured**, as above.

## Layout

```
apps/web        Next.js app: upload, waiting, and the mapping surface
apps/api        FastAPI + the pipeline
packages/contracts   the coordinate contract, generated into both languages
packages/evals       golden set, synthetic generator, metrics
tooling/scripts      sample builders, engine comparison, UI audit
deploy               container, IAM, and the deploy script
```

The geometry, hit-testing and mapping logic in the web app lives in `lib/`, with no
React and no DOM, which is why the interface could be rebuilt against a design
without touching any of it.
