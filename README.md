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
| Answer mapping accuracy | 92.7% |
| Highlight IoU, mean | 0.744 |
| Highlight IoU@0.5 hit rate | 95% |
| **False "unanswered" rate** | **0.0%** |

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

Tests: 384 on the API, 56 on the web app.

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

One Fargate task in `ap-south-1` running both processes behind one origin: Next
serves the browser and proxies `/api/*` to the FastAPI worker on loopback. No CORS,
no second hostname, and no request-body cap on uploads. Only the browser-facing port
is published, so the worker is unreachable from outside by construction rather than
by a security group rule.

See [`deploy/README.md`](deploy/README.md) for what the two IAM roles may do, and
for the scoped `iam:PassRole` the operator needs — the obvious way to fix the error
you get without it is privilege escalation to every role in the account.

Submissions are held in memory. The brief permitted it and it is still the right
call at this scale, but it is why this is a single task rather than a service with
several: a second task would answer questions about submissions it has never heard
of.

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

- **Reassignment has no interface.** The logic exists and is tested, but nothing
  calls it — so a teacher currently cannot correct a wrong mapping. Gradescope
  treats this as core, and so should this.
- **Marking varies by about one mark between runs.** Temperature 0 and a fixed seed
  narrowed it; hosted models are not bit-reproducible. Settling it needs
  self-consistency over several samples.
- **The deployed URL is a task IP over plain HTTP.** It changes on restart, and
  student work travels unencrypted. A load balancer with a certificate is the fix.
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
