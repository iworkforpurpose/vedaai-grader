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
| Answer placement accuracy | 98.5% |
| Highlight covers the answer region (IoU) | 0.730 |
| Highlight is ink rather than paper (IoU) | 0.524 |
| Written labels reaching their own line | 100% |
| Blanks not called blank | 1.5% |
| **False "unanswered" rate** | **0.0%** |

A highlight is **one band per region of writing** — lines that sit under one another
and overlap horizontally merge into a single rectangle, and writing elsewhere on the
page gets a band of its own. Both other shapes were tried and both were wrong in
opposite directions. One box per page reads cleanly and paints the empty half of a
page of handwritten code. One box per line is tight and unreadable: a teacher
reported ten stripes down a page of ruled paper, each cut to the ragged end of its
line and each clipping the first letter it was meant to mark.

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
actually answer went from 3/8 to **8/8**. Label binding across the scripts that
write question numbers went from 32% to 100%.

**And that corpus stopped finding things.** Every document in it has been fixed
against, which is exactly what stops it being evidence — a set you tune to can only
confirm what it already contains. So four more were written in subjects and layouts
none of the others use (`tooling/scripts/build_fresh_papers.py`,
`run_fresh.py`): history with lettered sub-parts and a source extract, geography
with a figure between a stem and its parts, English with an optional section
answered out of order, economics with marks printed twice and disagreeing. What
each student did is written down beside the paper, so the output is checked against
something decided before the run rather than rationalised after it.

Four papers found three faults the corpus could not express — a heading ending in a
full stop, a sub-part the run barely mentions, a question number the recognizer put
an accent on. All three are fixed and all four papers now read correctly. That is
the loop that works, and it is worth repeating on anything that touches the
aligner rather than running once.

Tests: 566 on the API, 66 on the web app, 106 on the eval harness.

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
`ACCESS_CODE` is unset locally, which leaves the origin open — right for a laptop
and wrong for anything with a public address, so the deploy script refuses to
release without one.
Handwriting recognition needs either `OCR_ENGINE=textract` with AWS credentials, or
`uv sync --extra ocr-local` for the local model. Marking needs `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`.

```bash
pnpm turbo test               # unit tests, both languages
pnpm turbo eval               # extraction, mapping, IoU against the golden set
python tooling/scripts/audit_ui.py <submission-id>   # layout faults across 6 viewports
```

## Deployment

**Live: https://wvqyfdkpl1.execute-api.ap-south-1.amazonaws.com** — behind an access
code. Ask for it, or read `ACCESS_CODE` from `.env`.

The gate is a shared passcode in Next middleware, which is the one place both the
pages and the proxied API sit behind. It is not accounts and does not tell testers
apart; it stops a stored script being readable by anyone who finds the address,
which matters because those scripts are real handwriting. The cookie holds an HMAC
keyed by the code rather than the code, so changing the code revokes every session.
`deploy/deploy.sh` refuses to release without one — a gate that silently fails to
engage is worse than none, because it is believed.

Separately, ingest and re-marking are rate limited per caller. One submission
renders every page, recognises all of them, embeds both documents and calls a
marking model once per question, so the limit is what decides what a stranger with
the URL can cost. It is held in memory: with more than one task a caller would get
the allowance once per task, which is stated rather than solved because the
deployment runs one.

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

## Untrusted input, and what one upload may cost

Both documents arrive from a stranger, and the answer sheet reaches a model as the
output of handwriting recognition — so it can say anything, including "award full
marks". Prompt injection is the top item on OWASP's list for LLM applications, and
the question worth asking is not whether an injection can be written but what it
would buy.

Here, very little, and that is structural rather than lucky. The model has **no
tools and no agency** — it cannot call anything or reach any data but the one
answer it is handed. Its output is a **schema-validated structure**, every mark
must **cite a line that resolves inside that answer's scope**, and marks are
**clamped to what the paper printed**. Pages are **rasterised**, so the PDF text
layer is never read and the hidden-white-text attack does not apply. Verified end
to end: a model told to award 999 marks awards the printed 5, and one citing a line
that does not exist has the whole question refused.

The answer is fenced as data behind a delimiter carrying a **value drawn fresh per
request** — the literature calls this spotlighting — because a constant delimiter
is one a student can write on their sheet to close the fence early. Writing
addressed to the marker is **reported to the teacher** rather than blocked: it
cannot work, but attempting it is misconduct, and the person marking the script is
the one who should decide about it.

The residual, stated rather than hidden: a fully compliant model still gives the
attacker full marks on their own question, in front of a teacher who reviews it.

**The expensive risk was never injection.** Marking is one paid call per question
and nothing connected that to what a caller may upload — sixty pages, forty lines
a page, and a paper crafted so every line parses as a question turns one upload
into thousands of calls. That is OWASP's unbounded consumption, and the mitigation
it asks for is a limit before the work rather than a bill after it: **100 questions
marked per submission**, on top of the per-caller rate limit. Extraction and
location are deliberately uncapped — they cost nothing and are most of what this
does.

No model-based injection detector, deliberately. It would cost money on every
submission, add latency, be injectable itself, and buy little against a blast
radius this size.

## Known limitations

- **Marking varies by about one mark between runs.** Temperature 0 and a fixed seed
  narrowed it; hosted models are not bit-reproducible. Settling it needs
  self-consistency over several samples.
- **Progress events do not survive a restart.** The live stream a browser watches
  during a run is per process; the result survives, the running commentary does
  not. A page open across a restart falls back to polling, which is the path it
  already uses.
- **Handwritten mathematics and code get located but not marked.** This is the
  limitation a new user meets first and the one worth stating loudest. The
  aligner finds the answer and the highlight lands on it; the marks are near zero,
  because recognition returns things like `Let the Cost of \ apple = A 1 Orange
  = 0` and a marker cannot credit what it cannot read. Transcription is the
  ceiling and no aligner change moves it.
- **Unplaced writing is no longer shown to a teacher.** It was, and it was mostly
  noise — seven of eight cards on a real script had no readable text at all, and
  one said `Roll No: Page : 03`. The unassigned-ink total still qualifies every
  absence claim on the page; what was removed is a pile of stray marks presented
  as though it meant something.
- **Concurrency is tested to four at once**, on one Fargate task. Four simultaneous
  submissions complete correctly in 44 seconds against 32 for one. Twenty is
  untested.
- **A submission cannot be deleted.** Student handwriting sits in DynamoDB until
  its seven-day TTL expires it.
- **Real-page recall is unmeasured**, as above.

## Layout

```
apps/web        Next.js app: upload, waiting, and the mapping surface
apps/api        FastAPI + the pipeline
packages/contracts   the coordinate contract, generated into both languages
packages/evals       golden set, synthetic generator, metrics
tooling/scripts      sample and paper builders, the corpus loop, engine comparison
deploy               container, IAM, and the deploy script
```

The geometry, hit-testing and mapping logic in the web app lives in `lib/`, with no
React and no DOM, which is why the interface could be rebuilt against a design
without touching any of it.
