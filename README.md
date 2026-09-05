# Answer sheet grader

Upload a printed question paper and one student's handwritten answer sheet. Every
question is extracted in printed order, every answer is located on the page, and
clicking a question highlights the exact region of the sheet that answers it.

Answers given out of order, questions left unanswered, writing that answers nothing
on the paper, and answers spanning several pages are all handled explicitly rather
than assumed away. Marking is included, every mark cites the lines it rests on, and
a teacher can change any of it.

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
| Highlight lands on the writing | 88.6% |
| Highlight is ink rather than paper (IoU) | 0.570 |
| Highlight covers the answer region (IoU) | 0.554 |
| Written labels reaching their own line | 100% |
| Blanks not called blank | 1.5% |
| **False "unanswered" rate** | **0.0%** |

A highlight is **the shape a text selection has**: one box per row of writing, each
extended down to meet the row below so a run renders as a single connected shape
with no gap to see, and only the first and last row ragged.

Both simpler shapes were tried and both were wrong in opposite directions. One
rectangle around all the lines reads cleanly and is mostly paper — ruled writing
leaves a gap of about four-fifths of a line between lines, and the last line stops
wherever the sentence stopped, so a box around five lines covers roughly twice the
area of the writing in it. One rectangle per line is tight and unreadable: a
teacher reported ten stripes down a page of ruled paper, each cut to the ragged end
of its line and each clipping the first letter it was meant to mark. The selection
shape removes the gaps that made the stripes and keeps the tightness that made them
worth wanting. Measured back to back on the same set:

| | one rectangle | selection |
|---|---|---|
| Highlight lands on the writing | 55.3% | **88.6%** |
| Highlights missing the writing | 57 of 132 | **13 of 132** |
| Ink IoU | 0.487 | **0.570** |
| Region IoU | 0.685 | 0.554 |
| Answer placement | 98.5% | 98.5% |

The region figure falls, which is the trade named below and made deliberately: that
metric rewards painting the blank paper around the writing, and painting it is the
fault a teacher complains about.

Placement and highlight quality are reported apart because they move for unrelated
reasons: redrawing a highlight cannot send an answer to a different question, and
it swings the combined figure by twenty-five points. The two highlight numbers pull
against each other by construction — a band that reads as one region necessarily
includes the space between its lines, so covering the region better means covering
less ink. This project has now been misled by each of them in turn.

Measured with the scorer the deployed service uses, and `pnpm turbo eval` now
reaches it without being asked. It did not: the eval and gate commands run `uv run`
without `--env-file`, so `OPENAI_API_KEY` was absent unless a developer had exported
it into their shell, the scorer fell back to word overlap, and the report said so on
a line nobody had to read. The harness loads the repository's `.env` itself now, and
a mismatch **fails the run** rather than printing a caption above the numbers it
invalidates. Read the `answer scorer` line anyway.

The same gap hid something larger. The eval package had no `aws` extra, so boto3 was
missing, no recognizer could be built, and four of the gate's nine documents are
scans: all four read nothing and scored zero, reported as a marking catastrophe
rather than as a missing dependency.

**These figures are reproducible within a session and were not across days.** Two
runs at the same commit agree exactly. A figure measured earlier in the work — 96.2%
where the same code now reads 88.6% — could not be reproduced hours later at that
same commit, with the fixtures untouched and the page cache cleared. The scorer
depends on a hosted embedding service, so the golden set is not the closed system it
looks like. Every comparison in this file is therefore measured back to back in one
session, and a number carried over from an earlier one should be re-measured rather
than believed.

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

## Are the marks right

Every figure above measures where an answer is or which question it belongs to.
None of them measures whether the mark is right, and for a long time that number
did not exist. It does now, and it is the gate: nine documents, each with a total a
competent marker could defend written down **before any run**, and it exits non-zero
rather than reporting.

| document | truth (band) | marks |
|---|---|---|
| history | 20 (15-20) | 17 |
| geography | 15 (13-15) | 15 |
| english | 20 (16-20) | 20 |
| economics | 13 (11-13) | **10** |
| physics | 5 (4-7) | 6 |
| math-paper | 15 (14-17) | 14 |
| asap-clean | 3 (3-3) | 3 |
| asap-middling | 2 (2-3) | 3 |
| asap-worst | 3 (3-3) | 3 |

**Eight of nine inside their band.** It was four of nine when the gate could first
see a recognizer, and every one of the five failures was *under* marking.

Two changes account for it and both were measured one at a time.

**The marker was too small a model.** It was `gpt-4o-mini`, chosen because marking
is a short constrained call and the two ways a weak model fails it are already
contained — malformed output is prevented by demanding a schema, invented citations
are refused by validation, so the failure is *no mark* rather than a wrong one.
Both are still true. What was never contained is judgement, and judgement is what
the gate measured: moving to `gpt-4.1` took five documents out of band down to two.
`physics` is the control — the one paper answered badly on purpose, which stays at
6 against a truth of 5, so the larger model is not simply more generous.

**Handwritten mathematics was unreadable, so it was unmarkable.** Answers whose
evidence is a calculation, a formula or a drawing are now read a second time from a
crop of the page. `math-paper` went from 10 to 14 and inside its band. The invariant
holds: the model is handed a rectangle **code** computed and returns a string per
line id, so every box, highlight and citation is exactly what it was before the
call. See "The one design decision" above — a pass that returned boxes would be the
thing this project exists to avoid; a pass that returns text is what OCR already is,
done better where OCR is worst.

The one that remains is **economics**, and it is not a marking fault: the student
labelled their elasticity working `Q4` in the margin, the mislabel wins through the
recognizer, and question 3 gets nothing. Three marks, on the aligner's side of the
line.

### What the open-weight marker changed, and what it costs

The table above was measured on `gpt-4.1`, which is not what this deployment runs
any more. It marks on Groq's `openai/gpt-oss-120b`, which costs roughly a
thirteenth as much per script and has a free tier a pilot fits inside.

Two figures from a full nine-document gate, one sample per question, no rate
limiting:

| marker | documents in band | direction of the misses |
| --- | --- | --- |
| `openai/gpt-oss-120b` | 8 of 9 | - |
| `openai/gpt-oss-20b` | 4 of 9 | under-marking in all five, no false credit |

The smaller model is not a smaller version of the same result. Under-marking is
the safer direction, but two of its five misses are answers a student earned
marks for that scored zero, and a false zero is the worst error this product can
make. It is therefore where marking goes when the larger model's daily budget is
spent, and not before. `provenance` names the model that actually answered, so a
script marked after the switch says so.

### The panel is what buys the accuracy, not the model

The same model, the same nine documents, the only difference being how many
samples vote on each check:

| `openai/gpt-oss-120b` | documents in band |
| --- | --- |
| five samples per question | 8 of 9 |
| one sample per question | 5 of 9 |

Both runs completed with no rate limiting, so this is the panel and nothing else.
All four extra failures at one sample are *under*-marking.

This matters because the obvious response to a tight free tier is to cut the
panel, and that was done here: the deployment shipped one sample to fit inside
200,000 tokens a day. It reads like a cost setting and it is an accuracy setting.
Three documents were traded away without the trade being written down.

So the panel is back to five and the capacity problem is solved where it actually
lives, in the allowance. A marking call is about 2,300 tokens and a nine-document
gate at one sample consumes very nearly a whole Groq day, which is the arithmetic
that makes a single host untenable: at five samples one host affords roughly six
scripts a day.

**The chain, and why more hosts is not the same as more models.** Free tiers
meter per model *and* per provider, so the same weights on a second host is a
second allowance for the same judgement - extra capacity that costs no accuracy.
That is why `clients.FALLBACK_CHAIN` begins with `gpt-oss-120b` twice, on two
hosts, before it reaches anything weaker. Everything below those two entries is a
worse marker and is reached only when the better ones have spent their day.

`GRADER_MODEL` pins one entry and collapses the chain, which is what the eval
harness does: a gate run that began on one model and quietly finished on another
would report a number belonging to neither.

**What is still not measured.** `gemini-3-flash` sits in the chain for capacity -
its free tier is metered in requests per day rather than tokens, which is the
shape marking actually has - but it has never been through the gate, so it is
placed below every measured entry rather than above them on the strength of its
allowance.

**Marks move between identical runs**, so a single pass cannot tell a fix from
noise. Each question is marked by a panel of five sampled independently, and
`--passes 3` takes the median and reports the spread. On `gpt-4.1`, 45 scored
questions over three identical passes: three moved. A single sample at temperature
zero moved four and swung further, so the panel still earns its five calls — the
provider's seed is best-effort, which is the premise it was built on. Placement was
identical on every pass, as it must be: it is deterministic and none of this touches
it.

A mark is a proposal. **A teacher can change any of it**, including on a question
the marker declined, and their number is kept beside the proposal rather than
replacing it — the gap between the two measures the marker on real scripts without
anybody writing truth down first.

Tests: 673 on the API, 109 on the web app, 145 on the eval harness, 46 on the
contracts.

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
`ANTHROPIC_API_KEY`, and so does the second read of handwritten mathematics.

The eval and gate commands read `.env` themselves. They did not, and the figures
they printed were quietly measured by a different scorer than the service uses.

```bash
pnpm turbo test               # unit tests, both languages
pnpm turbo eval               # extraction, mapping, IoU against the golden set
pnpm --filter @vedaai/evals gate            # are the marks right — costs paid calls
pnpm --filter @vedaai/evals gate -- --passes 3   # and are they the same twice
python tooling/scripts/audit_ui.py <submission-id>   # layout faults across 6 viewports
```

The gate is the one that fails rather than reports, and the one to run before a
commit that touches marking. It needs the documents under `data/`, which a clean
clone does not have — see the last of the known limitations.

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

- **Marking still varies between runs, on about one question in fifteen.** A panel
  of five independent samples brought it down from one in eleven and halved the
  worst swing; hosted models are not bit-reproducible and a seed is best-effort.
  What is left is not decode noise — five samples disagreeing consistently is an
  ambiguous question, and the next instrument is check-level logging rather than
  more samples.
- **Progress events do not survive a restart.** The live stream a browser watches
  during a run is per process; the result survives, the running commentary does
  not. A page open across a restart falls back to polling, which is the path it
  already uses.
- **Handwritten mathematics is read twice, and the second read is unmeasured
  against ground truth.** It used to be the limitation a new user met first: the
  aligner found the answer, the highlight landed on it, and the marks were near
  zero because recognition returned things like `Let the Cost of \ apple = A 1
  Orange = 0`. A crop of the answer now goes back through a vision model, and the
  mathematics paper moved from 10 marks to 14 and inside its band. What is *not*
  measured is the character error rate of that second read — the evidence is a
  mark total, which is the outcome that matters and a coarse instrument for the
  step. Five pages with ground-truth transcriptions would settle it and they have
  not been written.
- **An answer spanning a page boundary is re-read only on the page holding most of
  it.** A crop is one image. The rest keeps its first transcription, which is the
  same outcome as not re-reading rather than a worse one.
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
- **The golden-set figures were not reproducible across days.** Two runs at one
  commit agree exactly; a figure measured hours earlier at that same commit did
  not reproduce, with the fixtures untouched and the page cache cleared. The
  scorer depends on a hosted embedding service, so the set is not the closed
  system it looks like. Every comparison published here is measured back to back
  in one session for that reason, and the cause is not yet pinned down.
- **The gate cannot run in CI.** Its documents live under `data/`, which is
  gitignored because they are real student scripts. The generated papers rebuild
  from `tooling/scripts/build_fresh_papers.py`; the ASAP ones need a corpus a
  clean clone does not have. So the one check that measures whether the marks are
  right is a command somebody has to remember to run.

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
