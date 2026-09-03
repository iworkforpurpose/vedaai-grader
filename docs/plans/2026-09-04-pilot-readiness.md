# Pilot readiness: robustness, accuracy, efficiency

> **For agentic workers:** use `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` to work this task by task. Steps use `- [ ]` for tracking.

**Goal:** take this from "works when everything is up" to something a small number of
real teachers can use unattended on real board papers.

**Architecture:** no rewrites. Every change below is either a correction to an existing
decision, a signal that already exists being made visible, or a measurement that closes
a gap the current harness cannot see. The one design invariant is unchanged: **the model
never emits coordinates — it emits line ids and code computes the geometry.**

**Tech stack:** FastAPI + Python 3.13, Next.js 15 + React 19, pydantic contracts
generated into both languages, AWS Textract / S3 / DynamoDB / ECS Fargate, OpenAI for
embeddings, marking and the crop re-read.

## Decisions this plan is built on

Answered before writing:

- **Audience:** real teachers, small pilot. Weight toward robustness, failure visibility
  and ops over elegance.
- **Accuracy scope:** harden what exists *and* broaden to real board papers
  (CBSE / ICSE / AQA / Edexcel layouts).
- **Cost:** not a constraint. Optimise for accuracy; add monitoring so an empty account
  is loud rather than silent.

## Global constraints

- Never weaken the `ACCESS_CODE` gate.
- Never collapse the four absence states (`unanswered`, `ocr_failed`, `not_required`,
  `pages_missing`). A false `unanswered` remains the worst error this product can make.
- Contracts change in `packages/contracts` only, then `pnpm turbo codegen`. Never
  hand-edit `packages/generated`.
- Geometry, hit-testing and mapping logic in the web app stay in `apps/web/lib/`, with no
  React and no DOM.
- Accuracy claims come from a run **in this session**; two numbers being compared come
  from the same session.
- `deploy/` and IAM belong to the owner. This plan proposes infrastructure changes; it
  does not apply them.

---

# Diagnosis

Five parallel reads of the whole codebase, plus a live end-to-end run and a local
reproduction. What follows is what was found, not what was suspected.

## A. The single largest correctness defect, reproduced and measured

When the embedding provider is reachable-but-failing, placement collapses.

| document | no key at all | **key present, provider failing** |
|---|---|---|
| physics | 5 placed, 0 orphans | **3 placed, 2 orphans** |
| history | 6 placed, 0 orphans | **4 placed, 1 orphan** |
| geography | 5 placed, 0 orphans | **4 placed, 1 orphan** |
| english | 4 placed, 0 orphans | **2 placed** |
| economics | 4 placed, 0 orphans | 4 placed, 0 orphans |
| **total** | **24 placed** | **17 placed** |

**Having no key is strictly better than having a broken one.** The live service is in
the broken-key state right now.

The mechanism, verified:

1. `similarity.py:255` — `SemanticSimilarity.unrelated_below = 0.30`, calibrated on the
   embedding scale (unrelated pairs 0.148–0.154, real matches 0.536–0.779).
2. `similarity.py:280-288` — when `_fetch` fails, `score()` silently returns
   `self._fallback.score(a, b)`, a `StrongerOf` trigram value. **`unrelated_below` is a
   class attribute and stays 0.30.**
3. `align.py:888` reads `floor = 0.30`; `align.py:917` marks every pair `unrelated`
   because trigram scores between a question and a two-line answer do not reach 0.30;
   `align.py:945` sets `matrix[i][j] = -inf` across the board.
4. The DP's only legal move on the block axis becomes `skip_block`
   (`align.py:722-729`), and every block lands in `orphans` (`align.py:1367-1375`).

Measured directly: the physics Q3 pair scores **0.385 under both scorers**. The score is
identical. Only the declared floor differs. This is the exact scale-mismatch the
`Similarity` protocol docstring (`similarity.py:52-63`) was written to prevent,
reintroduced through the fallback.

## B. The label path that should have saved it does not exist

`parse_label` rejects a bare number followed by text. Verified against the real parser:

```
'3 A bird can sit on the wire safely because '  -> None
'5 R=V/I = 12/2 = 6 volts'                      -> None
'3'                                             -> None
'3. A bird can sit'                             -> ('3.', ('3',), numeric)
```

So for a student who writes `3` in the margin, **no `Anchor` object is ever created**
(`anchors.py:107-109`). `segment.py:274-277` documents this gap and works around it
geometrically; `anchors.py` has no equivalent.

Three further gaps compound it:

- `_order_consistent` (`anchors.py:278-302`) corroborates anchors against **other
  anchors only**. "This block sits between the block placed on Q2 and the block placed on
  Q4" is evidence the function cannot see.
- `len(known) < 2` (`anchors.py:292-295`) makes every anchor inconsistent when fewer than
  two resolve. One labelled answer can never be confirmed.
- `anchors.py` never reads `unrelated_below`. `_SEMANTIC_CONFIRM = 0.18`
  (`anchors.py:44`) sits **below** the embedding scorer's own unrelated floor of 0.30, so
  a pair the model calls unrelated is confirmed with `W_LABEL = 3.0` authority.
  `_RIVAL_MARGIN = 0.12` (`anchors.py:60`) never fires on trigram scales, so
  `_decide` rule 3 — the only route to `DISPUTED` — is dead code under lexical scoring.

## C. Security, with real student scripts in the pilot

- **`POST /uploads` is unauthenticated and unthrottled** (`routes.py:90-120`; no
  `_enforce`). `uploads.presign` signs only bucket and key (`uploads.py:93-109`) — no
  content-length condition, no content-type condition. Anyone reaching the service can
  mint unlimited presigned PUTs and write arbitrary objects up to S3's 5 GB single-PUT
  limit into the operator's bucket.
- **`uploads.read` loads the whole object into memory with no size check**
  (`uploads.py:112-116`). The `MAX_BYTES` guard runs afterwards, in `render.py:184-187`.
- **`GET /pages/{key:path}` is an arbitrary-object read** (`routes.py:644-667`).
  `S3PageStore._object_key` (`storage.py:122-127`) rejects only `..` and a leading `/`;
  it does not constrain the key to the shape `key_for` produces. The bucket also holds
  `uploads/` (original scripts) and `submissions/` (full payloads). The prefix is the
  only separation and it is operator-configurable. The local `PageStore` is correctly
  guarded with `resolve()` + `is_relative_to`; the S3 one is not equivalent.
- **No authorization on any submission endpoint** (`routes.py:330-641`). The only
  capability is a 48-bit id (`routes.py:214`). Anyone holding one can read a student's
  full transcribed script, reassign answers, overwrite teacher marks and trigger paid
  re-marking.
- **Student page images are served `public, max-age=31536000, immutable`**
  (`routes.py:663-667`) — shared caches are authorized to keep handwritten student work
  for a year, on an endpoint with no auth.
- **The rate-limit key is client-supplied** (`routes.py:69`, `x-client-key` /
  `x-forwarded-for`). Rotating it bypasses both limiters and grows `Throttle._seen`.
- **Hidden PDF text reaches the prompt.** `ocr/native_pdf.py` drops only out-of-rect and
  whitespace-only glyphs. White-on-white text, zero-size fonts, text under an image and
  hidden OCG layers are extracted normally and pasted into the grading prompt by
  `lineindex.numbered_text`.

## D. The product knows it is degraded and never says so

The API emits a full degradation vocabulary. The review screen reads **one** field of it.

| Signal | Produced at | Read by the UI? |
|---|---|---|
| `submission.warnings` | `pipeline.py:434,465,473,479,483`; `routes.py:305,521` | Only `DebugReview.tsx:130` (a route nothing links to) and `MapSurface.tsx:312` (failed runs only, `warnings[0]`) |
| `submission.error` | `pipeline.py:327` | `MapSurface.tsx:312` only |
| `graded_on_partial_text` | contract, per grade | **Never** |
| `graded_by` | `engine.py:583,754` | **Never** |
| `confidence` / `needs_review` | contract, per grade | **Never** |
| `MatchEvidence` | contract, per mapping | **Never** |
| `questions.gaps` | `pipeline.py:211-216` | **Never** |
| `mapping.orphans` | `align.py:1367` | Counted in `review.ts:127`, rendered nowhere. `orphanHighlightByPage` is exported and called from nowhere. |

On the live run this meant: 4 warnings on the payload — including "answers were matched
by wording rather than by meaning" and "answers were not marked: the provider is rate
limiting or the account is out of credit" — and a screen that said
`3 of 6 answered · rubric only` with no explanation.

## E. There is no operational visibility at all

`grep -r "logging|getLogger|logger\.|print(" apps/api/src/grader/` returns **nothing**.
No structured logs, no metrics, no traces, no alarms, no dashboard. `deploy/deploy.sh`
creates a log group and nothing that writes to it beyond uvicorn access lines.

Consequences, all real:

- `/health` (`main.py:84-94`) reports render DPI and contract count and **nothing about
  grading**. It returns 200 with no API key, in which case every submission is marked
  zero by `RubricOnly`. **The deploy goes green with marking dead.** That is exactly what
  happened.
- The ECS health check probes `:8000` (the API) while the published port is `:8080`
  (Next). A dead web process produces no signal in either direction
  (`deploy.sh:433`, `supervisord.conf:26-38`).
- `supervisord.conf:3-7` claims a dead worker brings the task down. There is no
  `[eventlistener]` in the file. It does not.
- No correlation id. `submission_id` never reaches a log line.
- No rollback path. `deploy.sh` exposes no `rollback` verb and `release` only rolls
  forward.

## F. The measurement cannot see the product

- **`pnpm turbo eval` never marks anything.** `runner.py` calls `pipeline.ingest` and
  stops. Every number it prints can be green with the marker entirely broken.
- **The gate defaults to `--engine text-layer`** (`gate.py:236-239`), which its own help
  says "removes recognition from the gate". Production runs Textract.
- **CI runs neither.** `.github/workflows/deploy.yml:70` runs `pnpm turbo test` only.
- **False zeros are computed and not gated.** `metrics.ScoringReport.false_zeros` is
  printed and `failures_for` never reads it — a document can under-mark every question
  and pass on total alone.
- **`failures_for(doc, run, truth)` never uses `truth`** (`gate.py:143`), and with
  `--passes 3` only pass 1 is checked for false credit, denominator and unjudged.
- **The panel is invisible in the payload.** A question marked by one surviving sample of
  five is byte-identical to one marked 5-0 (`engine.py:365-379`).
- **A confident zero has `confidence = 1.0` and `needs_review = False`** on the binary
  path (`engine.py:1152-1159`). The scalar path caps at 0.8; the binary path does not.
- **`credited_unverifiable` points are exempt from citation validation**
  (`engine.py:1120-1121`). A bank where the model sets `needs_material=true` everywhere
  awards full marks with no citations and passes the gate.

## G. Real board papers will break extraction

Verified against the parser:

| Input | Result | Papers affected |
|---|---|---|
| `3 Explain the process` | not a label | CIE, AQA — becomes a continuation of the question above |
| `1.1 Explain the term` | not a label | IB, Edexcel International, most university papers |
| `1 - Define the term` | not a label | dash-numbered papers, OCR that drops a period |
| `No. 5 Explain` | not a label | any non-`Q` prefix, including Hindi `प्र.1` |
| `SECTION-A` (no space) | not a section header | standard on Indian board papers |
| `Calculate sin (30)` | **marks = 30** | any question ending in a bracketed reference |
| `13 [4 marks] State the law` | marks = `None`, and `[4 marks]` stays in the text | marks printed before the text |
| a bare `5` in the right margin | eaten by `_PAGE_NUMBER` | **CBSE / ICSE / AQA marks columns** |

The worst of these is silent and structural: an unmatched section header leaves
`current_section = None`, so Section B question 1 canonicalises to the same qid as
Section A question 1, and `extract.py:241-263` **reopens and merges them** — half the
paper disappears into the other half, and `validate.suspicious` cannot see it because no
duplicate qid remains.

Second worst: `material_lines` is never cleared after use (`extract.py:562-567` returns a
copy and leaves the list intact; reset only on a section header, which usually does not
match). One source extract is attached to **every subsequent question in the paper**.

## H. Robustness gaps that will bite in a pilot

- **A mid-document OCR failure discards the whole document's transcription.**
  `pipeline.py:131-143` sets `engine = None` and `pipeline.py:171-172` returns `None` for
  the index — a throttle on page 58 of 60 throws away 57 successful Textract calls.
- **No timeout or retry on any AWS call.** No `botocore.Config` anywhere in the tree.
  Textract's own `ThrottlingException` — which has a hand-written "retry" message at
  `ocr/textract.py:132-135` — is converted to a terminal `EngineUnavailable` and nothing
  retries.
- **No timeout on any model call.** `AsyncOpenAI()` / `AsyncAnthropic()` are constructed
  bare, so the SDK default of 600 s applies. With `CONCURRENCY = 4` a submission can hang
  for hours at `processing` with no reaper.
- **Blocking S3 on the event loop.** `reread.repair_submission` calls
  `page_store.exists()` and `page_store.read()` synchronously inside an `asyncio.gather`
  (`reread.py:363-368`), and `_apply_marks` is awaited directly on the loop
  (`routes.py:304`). With one uvicorn worker this stalls every other request, the SSE
  stream and page-image serving.
- **The SSE stream closes before marking starts.** `pipeline.ingest` emits `Stage.DONE`,
  the generator returns (`routes.py:683-695`), and then `routes.py:301` sets the status
  back to `processing` for the marking phase. The failure path at `routes.py:272-279`
  emits **no** terminal event at all, so `store.is_finished` stays `False` forever and the
  connection hangs on keepalives against a submission that already failed.
- **A lost conditional write poisons the cache permanently.** `store.put`
  (`store.py:76-98`) does not update `entry.version` when `persistence.save` raises, so
  every subsequent write from that process fails against the same stale version.
- **The spilled-payload path defeats the conditional write.** `persistence.py:196-199`
  writes the gzipped body to a fixed S3 key *before* the conditional `put_item`. A reader
  in between gets the new body with the old version and can then clobber it.
- **The grader client is never closed in production.** `routes.py:530` builds one per
  submission; there is no `aclose` anywhere in `routes.py`. `score_marks.py:151-154`
  closes it; the service does not.
- **The check-bank cache is unbounded, un-keyed on model, and never invalidated.**
  `scheme.py:106,323-326` — the key omits the model and the system prompt, there is no
  TTL or eviction, and it outlives the DynamoDB TTL, holding question text from expired
  submissions in memory indefinitely.
- **Anthropic silently loses binary checks.** `scheme.available()` returns True if
  *either* key is present but `_ask` unconditionally constructs `AsyncOpenAI`
  (`scheme.py:366-368`). With only an Anthropic key the derivation raises, is swallowed,
  and marking falls back to the scalar path the docstrings blame for fluent-nonsense full
  marks — with no warning.

## I. The interface is not usable without a mouse

- **Selecting a question is not keyboard reachable.** `QuestionCard.tsx:69-83` is a
  `<div>` with `onClick`, no `tabIndex`, no `role`, no key handler — while carrying full
  button styling in CSS and a `.q-list` rule that reserves room for a focus ring. This
  regressed from a `<button>`.
- Placing a moved block and clicking the sheet are unreachable for the same reason.
- The `notice` line — which carries every mutation failure, every 409 and 429 — has no
  `role` and no `aria-live`. On narrow screens a mis-click notice renders inside the pane
  that is `inert` and at `opacity: 0`.
- Selection is conveyed by colour with no `aria-selected` or `aria-current`.
- `role="tablist"` with no `aria-controls`, no `tabpanel` role, no roving tabindex and no
  arrow keys.
- No `error.tsx` or `not-found.tsx` anywhere. An expired submission polls forever behind
  "Extracting…" because `MapSurface.tsx:97` swallows a 404 identically to a network blip.

---

# Phase 0 — Do not run a pilot without these

Security and the reproduced correctness defect. Nothing else starts until these land.

### Task 0.1 — Restore credit and make its absence loud

**Files:**
- Modify: `apps/api/src/grader/main.py` (health payload)
- Test: `apps/api/tests/test_health.py`

**Why:** the account is empty. Marking is off, embeddings are off, and the deploy is
green. Credit is a billing action; the *signal* is the engineering work.

- [ ] **Step 1** — failing test: `/health` reports grading readiness.

```python
def test_health_says_whether_marking_can_run(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    body = client.get("/health").json()
    assert body["grading"]["configured"] is False
    assert body["grading"]["engine"] == "rubric_only"
    assert body["similarity"] == "lexical"

def test_health_names_the_model_that_will_mark(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    body = client.get("/health").json()
    assert body["grading"]["configured"] is True
    assert body["grading"]["model"]
    assert body["similarity"] == "semantic"
```

- [ ] **Step 2** — run it, watch it fail on `KeyError: 'grading'`.
- [ ] **Step 3** — add `grading` and `similarity` blocks to the health payload, derived
      from `grading.select_grader()` and `similarity.default_similarity` without making a
      network call.
- [ ] **Step 4** — extend `.github/workflows/deploy.yml` "Report what is live" to assert
      `grading.configured == true`, so a deploy with marking dead is a **red build**.
- [ ] **Step 5** — commit.

**Done when:** `/health` distinguishes "marking will run with gpt-4.1" from "marking will
award zero", and CI fails on the second.

### Task 0.2 — A degraded scorer must not lie about its scale

**Files:**
- Modify: `apps/api/src/grader/answers/similarity.py:250-300`
- Test: `apps/api/tests/test_similarity.py`

**Why:** section A. This is the reproduced bug: 24 placements become 17.

**Interfaces produced:** `SemanticSimilarity.unrelated_below` becomes a property
returning the fallback's floor while degraded.

- [ ] **Step 1** — failing test:

```python
def test_a_degraded_semantic_scorer_reports_the_fallback_scale():
    """The score and the floor must come from the same measure.

    Reproduced cost of getting this wrong: 24 placements across five papers
    become 17, and a paper with no key at all scores better than one with a
    broken key.
    """
    scorer = SemanticSimilarity(embed=_always_fails)
    scorer.score("a question about refraction", "an answer about bending light")

    assert scorer.degraded is True
    assert scorer.unrelated_below == StrongerOf().unrelated_below

def test_an_undegraded_semantic_scorer_keeps_its_own_scale():
    scorer = SemanticSimilarity(embed=_returns_vectors)
    scorer.score("a", "b")

    assert scorer.degraded is False
    assert scorer.unrelated_below == 0.30
```

- [ ] **Step 2** — run, expect `0.30 != 0.0`.
- [ ] **Step 3** — convert `unrelated_below` from a class attribute to a property that
      returns `self._fallback.unrelated_below` when `self.degraded`.
- [ ] **Step 4** — run the reproduction from the diagnosis and confirm placement returns
      to the no-key numbers:

```bash
uv run --extra aws --extra grading python tooling/scripts/repro_degraded.py
# expect: physics 5 placed / 0 orphans, history 6/0, geography 5/0
```

- [ ] **Step 5** — commit.

**Done when:** a dead provider costs marks and costs **no placements**.

### Task 0.3 — Anchors must read the scorer's scale, not a constant

**Files:**
- Modify: `apps/api/src/grader/answers/anchors.py:44,60,367,273`
- Test: `apps/api/tests/test_answers.py`

**Why:** `_SEMANTIC_CONFIRM = 0.18` is below the embedding scorer's own unrelated floor
of 0.30, so an unrelated pair is confirmed with `W_LABEL = 3.0` authority; and
`_RIVAL_MARGIN = 0.12` never fires on trigram scales, so the mislabel defence is dead
code under lexical scoring.

- [ ] **Step 1** — failing tests: a pair scoring below the live scorer's
      `unrelated_below` is never `CONFIRMED` by the semantic route; the rival margin is
      expressed as a share of the scale rather than an absolute.
- [ ] **Step 2** — run, expect confirmation at 0.20 against an embedding scorer.
- [ ] **Step 3** — derive both thresholds from `unrelated_below`:
      confirm at `floor + (1 - floor) * 0.25`, rival margin at `(1 - floor) * 0.15`.
      Keep the constants as the *shape* of the rule, not as absolutes.
- [ ] **Step 4** — `pnpm turbo eval` and the placement reproduction; neither may regress.
- [ ] **Step 5** — commit.

### Task 0.4 — Presigned uploads must be bounded and authenticated

**Files:**
- Modify: `apps/api/src/grader/routes.py:90-120`, `apps/api/src/grader/uploads.py:93-116`
- Test: `apps/api/tests/test_uploads.py`

**Why:** section C. An unauthenticated, unthrottled write primitive against the
operator's bucket, with no size or type condition on the signed URL.

- [ ] **Step 1** — failing tests: `POST /uploads` is throttled; the signed URL carries a
      `ContentLength` range condition capped at `render.MAX_BYTES`; `uploads.read` calls
      `head_object` and refuses an object over the cap **before** downloading it.
- [ ] **Step 2** — run, expect no throttle and an unconditioned URL.
- [ ] **Step 3** — implement: add `_enforce(_INGEST, request, doing="uploads")`; switch
      to `generate_presigned_post` with `Conditions=[["content-length-range", 1,
      MAX_BYTES], ["starts-with", "$Content-Type", ""]]`; add the `head_object` precheck.
- [ ] **Step 4** — run, all pass.
- [ ] **Step 5** — commit.

### Task 0.5 — Page serving must not be an arbitrary object read

**Files:**
- Modify: `apps/api/src/grader/storage.py:116-127`, `apps/api/src/grader/routes.py:644-667`
- Test: `apps/api/tests/test_s3_store.py`

**Why:** section C. `GET /pages/uploads/{id}/answer_sheet.pdf` and
`GET /pages/submissions/{id}.json.gz` are reachable today, served as `image/png`, with no
auth and a one-year public cache directive.

- [ ] **Step 1** — failing tests:

```python
def test_a_key_outside_the_page_namespace_is_refused():
    store = S3PageStore(bucket="b", prefix="pages/")
    for key in ["uploads/abc/answer_sheet.pdf", "submissions/abc.json.gz",
                "../uploads/abc", "abc/../../secrets"]:
        with pytest.raises(ValueError):
            store._object_key(key)

def test_a_real_page_key_is_accepted():
    store = S3PageStore(bucket="b", prefix="pages/")
    assert store._object_key("2bb9b288de04b440/p0000.png").endswith(
        "pages/2bb9b288de04b440/p0000.png"
    )
```

- [ ] **Step 2** — run, expect the first two to pass through.
- [ ] **Step 3** — constrain the key to the shape `key_for` produces with an explicit
      pattern: `^[0-9a-f]{16}/p\d{4}\.png$`. Reject everything else. Change the response
      header to `private, max-age=3600` — student handwriting must not sit in shared
      caches for a year.
- [ ] **Step 4** — run, all pass.
- [ ] **Step 5** — commit.

### Task 0.6 — A submission id must not be the only capability

**Files:**
- Modify: `apps/api/src/grader/routes.py:214`, `apps/web/middleware.ts`
- Test: `apps/api/tests/test_routes.py`

**Why:** 48 bits of entropy is the whole authorization story for reading and writing a
student's script. Full accounts are out of scope for a pilot; two cheap changes are not.

- [ ] **Step 1** — failing test: a submission id is at least 128 bits of entropy, and a
      mutation endpoint refuses a request whose access cookie is absent.
- [ ] **Step 2** — run, expect a 12-hex id and a mutation succeeding without the cookie.
- [ ] **Step 3** — widen the id to `uuid.uuid4().hex` (full 128 bits). Extend the Next
      middleware matcher so the API proxy path enforces the access cookie on **every**
      `/api/*` route, not only pages.
- [ ] **Step 4** — run the live smoke test; confirm the gate still answers 307 on `/`.
- [ ] **Step 5** — commit.

### Task 0.7 — Hidden PDF text must not reach the prompt

**Files:**
- Modify: `apps/api/src/grader/ocr/native_pdf.py:75-95`
- Test: `apps/api/tests/test_transcription.py`

**Why:** the question paper is fully attacker-controlled. White-on-white text, zero-size
fonts and hidden OCG layers are extracted normally and pasted into the grading prompt.

- [ ] **Step 1** — failing test: a PDF with white text on a white background and a
      zero-size-font span yields no lines for that text.
- [ ] **Step 2** — run, expect both to be extracted.
- [ ] **Step 3** — drop spans whose fill colour matches the page background within a
      tolerance, whose font size rounds to zero, or whose OCG layer is not visible. Record
      the count and raise a warning naming it, because a paper that *contains* invisible
      text is itself a finding a teacher should see.
- [ ] **Step 4** — run, all pass.
- [ ] **Step 5** — commit.

---

# Phase 1 — Make failure visible

Nothing in a pilot survives a silent degradation. This phase surfaces signals that
already exist.

### Task 1.1 — Render the warnings the pipeline already produces

**Files:**
- Create: `apps/web/components/RunNotices.tsx`
- Modify: `apps/web/components/MapSurface.tsx:321-494`, `apps/web/lib/review.ts`,
  `apps/web/app/globals.css`
- Test: `apps/web/lib/review.test.ts`

**Why:** section D. This is the defect the live run exposed most sharply.

**Interfaces produced:** `classifyNotices(submission): Notice[]` in `lib/review.ts`, with
`Notice = { severity: "blocking" | "degraded" | "informational", text: string }`.

- [ ] **Step 1** — failing test in `lib/review.test.ts`: a submission carrying
      "the language service could not be reached" classifies as `degraded`; one carrying
      "Answers were not marked" classifies as `blocking`; an orphan count classifies as
      `informational`; a clean submission yields `[]`.
- [ ] **Step 2** — run, expect `classifyNotices is not a function`.
- [ ] **Step 3** — implement the classifier in `lib/review.ts` (pure, no React), and
      `RunNotices` to render it above the question list with `role="status"` for degraded
      and `role="alert"` for blocking.
- [ ] **Step 4** — render it in **both** the processing and complete branches of
      `MapSurface`, not only the failed one.
- [ ] **Step 5** — commit.

**Done when:** the live physics run would have said, on screen: *answers were matched by
wording rather than by meaning*, and *answers were not marked — the provider is out of
credit*.

### Task 1.2 — Show the per-grade signals that exist and are ignored

**Files:**
- Modify: `apps/web/components/QuestionCard.tsx`, `apps/web/lib/review.ts`
- Test: `apps/web/lib/review.test.ts`

**Why:** `graded_on_partial_text`, `graded_by`, `confidence`, `needs_review` and
`MatchEvidence` are all in the payload and none reach the screen. The contract says
explicitly that a grade computed from unreliable text "must be labelled as such rather
than presented at face value".

- [ ] **Step 1** — failing tests for a `gradeCaveats(grade)` helper returning the short
      phrases a teacher needs: "read from damaged handwriting", "marked by
      openai:gpt-4.1", "low confidence — worth checking".
- [ ] **Step 2..5** — implement, render under the score pill when the card is expanded,
      commit.

### Task 1.3 — Show orphans on the sheet

**Files:**
- Modify: `apps/web/components/SheetView.tsx`, `apps/web/components/MapSurface.tsx`
- Test: `apps/web/lib/geometry.test.ts`

**Why:** `orphanHighlightByPage` exists, is tested, and is called from nowhere. Its own
docstring says orphans are "writing a teacher must be able to see and place by hand".
On the live run two of five answers were orphans and the sheet showed nothing.

- [ ] **Step 1** — failing test: orphan boxes render in a distinct tone and are excluded
      from `stackEdges` grouping with the selected answer.
- [ ] **Step 2..5** — implement, wire the existing "place this" flow to them, commit.

### Task 1.4 — Structured logging with a correlation id

**Files:**
- Create: `apps/api/src/grader/observability.py`
- Modify: `apps/api/src/grader/main.py`, `routes.py`, `pipeline.py`, `grading/run.py`,
  `grading/engine.py`, `reread.py`, `answers/similarity.py`
- Test: `apps/api/tests/test_observability.py`

**Why:** section E. There is no logging anywhere in the service. An operator cannot
answer "how many submissions failed to mark today".

**Interfaces produced:** `log_event(event: str, *, submission_id: str | None = None,
**fields)` emitting one JSON object per line to stdout.

- [ ] **Step 1** — failing test: `log_event` emits parseable JSON carrying `event`,
      `submission_id`, a timestamp, and no secret-shaped values.
- [ ] **Step 2** — run, expect the module to be missing.
- [ ] **Step 3** — implement, then emit at every point that today only appends a warning:
      grader unavailable, per-question marking failure, scheme derivation failure, panel
      sample failure, citation refusal, re-read failure, similarity degraded, Textract
      throttled, stage timings per submission.
- [ ] **Step 4** — verify one full local ingest produces a readable timeline.
- [ ] **Step 5** — commit.

### Task 1.5 — Alarms on the signals that matter

**Files:**
- Modify: `deploy/deploy.sh` (a new `alarms` verb), `deploy/README.md`

**Why:** no alarm, no metric filter, no dashboard exists. **Propose only — do not apply.**

- [ ] **Step 1** — add a `deploy/alarms.sh` defining CloudWatch metric filters over the
      JSON logs from 1.4: `grader_unavailable`, `marking_failed`, `similarity_degraded`,
      `textract_throttled`, and alarms on each crossing zero in five minutes.
- [ ] **Step 2** — document the runbook entry for each in `deploy/README.md`: what it
      means, what to check, what to do.
- [ ] **Step 3** — commit. **Hand to the owner to apply.**

### Task 1.6 — Fix the health check that proves the wrong process

**Files:**
- Modify: `deploy/deploy.sh:409,433`, `deploy/supervisord.conf`

**Why:** the container health check probes the API on `:8000` while the published port is
Next on `:8080`. A dead web process is invisible. And `supervisord.conf` claims a dead
worker brings the task down; there is no `[eventlistener]` in the file.

- [ ] **Step 1** — point the health check at `:8080/api/health`, which exercises Next,
      the loopback proxy and the API in one probe.
- [ ] **Step 2** — add the missing supervisord event listener, or replace the claim in
      the comment with what the file actually does.
- [ ] **Step 3** — commit. **Hand to the owner to apply.**

---

# Phase 2 — Robustness

### Task 2.1 — A partial transcription is worth more than none

**Files:** modify `apps/api/src/grader/pipeline.py:117-194`; test `test_transcription.py`

**Why:** a throttle on page 58 of 60 discards 57 successful Textract calls.

- [ ] Failing test: an engine that raises on page 3 of 5 still yields an index containing
      pages 1, 2, 4 and 5, with a warning naming page 3 and `Line.page` still correct.
- [ ] Fix the positional page binding first — `lineindex.build_index` derives the page
      from list position (`lineindex.py:78`) and `TranscribedLine` carries no page number,
      so skipping a page today would shift every later page's boxes onto the wrong image.
      Add the page to `TranscribedLine` and key on it.
- [ ] Then continue past a per-page failure instead of killing the engine.
- [ ] Commit.

### Task 2.2 — Timeouts and retries on every external call

**Files:** modify `ocr/textract.py:82`, `storage.py:138`, `uploads.py:142`,
`persistence.py:144,154`, `grading/engine.py:507,681`, `grading/scheme.py:368`,
`reread.py:241`; test `test_textract.py`

- [ ] Failing test: a Textract `ThrottlingException` is retried with backoff and succeeds
      on the second attempt; a `ValidationException` is not retried.
- [ ] Add a shared `botocore.Config(retries={"mode": "adaptive", "max_attempts": 5},
      connect_timeout=5, read_timeout=30)` to every boto3 client.
- [ ] Give every model client an explicit timeout (60 s) and `max_retries=2`.
- [ ] Wrap `_run_ingest` in an overall deadline so a submission cannot sit at
      `processing` for hours.
- [ ] Commit.

### Task 2.3 — Get blocking I/O off the event loop

**Files:** modify `routes.py:304`, `reread.py:355-370`; test `test_routes.py`

**Why:** with one uvicorn worker, a synchronous S3 read inside `asyncio.gather` stalls
the SSE stream and every other request.

- [ ] Failing test: `repair_submission` fetches each page **once** and does not block the
      loop (assert via a counting page store and an `asyncio.sleep(0)` liveness probe).
- [ ] Memoise the page fetch per page index, move it to `run_in_executor`, and push
      `_apply_marks` to the executor as `pipeline.ingest` already is.
- [ ] Commit.

### Task 2.4 — The progress stream must end when the work ends

**Files:** modify `routes.py:272-320,683-695`, `store.py:199-203`; test `test_store.py`

- [ ] Failing tests: the stream stays open across marking and closes after it; a failure
      in `_run_ingest` emits a terminal `FAILED` event.
- [ ] Move the `DONE` emission after marking; emit `FAILED` on every failure path,
      including the one outside `pipeline.ingest`'s try block.
- [ ] Commit.

### Task 2.5 — A lost conditional write must be recoverable

**Files:** modify `store.py:76-113`, `persistence.py:190-225`; test `test_persistence.py`

- [ ] Failing tests: after a `ConcurrentUpdate` the store re-reads and the next write
      succeeds; a spilled payload is written to a **version-suffixed** key so a reader
      between the two writes cannot see a new body under an old version.
- [ ] Implement; delete the previous body key after the conditional put succeeds.
- [ ] Commit.

### Task 2.6 — Bound the caches and close the clients

**Files:** modify `grading/scheme.py:106,323-338`, `routes.py:498-535`, `store.py:65`,
`throttle.py:76`; test `test_grading.py`

- [ ] Failing tests: the check-bank cache key includes the model and the prompt version;
      the cache evicts at a bounded size; the grader client is closed after
      `_apply_marks`; the throttle refuses an unbounded number of distinct caller keys.
- [ ] Implement. Give `SubmissionStore._entries` an LRU bound — it currently retains every
      submission the process has ever seen, in full.
- [ ] Commit.

### Task 2.7 — Anthropic must not silently lose binary checks

**Files:** modify `grading/scheme.py:305-368`; test `test_grading.py`

**Why:** with only an Anthropic key, `_ask` raises on `AsyncOpenAI()`, `derive` swallows
it, and marking falls back to the scalar path — the one blamed for fluent-nonsense full
marks — with no warning.

- [ ] Failing test: with only `ANTHROPIC_API_KEY`, the bank is derived via Anthropic; if
      it cannot be, a warning names the reason.
- [ ] Implement a provider-aware `_ask`; make a failed derivation raise a named warning
      rather than an anonymous `None`.
- [ ] Commit.

---

# Phase 3 — Accuracy: placement

### Task 3.1 — A bare margin number is a label

**Files:** modify `apps/api/src/grader/questions/numbering.py:243-296`,
`apps/api/src/grader/answers/anchors.py:101-112`; test `test_questions.py`,
`test_answers.py`

**Why:** section B. `3 A bird can sit…` is not a label today, so no anchor exists for the
commonest thing a student writes.

- [ ] Failing tests: `parse_label("3 A bird can sit", allow_bare=True)` resolves;
      `parse_label("150 / 10 = 15", allow_bare=True)` does **not** (arithmetic is not a
      label); the paper's own question set is required to corroborate a bare number.
- [ ] Add an `allow_bare` mode used **only** by anchor detection on the answer sheet,
      never by question-paper parsing, and only where the number matches a question the
      paper actually has — the same corroboration `segment.py` already uses.
- [ ] Re-run the placement reproduction on all five papers; no regression.
- [ ] Commit.

### Task 3.2 — Order-consistency must see placed neighbours

**Files:** modify `apps/api/src/grader/answers/anchors.py:278-302`,
`apps/api/src/grader/align.py:330-360`; test `test_align.py`

**Why:** the evidence "this block sits between the block placed on Q2 and the block placed
on Q4" is invisible to every confirmation route today, and `len(known) < 2` kills the
route entirely when fewer than two anchors resolve.

- [ ] Failing test: a single anchor claiming Q3, between blocks placed on Q2 and Q4, is
      `CONFIRMED`.
- [ ] Give `_order_consistent` the placed assignments as a second corroboration source and
      drop the `< 2` cutoff when placed neighbours bracket the claim.
- [ ] Commit.

### Task 3.3 — Close the economics mislabel

**Files:** modify `apps/api/src/grader/align.py`; test `test_align.py`

**Why:** the last gate failure. The student labels their elasticity working `Q4` in the
margin; the mislabel wins and Q3 gets nothing. Three marks, on every run.

- [ ] Failing test reproducing the economics case from `data/fresh/economics`.
- [ ] With 3.1 and 3.2 landed, re-measure first — the disputed-anchor route
      (`_decide` rule 3) is currently dead code under lexical scoring and may already fix
      this once thresholds are scale-aware.
- [ ] Commit with the gate output in the message.

---

# Phase 4 — Accuracy: real board papers

The parser scores 100% on a golden set generated by the same code that parses it. This
phase replaces that with papers nobody tuned against.

### Task 4.1 — A corpus of real board layouts

**Files:** create `tooling/scripts/build_board_papers.py`,
`packages/evals/marks/{cbse,icse,aqa,edexcel}.json`

**Why:** every fix below needs a paper that fails before it and passes after. Truth is
written **before** the first run, as the existing `marks/*.json` are.

- [ ] Build four papers reproducing the styles verified as broken: CBSE with
      `SECTION-A` and a right-hand marks column; ICSE with `PART I` / `PART II` and
      `(a) (b) (c)` sub-parts; AQA with `0 1` numbering and marks in brackets at the right;
      Edexcel with `1.1` decimal numbering and `(Total for Question 3 = 6 marks)`.
- [ ] Write the expected question list, marks and student totals first.
- [ ] Confirm each one fails today, and record how.
- [ ] Commit the corpus and the failures.

### Task 4.2 — Section headers as papers actually print them

**Files:** modify `questions/furniture.py:33-36,567-570`; test `test_questions.py`

**Why:** the highest-consequence silent failure. An unmatched header leaves
`current_section = None`, Section B's question 1 collides with Section A's, and
`extract.py:241-263` **merges them** — half the paper disappears with no duplicate qid
left for `validate.suspicious` to find.

- [ ] Failing tests for `SECTION-A`, `SECTION A (20 Marks)`, `Section A: Reading`,
      `PART I (Compulsory)`, `SECTION A — Reading`.
- [ ] Widen the pattern; keep `PART` and `SECTION` in distinct namespaces so ICSE's
      `PART I` and `Section A` cannot collide on a letter.
- [ ] **Also add a guard:** a merge into an existing qid must warn rather than proceed
      silently. A silent merge is how half a paper vanishes.
- [ ] Commit.

### Task 4.3 — Marks in a right-hand column

**Files:** modify `questions/furniture.py:176-178`, `questions/extract.py:192-196`;
test `test_questions.py`

**Why:** `_PAGE_NUMBER` eats a bare `5` in the marks column before `parse_label` ever
sees it — and this is how CBSE, ICSE and AQA all print marks.

- [ ] Failing test: a bare number in the right third of the page, level with a question,
      is marks; the same number centred at the page foot is a page number.
- [ ] Use the box, not the text: position decides. Bind the marks box to the question
      whose **first line** it is level with, not to whichever building happens to be open
      — row-ordering currently attaches it to the next question.
- [ ] Commit.

### Task 4.4 — Numbering styles the parser rejects

**Files:** modify `questions/numbering.py:48,54,159-170,243-296`; test `test_questions.py`

- [ ] Failing tests: `1.1`, `1.2.3`, `1 - Define`, `No. 5`, `Item 3`, `(viii)`, `(xviii)`.
- [ ] Extend `_TOKEN_RE` for longer roman numerals; add decimal-dotted paths; add a
      dash terminator; widen `_Q_PREFIX`.
- [ ] Verify no regression on the existing golden set — the extraction F1 of 100% must
      hold.
- [ ] Commit.

### Task 4.5 — Marks that are not marks

**Files:** modify `questions/numbering.py:339`, `questions/extract.py:284-290`;
test `test_questions.py`

**Why:** `Calculate sin (30)` currently reads as a 30-mark question. A wrong denominator
makes every mark on that question wrong at once.

- [ ] Failing tests: `sin (30)`, `Fig (2)`, `Balance the equation. (2)` yield no marks;
      `(3 marks)` and `[4]` do; `13 [4 marks] State the law` yields 4 and strips the
      bracket from the text.
- [ ] Require a marks *word* or a plausible magnitude, and refuse a number that is
      preceded by a function name or `Fig`/`Table`/`Eq`. Handle leading marks.
- [ ] Commit.

### Task 4.6 — Material must not leak into every later question

**Files:** modify `questions/extract.py:556-570`; test `test_questions.py`

**Why:** `material_lines` is never cleared after use, and the reset only happens on a
section header — which usually does not match. One source extract is attached to every
subsequent question in the paper, and each one shows the model a passage it has nothing
to do with.

- [ ] Failing test: a source extract before question 4 is attached to question 4 and
      **not** to questions 5 and 6.
- [ ] Clear on consumption; re-attach only where the question text refers to the source.
- [ ] Commit.

### Task 4.7 — Nesting without indentation

**Files:** modify `questions/extract.py:111-138,356-396`; test `test_questions.py`

**Why:** two-column papers pool `x0` across columns, so every right-column question is
"deeper" than every left-column one; and papers with no sub-part indentation promote
`(b)` to a top-level question with qid `b`, which then receives a **whole** section rate.

- [ ] Failing tests for both shapes.
- [ ] Compute indent levels **per column** using the reading-order column detection that
      already exists; when no indentation signal exists, fall back to label grammar
      (`(a)` under `1.`) rather than to promotion.
- [ ] Commit.

### Task 4.8 — Bilingual papers

**Files:** modify `questions/extract.py:241-263`; test `test_questions.py`

**Why:** CBSE prints every question twice. Devanagari digits pass `str.isdigit()` and
`int()`, so `१` and `1` become different qids and the paper total doubles; if the Hindi
half repeats Latin digits, the merge branch appends the translation to the English text.

The README lists bilingual deduplication as a **deliberate non-goal**. This task does not
change that — it makes the failure *loud* instead of silent.

- [ ] Failing test: a bilingual paper produces a warning naming the duplication and the
      doubled total, rather than a plausible-looking wrong paper.
- [ ] Commit.

---

# Phase 5 — Measurement that cannot go green while broken

### Task 5.1 — The eval harness must run the marking path

**Files:** modify `packages/evals/src/vedaai_evals/runner.py`; test `test_scoring.py`

**Why:** `pnpm turbo eval` never marks. Every number it prints can be green with the
marker entirely broken.

- [ ] Add an opt-in `--mark` mode running `reread` and `grade_submission` exactly as
      `routes._apply_marks` does, and print the marking figures beside the mapping ones.
- [ ] Print the **effective marking configuration** — model, `MARK_SAMPLES`,
      `MARK_TEMPERATURE`, `MARK_AGREEMENT`, `MARK_CHECKS`, `REREAD` — the way the scorer
      line already names the scorer. All of these are read at import and invisible today.
- [ ] Commit.

### Task 5.2 — Gate the metrics that are computed and ignored

**Files:** modify `tooling/scripts/gate.py:143-230`; test manually against the corpus

- [ ] Fail on `false_zeros` above a written threshold — a document can currently
      under-mark every question and pass on total alone.
- [ ] Fail on `citation_rate` below a written threshold.
- [ ] Fail on `incoherent_points`.
- [ ] Check false credit, denominator, unjudged and truth-missing on **every** pass, not
      only pass 1 (`first = run.reports[0]`).
- [ ] Use the `truth` parameter that `failures_for` currently ignores, or delete it.
- [ ] Print the warnings each document produced — the gate has never read them.
- [ ] Commit.

### Task 5.3 — The panel must be visible in the payload

**Files:** modify `packages/contracts/src/vedaai_contracts/grading.py`,
`apps/api/src/grader/grading/engine.py:365-379,1152-1159`; test `test_grading.py`

**Why:** a question marked by one surviving sample of five is byte-identical to one marked
5-0. And on the binary path a unanimous zero has `confidence = 1.0` and
`needs_review = False` — a confident, review-free zero.

- [ ] Add `samples_returned`, `samples_requested` and `deferred_checks` to
      `QuestionGrade`; run codegen.
- [ ] Cap binary-path confidence the way the scalar path already does at 0.8, and treat a
      shrunken panel as needing review.
- [ ] Fail the gate when any question was marked by fewer than a majority of the requested
      panel.
- [ ] Commit.

### Task 5.4 — Close the `credited_unverifiable` hole

**Files:** modify `apps/api/src/grader/grading/engine.py:1120-1121`; test `test_grading.py`

**Why:** points credited because the paper did not supply the material are removed before
`citations.check` runs. A bank where the model sets `needs_material=true` on every check
awards full marks with no citations, `judged=True`, and passes the gate.

- [ ] Failing test: a bank that is entirely `needs_material` produces a grade flagged
      advisory and counted in a new `credited_without_evidence` metric.
- [ ] Fail the gate when that rate exceeds a written threshold.
- [ ] Commit.

### Task 5.5 — Run the gate in CI

**Files:** modify `.github/workflows/deploy.yml`, `tooling/scripts/build_fresh_papers.py`

**Why:** the one check that measures whether marks are right cannot run in CI, because its
documents live under `data/`, which is gitignored — they are real student scripts.

- [ ] Make the four generated papers rebuild deterministically from committed source, so
      CI can regenerate them without the private corpus.
- [ ] Add a `gate` job that runs on `main` with the OpenAI key from secrets, after tests
      and **before** deploy. The ASAP documents stay local.
- [ ] Commit.

### Task 5.6 — Measure the crop re-read against ground truth

**Files:** create `packages/evals/reread-truth/`, modify
`tooling/scripts/score_recognition.py`

**Why:** the re-read moved `math-paper` from 10 marks to 14, which is an outcome measure
and a coarse instrument for the step. Its character error rate is unmeasured.

- [ ] Write ground-truth transcriptions for five handwritten mathematics regions.
- [ ] Report CER before and after the re-read, and make `score_recognition.py` exit
      non-zero on a regression — it currently returns 0 unconditionally.
- [ ] Commit with the numbers in the message.

---

# Phase 6 — The interface

### Task 6.1 — The primary interaction must work from a keyboard

**Files:** modify `apps/web/components/QuestionCard.tsx:69-83`,
`apps/web/components/SheetView.tsx:212-221`; test `apps/web/lib/review.test.ts`

**Why:** the question card is a `<div>` with `onClick` — no `tabIndex`, no `role`, no key
handler — while carrying full button styling and a CSS rule that reserves room for a
focus ring. It regressed from a `<button>`.

- [ ] Restore `<button>` semantics on the card, add `aria-selected`, move focus to the
      sheet region on selection, and make the sheet click target focusable.
- [ ] Add `aria-controls` to the chevron and the tabs; add roving tabindex and arrow keys
      to the tablist.
- [ ] Commit.

### Task 6.2 — Announce what changes

**Files:** modify `apps/web/components/MapSurface.tsx:430`

- [ ] Give the notice line `role="alert"` and `aria-live="assertive"` — it carries every
      mutation failure, every 409 and every 429 and is silent to screen readers today.
- [ ] Render it in the pane the user is looking at; on narrow screens a mis-click notice
      currently renders inside the pane that is `inert` and at `opacity: 0`.
- [ ] Commit.

### Task 6.3 — Error and empty states

**Files:** create `apps/web/app/error.tsx`, `apps/web/app/not-found.tsx`,
`apps/web/app/review/[id]/error.tsx`; modify `apps/web/components/MapSurface.tsx:86-112`

- [ ] Add error and not-found boundaries with the shell intact and a retry.
- [ ] Stop the poll swallowing a 404 or a 401 identically to a network blip — an expired
      submission currently polls forever behind "Extracting…".
- [ ] On a 409, refresh from the server rather than keeping a stale copy that will 409
      again.
- [ ] Commit.

### Task 6.4 — Reconcile optimistic state

**Files:** modify `apps/web/components/MapSurface.tsx:210-296`

**Why:** the revert at `:231` restores a render-time snapshot, silently discarding
anything that landed in between; and there is no request sequencing anywhere, so the last
response wins regardless of which request was issued last.

- [ ] Move every `setSubmission` to the functional form.
- [ ] Add an `AbortController` and a request-id guard to each mutation and to the poll.
- [ ] Clear `citedPoint` when the submission is replaced — it currently holds a rubric
      point from a superseded grade set.
- [ ] Commit.

---

# Phase 7 — Efficiency

Cost is not a constraint, so this phase is about **latency and blast radius**, not spend.

### Task 7.1 — Stop sweeping the similarity grid five times

**Files:** modify `apps/api/src/grader/align.py`

**Why:** `similarity.score` is swept over the full questions × blocks grid five to six
times per submission, including one place (`align.py:626-628`) that recomputes a value
`_score_matrix` already computed and discarded.

- [ ] Compute the raw matrix once, pass it down, and assert with a counting scorer that
      one submission produces exactly one sweep.
- [ ] Commit.

### Task 7.2 — Warm the embedding cache before anchors run

**Files:** modify `apps/api/src/grader/answers/anchors.py:81-139`

**Why:** `align()` pre-warms; `anchors.detect` runs **first** and does not, so
`_outscored_by_a_rival` issues up to one HTTP round trip per question, sequentially,
before alignment begins. Any single failure trips the 60-second outage cooldown and
poisons the whole submission.

- [ ] Warm once with every question and block text, then score.
- [ ] Commit.

### Task 7.3 — Transcribe pages concurrently

**Files:** modify `apps/api/src/grader/pipeline.py:95-169`

**Why:** 60 pages is 60 sequential synchronous Textract round trips, with rendering and
OpenCV work between them so the connection is idle for seconds at a time.

- [ ] Bounded concurrency (4) over pages, preserving page order in the index — which
      requires the page-id fix from Task 2.1.
- [ ] Measure wall clock on the three-page script before and after.
- [ ] Commit.

### Task 7.4 — Cheaper page geometry and reading order

**Files:** modify `apps/api/src/grader/render.py:305-313`,
`apps/api/src/grader/reading_order.py:173`, `apps/api/src/grader/preprocess.py:285-292`

- [ ] `_png_size` fully decodes a PNG to read two integers — read the IHDR header instead.
- [ ] `reading_order._has_columns` does `line not in column` on a list of pydantic models,
      invoking deep equality inside a nested loop. Key on `line_id`.
- [ ] `_skew_angle` performs 65 full affine warps per page; coarse-to-fine search cuts it
      to about 12 with the same resolution.
- [ ] Commit.

### Task 7.5 — Prompt caching

**Files:** modify `apps/api/src/grader/grading/engine.py`, `grading/sampling.py`

**Why:** the system prompt, the question, the printed material and the whole check bank
are re-sent five times per question with no cache marker. `MAX_ANSWER_CHARS` is declared
at `engine.py:110` and never referenced, so a pathological OCR result goes to the model at
full size, five times.

- [ ] Add provider prompt caching on the stable prefix.
- [ ] Enforce `MAX_ANSWER_CHARS`, with a warning when it truncates.
- [ ] Commit.

---

# Acceptance

The plan is done when all of these hold in one session:

| | Now | Target |
|---|---|---|
| Placement with a dead provider | 17 of 24 | **24 of 24** |
| Gate documents in band | 8 of 9 | **9 of 9** |
| Gate documents, board corpus | not built | **4 of 4** |
| Extraction F1, existing golden set | 100% | 100% (no regression) |
| Highlight lands on the writing | 88.6% | ≥ 88% (no regression) |
| False "unanswered" | 0.0% | **0.0%** |
| Degradations visible on screen | 1 of 8 signals | **8 of 8** |
| Unauthenticated write primitives | 1 | **0** |
| Arbitrary-object reads | 1 | **0** |
| Logged events per submission | 0 | **a full stage timeline** |
| Deploy green with marking dead | yes | **no — red build** |
| Primary interaction keyboard reachable | no | **yes** |

## Sequencing

Phase 0 and Phase 1 are the pilot gate. Phase 2 and 3 can proceed in parallel with 4.
Phase 5 should land before 4 finishes, so the board-paper work has something that fails
before it and passes after. Phase 6 and 7 are last and are independently shippable.

One PR per task. Each PR carries the measurement that justifies it.
