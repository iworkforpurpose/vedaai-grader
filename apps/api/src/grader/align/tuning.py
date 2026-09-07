"""Every number the aligner weighs a decision with.

Gathered in one place because these are what get changed, and changing one in
isolation is how a signal quietly stops mattering. Each carries the reason it
holds its value; where a value was settled by measurement rather than by
argument, the measurement is named.
"""

from __future__ import annotations

#: Weights on the match score.
#:
#: The label term dominates by design: a student naming the question is stating
#: the answer directly, and no amount of semantic drift should outvote a
#: confirmed label.
#:
#: Order is meant to be weak, so that a real signal beats it — answering out of
#: order is permitted, and position is only a habit. At 0.3 it was not weak: the
#: prior spans [0, 1], so it contributed up to +0.30 while observed semantic
#: deviations on real prose span about ±0.22, which makes position the *dominant*
#: term rather than the tiebreaker the comment claimed.
#:
#: A real script showed the cost. A student's answer beginning "Invasive is a
#: significant word in the article because…" scored +0.218 against the question
#: that asks exactly that, and +0.063 against a later question about invasive
#: species — semantics preferred the right question by more than three times, and
#: the order prior handed the answer to the wrong one anyway.
#:
#: Halved. The golden set does not move at all across a sixfold sweep of this
#: weight, which is itself the finding: there, an answer's vocabulary always agrees
#: with its position, so the prior never has to decide anything and cannot be
#: measured. It earns its place only where semantics is silent — on scripts whose
#: recognition is too damaged to carry meaning — and for that a tiebreaker is
#: enough.
W_LABEL = 3.0


W_SEMANTIC = 1.0


W_ORDER = 0.15


W_LENGTH = 0.2


#: Cost of leaving a question unanswered, and of emitting an orphan block.
#:
#: Asymmetric, and the asymmetry is the point. Unanswered questions are ordinary.
#: An orphan is more often our own segmentation error than a real extra answer,
#: so the aligner should prefer attaching a stray block to a plausible question
#: over declaring it unattached.
SKIP_QUESTION = -0.40


SKIP_BLOCK = -0.80


#: Cost of extending an answer across another block, before evidence.
#: Cheap, because answers spanning blocks and pages is normal.
CONTINUE_BASE = -0.10


#: Cost of one block serving an additional question, before evidence.
#:
#: Tuning this alone could not work, and the two failed attempts show why. At
#: -0.55 the aligner preferred reporting sub-parts unanswered over sharing the
#: block they were plainly written in — false-unanswered claims, the worst error
#: available. At -0.30 it shared blocks onto questions the student had skipped
#: entirely, inventing answers for three optional questions.
#:
#: The missing signal was length. A block holding three sub-answers is roughly
#: three times the length of one, so length is what distinguishes a genuinely
#: merged answer from an unrelated block being spread across questions. The base
#: cost is therefore moderate and ``_share_support`` supplies the evidence.
SHARE_BASE = -0.45


#: Length beyond one question's expectation before sharing looks plausible. A
#: merged block must be substantially longer than a single answer, or there is no
#: second answer inside it to find.
SHARE_LENGTH_FACTOR = 1.4


#: Bonus when the block is long enough to hold another answer, and penalty when
#: it plainly is not. The penalty is what stops a short unrelated block being
#: spread across questions the student never attempted.
SHARE_LENGTH_BONUS = 0.55


SHARE_LENGTH_PENALTY = -0.9


#: Below this combined score a match is not worth making at all, and the aligner
#: must prefer a gap.
#:
#: Enforced by making such a pairing *unavailable*, which is not what the first
#: version did. It clamped low scores up to this value with ``max``, and since a
#: clamped -0.35 beats ``SKIP_QUESTION`` at -0.40, the floor guaranteed the DP
#: could never choose a gap at all — the precise opposite of the intent stated
#: here. On a real script with far more blocks than questions the effect was
#: total: every question received an answer, including ones nothing on the page
#: addressed.
#:
#: The golden set could not see it. There, blocks and questions are roughly equal
#: in number, so the DP has no surplus blocks it must place somewhere; the fault
#: only appears when a page carries several times more writing than the paper has
#: questions, which is the ordinary case for a real answer sheet.
MATCH_MINIMUM = -0.35


#: How much worse than a block's own best a question may score and still be
#: allowed to claim it.
#:
#: A ratio rather than a difference, so it means the same thing whatever scale the
#: measure works on. Seventy per cent: a question scoring less than seven tenths of
#: what the block's best question scores is not a plausible home for it.
#:
#: Found by looking at a review page. A real script answered question 2 across two
#: pages; the first page took question 2, and the second — scoring 0.689 for
#: question 2 and 0.439 for question 5 — was placed on question 5 and highlighted
#: there, under a question it plainly does not answer. The aligner had no notion of
#: "too much worse", only of "best available", and once the right question was
#: taken it settled for half as good.
SETTLE_RATIO = 0.70


#: How far a block's best question must beat the runner-up before the preference
#: counts as clear enough to stand in for the relatedness floor.
#:
#: A multiple rather than a difference, because the absolute scores this applies
#: to are tiny — 0.1555 against 0.0442 on the case it was measured from. A
#: difference threshold that admitted that pair would admit almost anything at
#: prose scale, where the same ratio sits between 0.44 and 0.12.
_PREFERENCE_MARGIN = 2.0


#: Share of substantive ink left unassigned that suppresses absence claims.
#:
#: When this much writing belongs to no block, some answer went unmapped and the
#: system is in no position to tell a teacher a question was left blank.
UNASSIGNED_INK_SUPPRESSES = 0.18


#: How far a block's best question must lead the rest before that preference is
#: treated as a statement rather than a lean.
#:
#: This exists because maximising the whole path and getting each block right are
#: different objectives, and on a real script they disagree. Four pages of genuine
#: handwriting answered two questions: pages one and two answered question 1, pages
#: three and four answered question 2. Semantics said so unambiguously — +0.603 and
#: +0.608 for question 1, +0.525 and +0.556 for question 2, with every runner-up
#: half that or less.
#:
#: The aligner spread them across six questions anyway, and its arithmetic was
#: sound. Continuing an answer consumes a block without advancing to the next
#: question, so every question after it has fewer blocks to draw on; whereas
#: advancing collects a positive match now and avoids a -0.40 skip later. Six
#: mediocre matches beat two good ones plus four honest gaps. The DP was rewarded
#: for inventing answers.
#:
#: A margin this wide is not a lean. When a block prefers one question by this
#: much, that preference is evidence, and no amount of downstream bookkeeping
#: should be able to trade it away. Below it the DP's global view is still the
#: better judge, which is why this is a gate on strong cases and not a new weight.
DECISIVE_MARGIN = 0.25


#: Similarity to any block above which an unassigned question is reported
#: uncertain rather than unanswered.
#:
#: A targeted guard on the worst error this system can make, and it exists because
#: moving the global share and skip costs only traded false-unanswered against
#: false-answered — one knob, two errors, no setting that fixes both.
#:
#: The reasoning is direct: if some writing on the sheet plausibly answers this
#: question, then the honest report is "found writing I could not place", not "the
#: student left this blank". The unassigned-ink check cannot cover this case,
#: because the block may well be assigned — to a different question.
#: How far above a scorer's own noise floor a block must score before it counts
#: as a plausible answer to a question nothing was assigned to.
#:
#: A share of the band rather than an absolute, and this is the third place the
#: same bug has appeared. It was 0.18 flat, which on the hosted embedder sat below
#: its own 0.30 unrelated floor, and on a local embedder — whose unrelated pairs
#: run 0.42 to 0.77 — makes every block plausible for every question. The effect
#: is not a wrong mark but something worse in this product: every absence claim
#: downgrades to `uncertain`, so `unanswered`, `ocr_failed` and `not_required`
#: collapse into one hedge, which is exactly the distinction the four states exist
#: to keep.
#:
#: Caught by the align tests when the default scorer changed, which is the only
#: reason it is not shipping.
PLAUSIBLE_SHARE_OF_BAND = 0.20


#: Weight of a label that was written but not confirmed.
#:
#: Well below W_LABEL, because the whole point of confirmation is that an
#: unverified label has not earned that authority. Well above zero, because a
#: student writing a question number is still the most direct statement of intent
#: available and ignoring it entirely throws away real information.
W_LABEL_UNCONFIRMED = 0.9


#: Weight of a label the evidence contradicts. Small and still positive: a
#: disputed label is usually a slip rather than a fiction, so it remains faint
#: evidence for the question it names.
W_LABEL_DISPUTED = 0.25


#: A block shorter than this is a fragment, not an answer: a trailing word that
#: segmentation cut off, a stray mark. Measured on a real script, "there." — the
#: last word of an answer, separated from it by a page break.
_FRAGMENT_MAX_CHARS = 12


#: Vertical space, as a share of page height, that a continuation may leave below
#: the answer it carries on from. About two blank lines of ordinary handwriting —
#: a paragraph break, not a fresh start further down the page.
_TAIL_MAX_GAP = 0.12


#: How far down a page a continuation may begin when it carries on from the page
#: before. An answer running over the page break resumes at the top of the next
#: one; writing that starts halfway down it is something else.
_TAIL_PAGE_TOP = 0.25


#: How much better a block must fit the question it was placed on before it stops
#: counting as writing that might belong somewhere else.
#:
#: Not a large margin, because the claim it guards is modest: this block has a
#: home, so its faint resemblance to another question is not evidence that
#: question was answered.
_SETTLED_ELSEWHERE = 1.5


#: How far apart two lines may be, and how little they may share horizontally,
#: before they stop being one band of writing.
#:
#: The gap is measured in line heights, and a full one is allowed because ruled
#: paper is written on every other line: on a real script the gap between lines was
#: 0.82 of a line height, and at the old threshold of 0.35 every line of every
#: answer became a rectangle of its own. Question 16 came out as ten stripes down
#: the page, each cut to the ragged end of its line.
#:
#: The horizontal condition is what stops a band reaching across the page. It is
#: kept, loosely: writing down the left of a sheet and more down the right are two
#: regions, and one rectangle over both paints the empty middle. Measured on a page
#: of handwritten code, a single page-wide box covered 0.77 of the sheet to mark
#: 0.28 of writing.
_MERGE_GAP_SHARE = 1.2


_MERGE_OVERLAP_SHARE = 0.15


#: How much two boxes must overlap vertically to be the same row of writing.
#:
#: Recognizers split one visual line into several boxes often — a margin number
#: beside its sentence, a word set slightly higher than its neighbours, a
#: fragment returned separately. Those are one row and must be unioned before any
#: vertical reasoning happens, or the row below gets its boundary set against
#: half of the row above.
_SAME_ROW_OVERLAP = 0.5


#: The widest space that may sit between two fragments of one row, as a share of
#: the page.
#:
#: Same-row is necessary and not sufficient. A margin number sits a few
#: hundredths of a page from the sentence it labels and belongs to it. The left
#: and right columns of a page of handwritten code also share a row, and joining
#: those paints the empty middle — measured, one page-wide box covered 0.77 of a
#: sheet to mark 0.28 of writing. The gap is what tells them apart, and 0.08 is
#: about one indent: wider than any margin, narrower than any column.
_ROW_GAP_MAX = 0.08
