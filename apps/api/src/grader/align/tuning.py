"""Every number the aligner weighs a decision with.

Gathered in one place because these are what get changed, and changing one in
isolation is how a signal quietly stops mattering. Each carries the reason it
holds its value. Where a value was settled by measurement, the measurement is
named; where it guards against a specific failure, that failure is named.
"""

from __future__ import annotations

#: Weights on the match score.
#:
#: The label term dominates by design: a student naming the question states the
#: answer directly, and semantic drift must not outvote a confirmed label.
#:
#: Order is a tiebreaker, not a signal. The prior spans [0, 1] while observed
#: semantic differences on real prose span about +/-0.22, so at 0.3 position
#: outvoted meaning and handed answers to the wrong question. The golden set
#: cannot measure this weight at all - there an answer's vocabulary always agrees
#: with its position - so it is set low enough to decide only when semantics is
#: silent, which is what happens on badly recognised handwriting.
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
#: This cost cannot do the job alone, and both directions fail loudly: too dear
#: and sub-parts are reported unanswered rather than sharing the block they were
#: written in, too cheap and blocks are spread onto questions the student skipped.
#: Length is the missing signal - a block holding three sub-answers is roughly
#: three times the length of one - so the base cost stays moderate and
#: `signals._share_support` supplies the evidence.
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


#: Below this combined score a match is not worth making, and the aligner must
#: prefer a gap.
#:
#: Enforced by making the pairing unavailable rather than by clamping the score
#: to this value: a clamped -0.35 beats SKIP_QUESTION at -0.40, so clamping
#: guarantees the dynamic program can never choose a gap - the exact opposite of
#: the intent. The golden set cannot catch that, because it has roughly as many
#: blocks as questions and so no surplus to place; a real sheet carries several
#: times more writing than the paper has questions, and every question then
#: receives an answer whether or not anything addressed it.
MATCH_MINIMUM = -0.35


#: How much worse than a block's own best a question may score and still claim it.
#:
#: A ratio rather than a difference, so it means the same thing whatever scale the
#: scorer works on. Without it the aligner knows only "best available": when the
#: right question is already taken, the second page of a two-page answer lands on
#: an unrelated question scoring half as much, and is highlighted there.
SETTLE_RATIO = 0.70


#: How far a block's best question must beat the runner-up for the preference to
#: stand in for the relatedness floor.
#:
#: A multiple rather than a difference, because the absolute scores it applies to
#: can be tiny. The same ratio separates 0.1555 from 0.0442 and 0.44 from 0.12; a
#: difference threshold admitting the first would admit almost anything at prose
#: scale.
_PREFERENCE_MARGIN = 2.0


#: Share of substantive ink left unassigned that suppresses absence claims.
#:
#: When this much writing belongs to no block, some answer went unmapped and the
#: system is in no position to tell a teacher a question was left blank.
UNASSIGNED_INK_SUPPRESSES = 0.18


#: How far a block's best question must lead the rest before that preference is
#: treated as a statement rather than a lean.
#:
#: Maximising the whole path and getting each block right are different
#: objectives, and they disagree. Continuing an answer consumes a block without
#: advancing to the next question, so every later question has fewer blocks to
#: draw on; advancing collects a match now and avoids a skip penalty later. Six
#: mediocre matches beat two good ones plus four honest gaps, so the dynamic
#: program is rewarded for inventing answers - four pages that unambiguously
#: answered two questions were spread across six.
#:
#: A lead this wide is evidence rather than a lean, and no downstream bookkeeping
#: should trade it away. Below it the global view is still the better judge, which
#: is why this gates strong cases rather than adding a weight.
DECISIVE_MARGIN = 0.25


#: How far above a scorer's own noise floor a block must score to count as a
#: plausible answer to a question nothing was assigned to.
#:
#: A share of the scorer's band, never an absolute. An absolute value belongs to
#: whichever scorer it was measured on: 0.18 sits below one embedder's 0.30
#: unrelated floor and below every pair another produces, which makes every block
#: plausible for every question. The damage is not a wrong mark but the loss of a
#: distinction - every absence downgrades to `uncertain`, collapsing `unanswered`,
#: `ocr_failed` and `not_required` into one hedge.
#:
#: This is the third place the same scale-mismatch bug has appeared. Thresholds
#: compared against a similarity score belong in the scorer's units.
PLAUSIBLE_SHARE_OF_BAND = 0.20


#: Weight of a label that was written but not confirmed.
#:
#: Well below W_LABEL, because the point of confirmation is that an unverified
#: label has not earned that authority. Well above zero, because a student writing
#: a question number is still the most direct statement of intent available.
W_LABEL_UNCONFIRMED = 0.9

#: Weight of a label the evidence contradicts. Small and still positive: a
#: disputed label is usually a slip rather than a fiction, so it remains faint
#: evidence for the question it names.
W_LABEL_DISPUTED = 0.25


#: A block shorter than this is a fragment, not an answer: a trailing word that
#: segmentation cut off, a stray mark. Measured on a real script, "there." - the
#: last word of an answer, separated from it by a page break.
_FRAGMENT_MAX_CHARS = 12


#: Vertical space, as a share of page height, that a continuation may leave below
#: the answer it carries on from. About two blank lines of ordinary handwriting: a
#: paragraph break, not a fresh start further down the page.
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
#: paper is written on every other line - at 0.35 every line of every answer
#: became a rectangle of its own, and one answer came out as ten stripes down the
#: page.
#:
#: The horizontal condition stops a band reaching across the page. Writing down
#: the left of a sheet and more down the right are two regions, and one rectangle
#: over both paints the empty middle: measured on handwritten code, a page-wide
#: box covered 0.77 of the sheet to mark 0.28 of writing.
_MERGE_GAP_SHARE = 1.2
_MERGE_OVERLAP_SHARE = 0.15


#: How much two boxes must overlap vertically to be the same row of writing.
#:
#: Recognizers split one visual line into several boxes often - a margin number
#: beside its sentence, a word set slightly higher than its neighbours. Those are
#: one row and must be unioned before any vertical reasoning, or the row below
#: gets its boundary set against half of the row above.
_SAME_ROW_OVERLAP = 0.5


#: The widest space that may sit between two fragments of one row, as a share of
#: the page.
#:
#: Same-row is necessary and not sufficient. A margin number sits a few hundredths
#: of a page from the sentence it labels and belongs to it; the left and right
#: columns of a page of code also share a row, and joining those paints the empty
#: middle. The gap tells them apart, and 0.08 is about one indent - wider than any
#: margin, narrower than any column.
_ROW_GAP_MAX = 0.08
