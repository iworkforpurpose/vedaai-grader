"""Marking an answer, and the public surface of how it is done.

The work lives in four modules, split by what each is responsible for:

* `schemas`   the shape a marker must answer in
* `panel`     marking a question several times and reconciling the answers
* `assembly`  turning a judgement into a grade a teacher can check
* `graders`   the markers themselves, and how one is chosen

This module re-exports them so `grading.engine` stays the one import callers and
tests need, and so the names they already reach for keep working.

The decision worth restating here, because it shapes all four: this service does
not ask a model for a mark and record it. It asks for a judgement in a fixed
schema, checks every citation resolves to a line that exists, defers what a panel
cannot agree on, and records which host and model answered. A mark is only
checkable if you know what made it and can see what it was based on.
"""

from __future__ import annotations

# Re-exports, written in the redundant `X as X` form so that a linter reads them
# as a public surface rather than as unused imports. The private names are here
# because the tests reach for them: they exercise the panel and the assembly
# directly, and `grading.engine` is the one import they should need.
from .assembly import _cited_share as _cited_share
from .assembly import _confidence as _confidence
from .assembly import _needs_a_person as _needs_a_person
from .assembly import _point_for as _point_for
from .assembly import _tool_input as _tool_input
from .assembly import _unjudged_points as _unjudged_points
from .assembly import assemble as assemble
from .assembly import assemble_checks as assemble_checks
from .graders import DEFAULT_MODELS as DEFAULT_MODELS
from .graders import GRADER_PROVIDER as GRADER_PROVIDER
from .graders import MAX_ANSWER_CHARS as MAX_ANSWER_CHARS
from .graders import Claude as Claude
from .graders import Grader as Grader
from .graders import GraderUnavailable as GraderUnavailable
from .graders import LocalChecks as LocalChecks
from .graders import OpenAIGrader as OpenAIGrader
from .graders import RubricOnly as RubricOnly
from .graders import _local_checks_preferred as _local_checks_preferred
from .graders import select_grader as select_grader
from .panel import GRADER_SEED as GRADER_SEED
from .panel import MARK_AGREEMENT as MARK_AGREEMENT
from .panel import MARK_SAMPLES as MARK_SAMPLES
from .panel import MARK_TEMPERATURE as MARK_TEMPERATURE
from .panel import _agreement as _agreement
from .panel import _default_temperature as _default_temperature
from .panel import _panel as _panel
from .panel import _seeds as _seeds
from .panel import median_sample as median_sample
from .panel import vote_checks as vote_checks
from .schemas import CHECK_JUDGEMENT_SCHEMA as CHECK_JUDGEMENT_SCHEMA
from .schemas import JUDGEMENT_SCHEMA as JUDGEMENT_SCHEMA
from .schemas import STRICT_JUDGEMENT_SCHEMA as STRICT_JUDGEMENT_SCHEMA
