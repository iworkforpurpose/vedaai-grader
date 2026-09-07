"""The shape a marker must answer in.

Every marking call asks for `response_format: json_schema` with `strict: true`
rather than for prose. That guarantee is what makes a weak model safe here:
malformed output becomes impossible, and an invented citation is refused by
`citations` rather than becoming a mark.

Kept apart from the graders because it is a contract with the provider, not a
policy of ours - all fields required, `additionalProperties: false`, nullables as
unions, which is the subset every OpenAI-shaped host agrees on.
"""

from __future__ import annotations

#: The shape the model must return. Written by hand rather than generated from the
#: contract so that the model is asked for exactly the fields it should decide -
#: it never sets identity, availability, or anything downstream code computes.
JUDGEMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "1-based position of the rubric point being judged.",
                    },
                    "marks_awarded": {"type": "number", "minimum": 0},
                    "satisfied": {"type": "boolean"},
                    "cited_line_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Line IDs from inside the answer fence. Required "
                        "whenever marks are awarded.",
                    },
                    "comment": {"type": "string"},
                    "error": {
                        "type": "string",
                        "description": "Where marks are withheld, what specifically is "
                        "wrong. Naming it is required; 'incomplete' is not naming it.",
                    },
                },
                "required": ["index", "marks_awarded", "satisfied", "cited_line_ids"],
            },
        },
        "feedback": {
            "type": "string",
            "description": "One or two sentences addressed to the student.",
        },
        "uncertain": {
            "type": "boolean",
            "description": "True when the transcription was too damaged to judge fairly.",
        },
    },
    "required": ["points", "uncertain"],
}


#: The response shape when the question came with a bank of binary checks.
#:
#: `met` is a nullable boolean on purpose. True earns the mark, False refuses it,
#: and null is "unsure" - that mark is deferred to the teacher rather than guessed
#: in either direction. A scalar `marks_awarded` is deliberately absent: asking for
#: a number is what let a fluent wrong answer be talked into three marks, and
#: rubric-conditioned grading is measured to agree with human markers on binary
#: judgements and to degrade as granularity grows.
CHECK_JUDGEMENT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["checks", "feedback", "uncertain"],
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "met", "cited_line_ids", "error"],
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "1-based position of the check being answered.",
                    },
                    "met": {
                        "type": ["boolean", "null"],
                        "description": "true = yes, earns the mark. false = no. "
                        "null = unsure, defer this mark to the teacher.",
                    },
                    "cited_line_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Line IDs evidencing the answer. Required when met "
                        "is true.",
                    },
                    "error": {
                        "type": ["string", "null"],
                        "description": "When met is false, what specifically is missing or "
                        "wrong. Naming it is required.",
                    },
                },
            },
        },
        "feedback": {"type": ["string", "null"]},
        "uncertain": {"type": "boolean"},
    },
}


#: The same judgement, expressed for OpenAI's structured-output mode.
#:
#: A separate schema rather than a shared one, because strict mode is genuinely
#: stricter and the differences are not cosmetic: every property must appear in
#: ``required``, ``additionalProperties`` must be false, and numeric bounds like
#: ``minimum`` are not supported. An optional field is therefore expressed as a
#: nullable type instead of an absent key.
#:
#: Worth the duplication. Strict mode guarantees the response parses and matches,
#: which removes the failure a small model is most likely to produce.
STRICT_JUDGEMENT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["points", "feedback", "uncertain"],
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "marks_awarded", "satisfied", "cited_line_ids",
                             "comment", "error"],
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "1-based position of the rubric point being judged.",
                    },
                    "marks_awarded": {"type": "number"},
                    "satisfied": {"type": "boolean"},
                    "cited_line_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Line IDs copied from inside the answer fence. "
                        "Required whenever marks are awarded.",
                    },
                    "comment": {"type": ["string", "null"]},
                    "error": {
                        "type": ["string", "null"],
                        "description": "Where marks are withheld, what specifically is "
                        "wrong. Naming it is required; 'incomplete' is not naming it.",
                    },
                },
            },
        },
        "feedback": {
            "type": ["string", "null"],
            "description": "One or two sentences addressed to the student.",
        },
        "uncertain": {
            "type": "boolean",
            "description": "True when the transcription was too damaged to judge fairly.",
        },
    },
}
