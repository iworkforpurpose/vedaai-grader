"""Generate TypeScript types from the pydantic models.

Run via ``pnpm --filter @vedaai/contracts codegen``, which Turbo schedules ahead
of every ``typecheck`` and ``build``.

Why this exists: the frontend draws highlight rectangles from geometry the
Python pipeline computes. If the two sides disagree about whether ``y`` grows
downward, whether pages are 0- or 1-indexed, or whether coordinates are
normalized, everything still compiles and every highlight lands in the wrong
place. Generating one side from the other removes the possibility.

Why the emitter is hand-written rather than ``json-schema-to-typescript``: the
schema shapes involved are entirely under our control and narrow, the codegen
path stays in a single language, and we get to emit string-literal unions for
``StrEnum`` instead of bare ``string`` — so a typo in a status value is a
compile error on the frontend rather than a silent no-match at runtime.

The schema is generated in *serialization* mode, not validation mode, because
what the frontend actually receives is serialized output — which includes
computed fields like ``Question.depth`` and ``Mapping.needs_review``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic.json_schema import models_json_schema

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vedaai_contracts import EXPORTED_MODELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_OUT = ROOT / "schema" / "contracts.schema.json"
TS_OUT = ROOT / "dist" / "types.ts"

HEADER = """/**
 * GENERATED FILE — DO NOT EDIT.
 *
 * Produced from the pydantic models in packages/contracts/src/vedaai_contracts
 * by scripts/gen_types.py. Edit the Python models and re-run codegen.
 *
 * The coordinate contract these types encode:
 *   - all box coordinates are normalized [0,1] floats, never pixels
 *   - origin is top-left, y increases downward
 *   - `page` is 0-indexed
 *   - geometry is relative to a page rendered at RENDER_DPI
 */

"""

# JSON Schema primitives → TypeScript.
PRIMITIVES: dict[str, str] = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
}


def ts_name(ref: str) -> str:
    """Turn a ``#/$defs/Foo`` pointer into ``Foo``."""
    return ref.rsplit("/", 1)[-1]


def render_type(node: dict[str, Any]) -> str:
    """Render one schema node as a TypeScript type expression."""
    if "$ref" in node:
        return ts_name(node["$ref"])

    if "const" in node:
        return json.dumps(node["const"])

    # StrEnum and Literal unions.
    if "enum" in node:
        return " | ".join(json.dumps(v) for v in node["enum"])

    # Optionals arrive as anyOf: [X, {type: "null"}].
    for key in ("anyOf", "oneOf"):
        if key in node:
            parts = [render_type(sub) for sub in node[key]]
            # Collapse duplicates while preserving order, so `X | null | null`
            # cannot happen when a nullable field also carries a default.
            seen: list[str] = []
            for p in parts:
                if p not in seen:
                    seen.append(p)
            return " | ".join(seen)

    node_type = node.get("type")

    if node_type == "array":
        items = node.get("items")
        if not items:
            return "unknown[]"
        inner = render_type(items)
        # Parenthesize unions so `A | B[]` cannot be misread.
        return f"({inner})[]" if "|" in inner else f"{inner}[]"

    if node_type == "object":
        extra = node.get("additionalProperties")
        if isinstance(extra, dict):
            return f"Record<string, {render_type(extra)}>"
        if extra is True or extra is None:
            return "Record<string, unknown>"
        return "Record<string, never>"

    if isinstance(node_type, list):
        return " | ".join(PRIMITIVES.get(t, "unknown") for t in node_type)

    if isinstance(node_type, str):
        return PRIMITIVES.get(node_type, "unknown")

    return "unknown"


def render_doc(node: dict[str, Any], indent: str) -> str:
    """Carry a field or model description across into a JSDoc comment.

    The descriptions on these models explain non-obvious invariants — why
    absence claims get suppressed, why cited line IDs are validated. Dropping
    them at the language boundary would leave the frontend with the shape but
    none of the reasoning.
    """
    desc = node.get("description")
    if not desc:
        return ""
    text = " ".join(desc.split())
    if len(text) <= 76:
        return f"{indent}/** {text} */\n"
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 76:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    body = "\n".join(f"{indent} * {ln}" for ln in lines)
    return f"{indent}/**\n{body}\n{indent} */\n"


def render_enum_alias(name: str, node: dict[str, Any]) -> str:
    values = " | ".join(json.dumps(v) for v in node["enum"])
    return f"{render_doc(node, '')}export type {name} = {values};\n"


def render_interface(name: str, node: dict[str, Any]) -> str:
    props: dict[str, Any] = node.get("properties", {})

    out = render_doc(node, "")
    out += f"export interface {name} {{\n"
    if not props:
        out += "  [key: string]: unknown;\n"
    for prop_name, prop in props.items():
        out += render_doc(prop, "  ")
        # Every property is emitted as required, deliberately, even though the
        # serialization schema marks fields-with-defaults as optional. The API
        # serializes whole models — no `exclude_unset`, no `exclude_none` — so a
        # field with a default is always present in the payload. Marking it
        # optional would force the frontend to null-check values that cannot be
        # absent, and would let a genuinely missing field pass unnoticed.
        # Nullability is carried by the type itself (`X | null`), not by `?`.
        out += f"  {prop_name}: {render_type(prop)};\n"
    out += "}\n"
    return out


def build_schema() -> dict[str, Any]:
    _, schema = models_json_schema(
        [(m, "serialization") for m in EXPORTED_MODELS],
        ref_template="#/$defs/{model}",
        title="VedaaiContracts",
    )
    defs: dict[str, Any] = schema.get("$defs", {})
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "VedaaiContracts",
        "$defs": defs,
    }


def emit_typescript(defs: dict[str, Any]) -> str:
    out = HEADER
    # Enums first: interfaces reference them, and reading the value sets before
    # the shapes that use them makes the generated file easier to scan.
    enums = {k: v for k, v in defs.items() if "enum" in v}
    objects = {k: v for k, v in defs.items() if "enum" not in v}

    if enums:
        out += "// ---- value sets ----\n\n"
        for name in sorted(enums):
            out += render_enum_alias(name, enums[name]) + "\n"

    out += "// ---- shapes ----\n\n"
    for name in sorted(objects):
        out += render_interface(name, objects[name]) + "\n"
    return out


def main() -> int:
    schema = build_schema()
    defs = schema["$defs"]

    SCHEMA_OUT.parent.mkdir(parents=True, exist_ok=True)
    TS_OUT.parent.mkdir(parents=True, exist_ok=True)

    SCHEMA_OUT.write_text(json.dumps(schema, indent=2) + "\n")
    TS_OUT.write_text(emit_typescript(defs))

    model_count = len(defs)
    print(f"codegen: {model_count} definitions -> {SCHEMA_OUT.relative_to(ROOT)}")
    print(f"codegen: {model_count} definitions -> {TS_OUT.relative_to(ROOT)}")

    # Type-check the generated file immediately. Generated code that does not
    # compile is worse than no generated code, because the failure surfaces in
    # whichever app imports it rather than here.
    tsc = ROOT.parents[1] / "node_modules" / ".bin" / "tsc"
    if tsc.exists():
        result = subprocess.run(
            [str(tsc), "--noEmit", "--strict", "--skipLibCheck", str(TS_OUT)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("codegen: generated TypeScript does not compile:", file=sys.stderr)
            print(result.stdout or result.stderr, file=sys.stderr)
            return 1
        print("codegen: generated TypeScript compiles under --strict")
    else:
        print("codegen: tsc not installed yet, skipping compile check")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
