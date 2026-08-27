#!/usr/bin/env python3
"""Upload a question paper and an answer sheet, and print the review URL.

A way round the browser entirely. Useful when the file picker misbehaves, and
useful anyway for running several sheets against one paper without clicking.

    tooling/scripts/upload.py samples/programming_lab_set1.pdf \\
        data/samples/student_a_in_order.pdf

Standard library only, so it runs under any Python on the machine without the
project's environment being active.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

API = os.environ.get("API", "http://127.0.0.1:8000")
WEB = os.environ.get("WEB", "http://127.0.0.1:3001")


def multipart(files: dict[str, Path]) -> tuple[bytes, str]:
    """Encode files as a multipart/form-data body."""
    boundary = f"----vedaai{uuid.uuid4().hex}"
    parts: list[bytes] = []

    for field, path in files.items():
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n".encode()
        )
        parts.append(path.read_bytes())
        parts.append(b"\r\n")

    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def summarize(submission: dict) -> None:
    """Print what the pipeline made of the submission, on stderr."""
    questions = submission["questions"]["questions"]
    mapping = submission["mapping"]
    counts = Counter(m["status"] for m in mapping["mappings"])

    print(f"  questions  {len(questions)}", file=sys.stderr)
    print(f"  statuses   {dict(counts)}", file=sys.stderr)
    print(f"  orphans    {len(mapping['orphans'])}", file=sys.stderr)
    for warning in submission["warnings"]:
        print(f"  ! {warning}", file=sys.stderr)
    print(file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question_paper", type=Path)
    parser.add_argument("answer_sheet", type=Path)
    parser.add_argument(
        "--grade",
        action="store_true",
        help="Also propose marks. Without ANTHROPIC_API_KEY this returns the rubric unjudged.",
    )
    args = parser.parse_args()

    for path in (args.question_paper, args.answer_sheet):
        if not path.is_file():
            print(f"no such file: {path}", file=sys.stderr)
            return 1

    body, content_type = multipart(
        {"question_paper": args.question_paper, "answer_sheet": args.answer_sheet}
    )

    print(
        "uploading… handwriting recognition takes about 14s per page",
        file=sys.stderr,
    )
    request = urllib.request.Request(
        f"{API}/submissions", data=body, headers={"Content-Type": content_type}
    )

    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            submission = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        print(f"upload failed: HTTP {error.code}\n{detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as error:
        print(
            f"cannot reach the pipeline service at {API} ({error.reason}). "
            "Start it with: cd apps/api && uv run --extra ocr-local uvicorn "
            "grader.main:app --port 8000",
            file=sys.stderr,
        )
        return 1

    summarize(submission)

    if args.grade:
        grade_request = urllib.request.Request(
            f"{API}/submissions/{submission['submission_id']}/grades", method="POST"
        )
        try:
            with urllib.request.urlopen(grade_request, timeout=900) as response:
                graded = json.load(response)
            marks = graded["grades"]
            print(
                f"  marks      {marks['total_awarded']} of {marks['total_available']}",
                file=sys.stderr,
            )
        except urllib.error.HTTPError as error:
            print(f"  marking refused: {error.read().decode(errors='replace')}", file=sys.stderr)

    print(f"{WEB}/review/{submission['submission_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
