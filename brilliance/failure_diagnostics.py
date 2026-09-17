"""Small, non-classifying summaries of archived execution diagnostics.

An exception's name is not a repair policy.  Read Python's traceback structure
instead of requiring the name to belong to a list (or to end in ``Error``).
The complete stderr and failure locals remain available in the failure archive.
"""

from __future__ import annotations

import re

_TRACEBACK = "Traceback (most recent call last):"
_TERMINAL = re.compile(r"^(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?::[^\r\n]*)?$")
_PLAIN_EXCEPTION = re.compile(
    r"^(?:(?:[A-Za-z_]\w*\.)*[A-Z](?=\w*[a-z])\w*[A-Z]\w*)(?::.*)?$"
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ARCHIVE_TRAILER = re.compile(
    r"(?m)^(?:BRILLIANCE_FAILURE_LOCALS=|Exact archived output directory:|Stdout:|Stderr:)"
)
_NATIVE_FAILURE = re.compile(
    r"(?:fatal (?:python )?error|segmentation fault|access violation|"
    r"signal\s*\(\d+\)|\b(?:LLVM\s+)?ERROR:|process (?:aborted|crashed))",
    re.IGNORECASE,
)


def execution_failure_summary(error: str) -> str:
    """Return the terminal exception or a useful bounded process diagnostic.

    In each traceback the first unindented exception line terminates the
    frames.  Later tracebacks take precedence for chained exceptions; native
    shutdown warnings after that terminal line do not.  An outer transport
    ``RuntimeError`` containing an embedded traceback is therefore not mistaken
    for the actual generated-program exception.

    This function identifies no errors as safe to repair and reads no files.
    """
    diagnostic = _ANSI_ESCAPE.sub("", str(error or "")).strip()
    if not diagnostic:
        return "Calculation could not complete"
    trailer = _ARCHIVE_TRAILER.search(diagnostic)
    if trailer:
        diagnostic = diagnostic[:trailer.start()].rstrip()

    blocks = diagnostic.split(_TRACEBACK)
    for block in reversed(blocks[1:]):
        for line in block.splitlines():
            # Frame source, carets and indented native stacks are not exceptions.
            if line and not line[0].isspace() and _TERMINAL.fullmatch(line):
                return line[:800]

    lines = [line.strip() for line in diagnostic.splitlines() if line.strip()]
    # A direct exception often has no traceback (API validation, subprocess
    # wrappers).  Avoid turning arbitrary standalone log words into exceptions.
    for line in reversed(lines):
        if _PLAIN_EXCEPTION.fullmatch(line) and _TRACEBACK not in line:
            return line[:800]
    for line in lines:
        if _NATIVE_FAILURE.search(line):
            return line[:800]

    if len(blocks) > 1:
        prefix = blocks[0].strip().rstrip(":").strip()
        if prefix:
            return (prefix + " (incomplete traceback; see archived diagnostic)")[:800]
        return "Python traceback is incomplete; see the archived diagnostic"
    return lines[0][:800] if lines else "Calculation could not complete"
