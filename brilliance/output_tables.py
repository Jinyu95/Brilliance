"""Conservative presentation of line-oriented numeric output, never validation."""
import math
import re
from collections import defaultdict

_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _scalar(token):
    if token in {"True", "False"}:
        return token == "True"
    if _NUMBER.fullmatch(token):
        value = float(token)
        return value if math.isfinite(value) else None
    return None


def output_tables(stdout: str) -> list[dict]:
    groups = defaultdict(list)
    for line in stdout.splitlines()[:5000]:
        tokens = line.split()
        if not tokens or not re.fullmatch(r"[A-Z][A-Z0-9_]*", tokens[0]):
            continue
        row, index, position = {}, 1, 1
        while index < len(tokens) and _scalar(tokens[index]) is not None:
            row[f"Index {position}"] = _scalar(tokens[index])
            index += 1
            position += 1
        fields = 0
        while index + 1 < len(tokens):
            key, value = tokens[index:index+2]
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) or _scalar(value) is None or key in row:
                break
            row[key] = _scalar(value)
            fields += 1
            index += 2
        # Reject partial parses, vectors and prose instead of guessing their columns.
        if index == len(tokens) and fields >= 1:
            groups[(tokens[0], tuple(row))].append(row)
    return [{"title": f"{tag} — reported program values (not validated)",
             "kind": "program_output", "rows": rows[:200]}
            for (tag, _), rows in groups.items() if len(rows) >= 2][:12]
