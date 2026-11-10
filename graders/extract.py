"""Take the answer's kernel source out of the model's text."""
import re

FENCE = re.compile(r"^```[ \t]*([A-Za-z0-9+_-]*)[^\n]*\n(.*?)^```[ \t]*$", re.S | re.M)
TAGS = {"cuda", "cu", "cpp", "c++"}


def extract(text, function):
    """Return (code, status). status is "ok", "parse_error" or "truncated".

    Rule: the last block tagged cuda, cu, cpp or c++; else the last fenced block that
    defines `function`. A final fence that never closes gives "truncated"."""
    if len(re.findall(r"^```", text, re.M)) % 2 == 1:
        return None, "truncated"
    blocks = [(m.group(1).lower(), m.group(2)) for m in FENCE.finditer(text)]
    tagged = [b for tag, b in blocks if tag in TAGS]
    if tagged:
        return tagged[-1], "ok"
    defines = re.compile(r"\b" + re.escape(function) + r"\s*\(")
    named = [b for _, b in blocks if defines.search(b)]
    if named:
        return named[-1], "ok"
    return None, "parse_error"
