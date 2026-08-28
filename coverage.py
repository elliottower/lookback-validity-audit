#!/usr/bin/env python3
"""Which citations does a paper make claims about, and which of those are pinned?

`verify_claims.py` proves that the quotes you wrote down are real. It cannot tell
you about a claim you never wrote down. This does the other half: it reads a
LaTeX source, finds every citation key used near a number, and reports which of
those have no claim file.

    uv run --with pyyaml coverage.py paper/main.tex claims/

A citation used near a number is where fabrications land, because a number is
specific enough to be wrong and short enough to look like it came from somewhere.
A citation with no number is usually positioning and needs no pin.

Exit code is 1 if any numeric citation is unpinned, so this can gate a build too.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import yaml

CITE = re.compile(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
# A number that is not a year, a section reference, or a LaTeX length.
NUM = re.compile(r"(?<![\w.\\])\d+(?:\.\d+)?\s*\\?%?(?![\w])")
YEAR = re.compile(r"^(1[89]|20|21)\d{2}$")


def sentences(tex: str):
    body = re.sub(r"(?m)^\s*%.*$", "", tex)
    body = body.replace("\n", " ")
    for s in re.split(r"(?<=[.!?])\s+", body):
        yield s


def pinned_keys(claims_dir: pathlib.Path) -> set[str]:
    keys = set()
    for f in sorted(claims_dir.glob("*.yaml")):
        try:
            d = yaml.safe_load(f.read_text()) or {}
        except yaml.YAMLError:
            continue
        cite = (d.get("source") or {}).get("citation")
        if cite:
            keys.add(cite.strip())
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tex", help="LaTeX source of the paper")
    ap.add_argument("claims", help="directory of claim YAML files")
    ap.add_argument("--quiet", action="store_true", help="only print the unpinned list")
    args = ap.parse_args()

    tex = pathlib.Path(args.tex).read_text(errors="ignore")
    pinned = pinned_keys(pathlib.Path(args.claims))

    numeric: dict[str, list[str]] = {}
    plain: set[str] = set()
    for s in sentences(tex):
        keys = [k.strip() for grp in CITE.findall(s) for k in grp.split(",") if k.strip()]
        if not keys:
            continue
        nums = [n.strip() for n in NUM.findall(s) if not YEAR.match(n.strip())]
        for k in keys:
            if nums:
                numeric.setdefault(k, []).extend(nums[:4])
            else:
                plain.add(k)
    plain -= set(numeric)

    unpinned = sorted(set(numeric) - pinned)
    if not args.quiet:
        print(f"citations used with a number : {len(numeric)}")
        print(f"...of which pinned to a quote: {len(set(numeric) & pinned)}")
        print(f"citations with no number     : {len(plain)} (no pin needed)")
        print()
    if unpinned:
        print("UNPINNED NUMERIC CITATIONS -- each of these states a figure nothing checks:")
        for k in unpinned:
            ns = ", ".join(dict.fromkeys(numeric[k]))
            print(f"  {k:34} {ns}")
    else:
        print("every numeric citation is pinned to a verified quote")
    return 1 if unpinned else 0


if __name__ == "__main__":
    sys.exit(main())
