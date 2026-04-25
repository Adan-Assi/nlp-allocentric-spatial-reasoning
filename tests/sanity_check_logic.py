"""
sanity_check_logic.py
Data-free sanity tests for the parts of the improved version that don't need
graph or POI files: direction extraction, the 8-way classifier, and the
direction_matches() compatibility rule.

Run from the repo root:
    python tests/sanity_check_logic.py

If everything passes you'll see "ALL CHECKS PASSED" at the end.
"""

import os
import sys

# Make repo root importable when run as `python tests/sanity_check_logic.py`
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.extraction_utils import extract_rvs_target  # noqa: E402
from src.utils import direction_matches, get_direction_8way  # noqa: E402


# A neutral reference point in NYC. Δlat=+ → north, Δlon=+ → east.
NYC_LAT, NYC_LON = 40.7589, -73.9851


def _check(label: str, actual, expected) -> bool:
    ok = actual == expected
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label}: got={actual!r:<10} expected={expected!r}")
    return ok


def test_direction_extraction() -> int:
    print("\n[1] 8-way direction extraction (context-aware)")
    cases = [
        ("Head northeast to meet me at the cafe", "NE"),
        ("Walk north and meet me at the park", "N"),
        ("Go southwest to find the restaurant", "SW"),
        ("Head northwest to the bank", "NW"),
        # False-positive cases — bare cardinals in named entities are NOT directions:
        ("Meet me at the cafe on East 49th Street", None),
        ("Head to North Face on Broadway", None),
        # Other valid frames:
        ("The park is north of the bank", "N"),
        ("Two blocks east of here", "E"),
        ("On my southeast", "SE"),
    ]
    failed = 0
    for text, expected in cases:
        _, _, direction = extract_rvs_target(text)
        if not _check(f"'{text[:50]:50s}'", direction, expected):
            failed += 1
    return failed


def test_direction_8way_sectors() -> int:
    print("\n[2] get_direction_8way() compass sectors")
    sectors = [
        ((+0.01, 0), "N"),
        ((+0.01, +0.01), "NE"),
        ((0, +0.01), "E"),
        ((-0.01, +0.01), "SE"),
        ((-0.01, 0), "S"),
        ((-0.01, -0.01), "SW"),
        ((0, -0.01), "W"),
        ((+0.01, -0.01), "NW"),
    ]
    failed = 0
    for (dlat, dlon), expected in sectors:
        got = get_direction_8way(NYC_LAT, NYC_LON, NYC_LAT + dlat, NYC_LON + dlon)
        if not _check(f"Δlat={dlat:+.2f} Δlon={dlon:+.2f}", got, expected):
            failed += 1
    return failed


def test_direction_matches() -> int:
    print("\n[3] direction_matches() compatibility (cardinals coarse, intercardinals exact)")
    cases = [
        # (actual, target, expected)
        ("NE", "N", True),
        ("NW", "N", True),
        ("SE", "N", False),
        ("NE", "NE", True),
        ("N", "NE", False),  # intercardinal target requires exact match
        ("NE", "NW", False),
        ("S", "S", True),
        ("SW", "S", True),
        ("E", "W", False),
        ("", "N", True),     # empty actual → no constraint
        ("N", "", True),     # empty target → no constraint
    ]
    failed = 0
    for actual, target, expected in cases:
        got = direction_matches(actual, target)
        if not _check(f"actual={actual!r:5s} target={target!r:5s}", got, expected):
            failed += 1
    return failed


def test_extraction_returns_three_part_tuple() -> int:
    print("\n[4] extract_rvs_target() always returns (category, noun, direction)")
    samples = [
        "Head northeast to meet me at the cafe",
        "",
        "qq",
        "Just a totally non-spatial sentence about cats.",
        "Meet me at the bank on Broadway",
    ]
    failed = 0
    for text in samples:
        result = extract_rvs_target(text)
        ok = isinstance(result, tuple) and len(result) == 3
        if not _check(f"len(result)==3 for {text!r:50s}", ok, True):
            failed += 1
    return failed


def main() -> int:
    failures = 0
    for fn in (
        test_direction_extraction,
        test_direction_8way_sectors,
        test_direction_matches,
        test_extraction_returns_three_part_tuple,
    ):
        failures += fn()

    print("\n" + "=" * 60)
    if failures == 0:
        print("✅ ALL CHECKS PASSED")
        return 0
    print(f"❌ {failures} CHECK(S) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
