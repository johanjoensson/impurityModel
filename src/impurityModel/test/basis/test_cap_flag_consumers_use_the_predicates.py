"""Nothing outside ``truncation_flags`` may read a ``truncation_report``'s existence as "cap-bound".

The report exists whenever a cap cycle ran, so ``report is not None`` is the *history* flag; a consumer that
wants the final state calls :func:`is_cap_bound`, one that wants the history :func:`cap_ever_hit`.
"""

import pathlib
import re

ED = pathlib.Path(__file__).resolve().parents[2] / "ed"
BANNED = re.compile(
    r"(truncation_report|\btruncation|\braw_truncation|\bwalk_report)\s+is\s+not\s+None"
    r"|\.get\(\s*[\"']cap_hit[\"']\s*\)"
)
EXEMPT = {"truncation_flags.py"}


def test_no_consumer_reads_report_existence_or_cap_hit_directly():
    offenders = []
    for path in sorted(ED.glob("*.py")):
        if path.name in EXEMPT or path.name.startswith("gf_") or path.name in {"greens_function.py", "selfenergy.py"}:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if BANNED.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "use truncation_flags.is_cap_bound / cap_ever_hit:\n" + "\n".join(offenders)
