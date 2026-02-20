"""cProfile the full scan pipeline on sheet 001, dump .prof + flame graph SVG."""

import cProfile
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import process_sheet  # noqa: E402 (path set above)

SHEET    = ROOT / "sheets" / "001.jpeg"
PROF_OUT = ROOT / "profile.prof"
SVG_OUT  = ROOT / "flamegraph.svg"


def main() -> None:
    # Warm-up: load models + JIT so timing reflects steady state
    process_sheet(SHEET)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):          # 3 reps to smooth noise
        process_sheet(SHEET)
    pr.disable()

    pr.dump_stats(str(PROF_OUT))
    print(f"Profile saved → {PROF_OUT}")

    # Top-30 by cumulative time
    stats = pstats.Stats(str(PROF_OUT), stream=sys.stdout)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(40)

    # Flame graph
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "flameprof", str(PROF_OUT)],
        capture_output=True, text=True,
    )
    SVG_OUT.write_text(result.stdout)
    print(f"\nFlame graph → {SVG_OUT}")


if __name__ == "__main__":
    main()
