"""Profile the full scan pipeline on sheet 001.

Generates:
  profile.prof     — cProfile binary (for pstats / snakeviz)
  flamegraph.html  — pyinstrument interactive flame graph
"""

import cProfile
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import process_sheet  # noqa: E402

SHEET = ROOT / "sheets" / "001.jpeg"
PROF_OUT = ROOT / "profile.prof"
FLAME_OUT = ROOT / "flamegraph.html"


def main() -> None:
    # ── cProfile (totals) ──────────────────────────────────────────────────
    # Warm-up: load models so timing reflects steady state
    process_sheet(SHEET)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        process_sheet(SHEET)
    pr.disable()

    pr.dump_stats(str(PROF_OUT))
    print(f"cProfile saved → {PROF_OUT}")

    stats = pstats.Stats(str(PROF_OUT), stream=sys.stdout)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    # ── pyinstrument flame graph (single run, wall-clock) ──────────────────
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()
    process_sheet(SHEET)
    profiler.stop()

    FLAME_OUT.write_text(profiler.output_html())
    print(f"\nFlame graph → {FLAME_OUT}")
    print(profiler.output_text(unicode=True, color=False))


if __name__ == "__main__":
    main()
