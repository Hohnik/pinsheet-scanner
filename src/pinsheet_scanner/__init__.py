from pinsheet_scanner.classify import (
    classify_pins,
    classify_pins_batch,
    classify_pins_batch_with_confidence,
    classify_pins_with_confidence,
    load_classifier,
)
from pinsheet_scanner.detect import (
    Detection,
    detect_pin_diagrams,
    load_model,
    sort_detections,
)
from pinsheet_scanner.pipeline import SheetResult, ThrowResult, process_sheet

__all__ = [
    "Detection",
    "SheetResult",
    "ThrowResult",
    "classify_pins",
    "classify_pins_batch",
    "classify_pins_batch_with_confidence",
    "classify_pins_with_confidence",
    "detect_pin_diagrams",
    "load_classifier",
    "load_model",
    "main",
    "process_sheet",
    "sort_detections",
]


def main() -> None:
    """CLI entry point."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Extract 9-pin bowling scores from scanned score sheets."
    )
    parser.add_argument(
        "image", type=Path, help="Path to the scanned score sheet image."
    )
    parser.add_argument(
        "--model", type=Path, default=None, help="YOLO detector weights (.pt)."
    )
    parser.add_argument(
        "--classifier-model",
        type=Path,
        default=None,
        help="CNN classifier weights (.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5).",
    )
    args = parser.parse_args()

    result = process_sheet(
        image_path=args.image,
        model_path=args.model,
        classifier_path=args.classifier_model,
        confidence=args.confidence,
    )

    print(f"Detected {len(result.throws)} throws across {result.columns} columns\n")
    for t in result.throws:
        pins = "".join(str(p) for p in t.pins_down)
        print(
            f"  Col {t.column:>2} | Row {t.row:>2} | Score {t.score:>1} | Pins {pins} | Det {t.confidence:.2f} | Cls {t.classification_confidence:.2f}"
        )
    print(f"\nTotal pins knocked down: {result.total_pins}")
