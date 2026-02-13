from pinsheet_scanner.classify import pins_from_diagram
from pinsheet_scanner.detect import (
    Detection,
    detect_pin_diagrams,
    load_model,
    sort_detections,
)
from pinsheet_scanner.pipeline import SheetResult, ThrowResult, process_sheet


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
        "--model",
        type=Path,
        default=None,
        help="Path to a trained YOLO model (.pt). Defaults to models/pin_diagram.pt in the project root.",
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
        confidence=args.confidence,
    )

    print(f"Detected {len(result.throws)} throws across {result.columns} columns\n")

    for throw in result.throws:
        pins = "".join(str(p) for p in throw.pins_down)
        print(
            f"  Col {throw.column:>2} | "
            f"Row {throw.row:>2} | "
            f"Score {throw.score:>1} | "
            f"Pins {pins} | "
            f"Conf {throw.confidence:.2f}"
        )

    print(f"\nTotal pins knocked down: {result.total_pins}")


__all__ = [
    "Detection",
    "SheetResult",
    "ThrowResult",
    "detect_pin_diagrams",
    "load_model",
    "pins_from_diagram",
    "process_sheet",
    "sort_detections",
    "main",
]
