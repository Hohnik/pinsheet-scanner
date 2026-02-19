"""Unified CLI for pinsheet-scanner.

Commands
--------
scan            Scan a score sheet and print results.
train           K-fold cross-validate and retrain the CNN classifier.
train-detector  Train the YOLO pin-diagram detector.
tune            Hyperparameter search with Optuna.
extract         Extract and classify pin-diagram crops from a sheet.
label           Open the browser labeling UI.
accuracy        Validate CNN predictions against ground-truth labels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="pinsheet-scanner",
    help="Extract 9-pin bowling (Kegeln) scores from scanned score sheets.",
    add_completion=False,
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared option types & helpers
# ---------------------------------------------------------------------------

CropsOpt = Annotated[Path, typer.Option(help="Directory containing crop PNGs.")]
LabelsOpt = Annotated[Path, typer.Option(help="Ground-truth labels CSV.")]

_SEED = 42


def _require_path(path: Path, label: str, hint: str = "") -> None:
    if not path.exists():
        msg = f"{label} not found at {path}."
        if hint:
            msg += f" {hint}"
        raise typer.BadParameter(msg)


def _load_labeler_html() -> str:
    """Load the labeler HTML template from alongside this file."""
    return (Path(__file__).parent / "labeler.html").read_text()


def _setup_logging() -> None:
    """Configure root logger for CLI output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


# ── Inference ─────────────────────────────────────────────────────────────


@app.command()
def scan(
    image: Annotated[Path, typer.Argument(help="Path to the scanned score sheet.")],
    classifier: Annotated[
        Optional[Path], typer.Option(help="CNN classifier weights (.pt).")
    ] = None,
    confidence: Annotated[
        float, typer.Option(help="YOLO fallback confidence threshold.")
    ] = 0.25,
) -> None:
    """Scan a score sheet and print per-throw results."""
    _setup_logging()
    from pipeline import process_sheet

    result = process_sheet(
        image_path=image,
        classifier_path=classifier,
        confidence=confidence,
    )

    print(f"Detected {len(result.throws)} throws across {result.columns} columns\n")
    for t in result.throws:
        pins = "".join(str(p) for p in t.pins_down)
        mismatch = " ⚠ OCR mismatch" if t.ocr_mismatch else ""
        print(
            f"C{t.column:>2} | R{t.row:>2} | {pins} => {t.score}"
            f" | YOLO {t.confidence:.2f} | CNN {t.classification_confidence:.2f}{mismatch}"
        )
    print(f"\nTotal pins knocked down: {result.total_pins}")


# ── Training ──────────────────────────────────────────────────────────────


def _format_kfold_table(result) -> str:  # noqa: ANN001 (KFoldResult)
    """Format a K-fold result summary as a printable table."""
    lines = [
        f"\n{'=' * 50}",
        f"{'Fold':>5}  {'Val Loss':>9}  {'Val Acc':>8}",
        "-" * 28,
    ]
    for i, (loss, acc) in enumerate(zip(result.losses, result.accuracies)):
        lines.append(f"{i + 1:>5}  {loss:>9.4f}  {acc:>7.2%}")
    lines.append("-" * 28)
    lines.append(f"{'Mean':>5}  {result.mean_loss:>9.4f}  {result.mean_acc:>7.2%}")
    lines.append(f"{'±Std':>5}  {result.std_loss:>9.4f}  {result.std_acc:>7.2%}")
    return "\n".join(lines)


@app.command()
def train(
    crops: CropsOpt = Path("debug_crops/raw"),
    labels: LabelsOpt = Path("debug_crops/labels.csv"),
    output: Annotated[Path, typer.Option(help="Output weights path.")] = Path(
        "models/pin_classifier.pt"
    ),
    folds: Annotated[int, typer.Option(help="Number of cross-validation folds.")] = 5,
    epochs: Annotated[int, typer.Option(help="Training epochs per fold.")] = 60,
) -> None:
    """K-fold cross-validate then retrain the CNN classifier on all data.

    Hyperparameters are loaded from ``models/hyperparams.json`` if present
    (run ``tune`` first to generate it).  Saves a model bundle sidecar
    alongside the weights and appends a record to ``experiments.jsonl``.
    """
    _setup_logging()
    from classify import resolve_device
    from labels import load_labels_as_list
    from training import (
        HYPERPARAMS_PATH,
        kfold_train,
        load_defaults,
        log_experiment,
    )

    _require_path(labels, "Labels", "Run `pinsheet-scanner label` first.")
    _require_path(crops, "Crops directory")

    defaults = load_defaults()
    hp_kwargs = dict(
        lr=float(defaults["lr"]),
        weight_decay=float(defaults["weight_decay"]),
        dropout=float(defaults["dropout"]),
        batch_size=int(defaults["batch_size"]),
        scheduler_name=str(defaults["scheduler"]),
    )

    all_entries = load_labels_as_list(labels)
    if len(all_entries) < folds:
        raise typer.BadParameter(
            f"Only {len(all_entries)} images — need at least {folds}."
        )

    dev = resolve_device(None)

    if HYPERPARAMS_PATH.exists():
        print(f"Loaded tuned defaults from {HYPERPARAMS_PATH}")
    print(f"Device: {dev}")
    print(f"Total: {len(all_entries)}  |  Folds: {folds}  |  Epochs: {epochs}")
    print(
        f"Hyperparams: lr={hp_kwargs['lr']:.0e}  wd={hp_kwargs['weight_decay']:.0e}  "
        f"dropout={hp_kwargs['dropout']}  bs={hp_kwargs['batch_size']}  "
        f"sched={hp_kwargs['scheduler_name']}\n"
    )

    result = kfold_train(
        all_entries, crops, output,
        folds=folds, epochs=epochs, device=dev, seed=_SEED,
        hp_kwargs=hp_kwargs,
    )

    print(_format_kfold_table(result))
    print(f"\n  Final train loss: {result.final_train_loss:.4f}")
    print(f"  Saved to {result.output.resolve()}")

    log_experiment({
        "command": "train",
        "folds": folds, "epochs": epochs,
        "val_acc_mean": round(result.mean_acc, 6),
        "val_acc_std": round(result.std_acc, 6),
        "val_loss_mean": round(result.mean_loss, 6),
        "hyperparams": hp_kwargs,
        "output": str(output),
    })
    print("  Experiment logged to experiments.jsonl")


@app.command("train-detector")
def train_detector(
    data: Annotated[Path, typer.Option(help="YOLO dataset YAML config.")] = Path(
        "data/dataset.yaml"
    ),
    model: Annotated[str, typer.Option(help="Pretrained base model.")] = "yolo11n.pt",
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 50,
    imgsz: Annotated[int, typer.Option(help="Training image size.")] = 640,
) -> None:
    """Train a YOLOv11n model to detect pin diagrams on score sheets."""
    _setup_logging()
    _require_path(data, "Dataset config", "Populate data/train/ and data/val/ first.")

    from ultralytics import YOLO  # type: ignore[attr-defined]

    yolo = YOLO(model)
    yolo.train(
        data=str(data), epochs=epochs, imgsz=imgsz, batch=-1,
        project="runs", name="pin_diagram",
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
        degrees=5.0, translate=0.05, scale=0.2,
        flipud=0.0, fliplr=0.0, mosaic=0.5,
    )
    metrics = yolo.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    best = Path("runs/pin_diagram/weights/best.pt")
    if best.exists():
        print(f"\nBest weights: {best}\n  cp {best} models/pin_diagram.pt")


# ── Tuning ────────────────────────────────────────────────────────────────


@app.command()
def tune(
    crops: CropsOpt = Path("debug_crops/raw"),
    labels: LabelsOpt = Path("debug_crops/labels.csv"),
    trials: Annotated[int, typer.Option(help="Number of Optuna trials.")] = 20,
    epochs: Annotated[int, typer.Option(help="Epochs per trial.")] = 40,
) -> None:
    """Hyperparameter search with Optuna — saves best config to models/hyperparams.json."""
    _setup_logging()
    import optuna

    from classify import resolve_device
    from labels import load_labels_as_list
    from training import HYPERPARAMS_PATH, log_experiment, split_entries, train_and_evaluate

    _require_path(labels, "Labels", "Run `pinsheet-scanner label` first.")
    _require_path(crops, "Crops directory")

    all_entries = load_labels_as_list(labels)
    val_count = max(1, len(all_entries) // 5)
    if len(all_entries) <= val_count:
        raise typer.BadParameter(f"Only {len(all_entries)} images — need more.")

    train_entries, val_entries = split_entries(all_entries, val_count, _SEED)
    dev = resolve_device(None)

    print(f"Train: {len(train_entries)}  |  Val: {len(val_entries)}")
    print(f"Trials: {trials}  |  Epochs per trial: {epochs}\n")

    def objective(trial: optuna.Trial) -> float:
        hp = dict(
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
            scheduler_name=trial.suggest_categorical(
                "scheduler", ["plateau", "cosine", "onecycle", "step"]
            ),
        )
        val_loss, val_acc = train_and_evaluate(
            train_entries=train_entries, val_entries=val_entries,
            crops_dir=crops, epochs=epochs, device=dev, seed=_SEED, **hp,
        )
        trial.set_user_attr("val_acc", val_acc)
        return val_loss

    sampler = optuna.samplers.TPESampler(seed=_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials)

    best = study.best_trial
    print(f"\n{'=' * 60}\n  BEST TRIAL\n{'=' * 60}")
    for key, value in best.params.items():
        print(f"  {key:<16} {value}")
    print(f"  {'val_loss':<16} {best.value:.4f}")
    print(f"  {'val_acc':<16} {best.user_attrs['val_acc']:.2%}")

    hp_dict = {
        "lr": best.params["lr"],
        "weight_decay": best.params["weight_decay"],
        "dropout": best.params["dropout"],
        "batch_size": best.params["batch_size"],
        "scheduler": best.params["scheduler"],
    }
    HYPERPARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    HYPERPARAMS_PATH.write_text(json.dumps(hp_dict, indent=2) + "\n")
    print(f"\n  Saved to {HYPERPARAMS_PATH}")

    log_experiment({
        "command": "tune",
        "trials": trials, "epochs_per_trial": epochs,
        "best_val_loss": round(best.value, 6),
        "best_val_acc": round(best.user_attrs["val_acc"], 6),
        "best_hyperparams": hp_dict,
    })


# ── Debugging ─────────────────────────────────────────────────────────────


def _load_detection_pipeline(
    image_path: Path,
    confidence: float = 0.25,
):
    """Shared setup for commands that detect + optionally classify crops.

    Returns:
        ``(rectified, sorted_dets, crops)`` — grayscale image, sorted
        detections, and cropped pin diagrams.
    """
    import cv2

    from detect import crop_detections, detect_pin_diagrams, load_model, sort_detections
    from pipeline import DEFAULT_DETECTOR_PATH
    from preprocess import rectify_sheet

    raw = cv2.imread(str(image_path))
    if raw is None:
        raise typer.BadParameter(f"Could not load image: {image_path}")

    rectified = rectify_sheet(raw)
    yolo = load_model(DEFAULT_DETECTOR_PATH) if DEFAULT_DETECTOR_PATH.exists() else None
    sorted_dets = sort_detections(detect_pin_diagrams(yolo, rectified, confidence))
    crops = crop_detections(rectified, sorted_dets)

    return rectified, sorted_dets, crops


@app.command()
def extract(
    image: Annotated[Path, typer.Argument(help="Path to the scanned score sheet.")],
    output: Annotated[Path, typer.Option(help="Output directory.")] = Path("debug_crops"),
    confidence: Annotated[float, typer.Option(help="Detection confidence.")] = 0.25,
) -> None:
    """Extract and classify pin-diagram crops from a sheet image."""
    _setup_logging()
    import csv

    import cv2

    from classify import classify_pins_batch_with_confidence, load_classifier
    from detect import draw_detections
    from pipeline import DEFAULT_CLASSIFIER_PATH

    _require_path(image, "Image")

    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rectified, sorted_dets, crops = _load_detection_pipeline(image, confidence)
    print(f"Detected {len(sorted_dets)} pin diagrams")

    vis = cv2.cvtColor(rectified, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(output / "annotated_full.jpg"), draw_detections(vis, sorted_dets))
    print(f"Saved annotated image to {output / 'annotated_full.jpg'}")

    names = [f"c{d.column:02d}_r{d.row:02d}" for d in sorted_dets]
    for name, crop in zip(names, crops):
        cv2.imwrite(str(raw_dir / f"{name}.png"), crop)

    has_classifier = DEFAULT_CLASSIFIER_PATH.exists()
    classifications: list[tuple[list[int], float]] = []
    if has_classifier:
        cnn, dev = load_classifier(DEFAULT_CLASSIFIER_PATH)
        classifications = classify_pins_batch_with_confidence(cnn, crops, device=dev)
    else:
        print(f"\nClassifier not found at {DEFAULT_CLASSIFIER_PATH} — skipping.")

    header = f"{'Name':<20} {'Size':<12}" + (
        f" {'Pins':>10} {'Score':>6} {'Conf':>6}" if has_classifier else ""
    )
    print(f"\n{header}")
    print("-" * len(header))

    csv_rows: list[dict] = []
    for i, (det, crop) in enumerate(zip(sorted_dets, crops)):
        row: dict = {
            "name": names[i], "width": crop.shape[1], "height": crop.shape[0],
            "column": det.column, "row": det.row, "det_confidence": round(det.confidence, 4),
        }
        line = f"{names[i]:<20} {crop.shape[1]}x{crop.shape[0]:<12}"
        if has_classifier and i < len(classifications):
            pins, cls_conf = classifications[i]
            pins_str = "".join(str(p) for p in pins)
            line += f" {pins_str:>10} {sum(pins):>6} {cls_conf:>6.2f}"
            row.update(pins=pins_str, score=sum(pins), cls_confidence=round(cls_conf, 4))
        print(line)
        csv_rows.append(row)

    if csv_rows:
        csv_path = output / "predictions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nPredictions saved to {csv_path}")

    print(f"\nAll crops saved to {output.resolve()}")


# ── Labeling ──────────────────────────────────────────────────────────────


@app.command()
def label(
    crops: CropsOpt = Path("debug_crops/raw"),
) -> None:
    """Open the browser labeling UI to annotate ground-truth pin states.

    Crops are presented in ascending CNN-confidence order (least certain
    first) so each session targets the examples the model needs most.
    """
    _setup_logging()
    import json as _json
    import webbrowser
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    from urllib.parse import urlparse

    import cv2

    from classify import classify_pins_batch_with_confidence, load_classifier
    from labels import load_labels_as_dict, save_labels
    from pipeline import DEFAULT_CLASSIFIER_PATH

    _require_path(crops, "Crops directory", "Run `pinsheet-scanner extract <image>` first.")

    all_names = sorted(p.name for p in crops.glob("*.png"))
    if not all_names:
        raise typer.BadParameter(f"No .png files found in {crops}.")

    labels_path = crops.parent / "labels.csv"
    existing_labels = load_labels_as_dict(labels_path)
    labeled_count = sum(1 for n in all_names if n in existing_labels)
    print(f"Found {len(all_names)} crops in {crops}")
    print(f"Already labeled: {labeled_count}/{len(all_names)}")

    # ── Run CNN for suggestions + sort by ascending confidence ─────────
    predictions: dict[str, dict] = {}
    if DEFAULT_CLASSIFIER_PATH.exists():
        print("Running CNN for suggestions (sorted by confidence)...")
        cnn, dev = load_classifier(DEFAULT_CLASSIFIER_PATH)
        images = [
            img for name in all_names
            if (img := cv2.imread(str(crops / name), cv2.IMREAD_GRAYSCALE)) is not None
        ]
        results = classify_pins_batch_with_confidence(cnn, images, device=dev)
        for name, (pins, conf) in zip(all_names, results):
            predictions[name] = {"pins": pins, "conf": round(conf, 4)}
        print(f"  Got predictions for {len(predictions)} crops")
    else:
        print(f"  Classifier not found at {DEFAULT_CLASSIFIER_PATH} — no suggestions.")

    # Sort: unlabeled first (ascending confidence), labeled last.
    crop_names = sorted(
        all_names,
        key=lambda n: (
            n in existing_labels,
            predictions.get(n, {}).get("conf", 1.0),
        ),
    )

    _html_template = _load_labeler_html()

    def make_handler():
        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # silence request log

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    page = _html_template
                    page = page.replace("/*CROPS_JSON*/", _json.dumps(crop_names))
                    page = page.replace("/*PREDICTIONS_JSON*/", _json.dumps(predictions))
                    page = page.replace("/*LABELS_JSON*/", _json.dumps(existing_labels))
                    self._respond(200, "text/html", page.encode())
                elif parsed.path.startswith("/crop/"):
                    img_path = crops / parsed.path[6:]
                    if img_path.exists() and img_path.parent == crops:
                        self._respond(200, "image/png", img_path.read_bytes())
                    else:
                        self._respond(404, "text/plain", b"Not found")
                else:
                    self._respond(404, "text/plain", b"Not found")

            def do_POST(self):
                if self.path == "/save":
                    length = int(self.headers.get("Content-Length", 0))
                    body = _json.loads(self.rfile.read(length))
                    if body["filename"] not in set(crop_names):
                        self._respond(400, "text/plain", b"Unknown crop")
                        return
                    existing_labels[body["filename"]] = body["pins"]
                    save_labels(labels_path, existing_labels)
                    self._respond(200, "application/json", b'{"ok":true}')
                else:
                    self._respond(404, "text/plain", b"Not found")

            def _respond(self, code: int, content_type: str, data: bytes):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    port = 8787
    server = HTTPServer(("127.0.0.1", port), make_handler())
    url = f"http://127.0.0.1:{port}"
    print(f"\nLabeler running at {url}")
    print("Press Ctrl+C to stop.\n")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        done = sum(1 for n in all_names if n in existing_labels)
        print(f"\n\nLabels saved to {labels_path}")
        print(f"Progress: {done}/{len(all_names)} labeled")


# ── Validation ────────────────────────────────────────────────────────────


@app.command()
def accuracy(
    crops: CropsOpt = Path("debug_crops/raw"),
    labels: LabelsOpt = Path("debug_crops/labels.csv"),
    classifier: Annotated[
        Optional[Path], typer.Option(help="CNN classifier weights (.pt).")
    ] = None,
) -> None:
    """Compare ground-truth labels against CNN predictions."""
    _setup_logging()
    import cv2

    from classify import classify_pins_batch_with_confidence, load_classifier
    from constants import NUM_PINS
    from labels import load_labels_as_dict
    from pipeline import DEFAULT_CLASSIFIER_PATH

    classifier_path = classifier or DEFAULT_CLASSIFIER_PATH

    _require_path(labels, "Labels", "Run `pinsheet-scanner label` first.")
    _require_path(crops, "Crops directory")

    label_map = load_labels_as_dict(labels)
    if not label_map:
        print("No labels found — nothing to validate.")
        return

    print(f"Loaded {len(label_map)} ground-truth labels from {labels}")

    names = sorted(label_map.keys())
    images, valid_names = [], []
    for name in names:
        img = cv2.imread(str(crops / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Warning: could not load {name}, skipping")
            continue
        images.append(img)
        valid_names.append(name)

    model, dev = load_classifier(classifier_path)
    results = classify_pins_batch_with_confidence(model, images, device=dev)

    total_pins = correct_pins = total_diagrams = correct_diagrams = 0
    per_pin_correct = [0] * NUM_PINS
    per_pin_total = [0] * NUM_PINS
    mismatches: list[tuple[str, list[int], list[int], float]] = []

    for name, (pred_pins, conf) in zip(valid_names, results):
        gt = label_map[name]
        total_diagrams += 1
        diagram_ok = True
        for i in range(NUM_PINS):
            per_pin_total[i] += 1
            total_pins += 1
            if pred_pins[i] == gt[i]:
                correct_pins += 1
                per_pin_correct[i] += 1
            else:
                diagram_ok = False
        if diagram_ok:
            correct_diagrams += 1
        else:
            mismatches.append((name, gt, pred_pins, conf))

    pin_acc = correct_pins / total_pins * 100 if total_pins else 0
    diag_acc = correct_diagrams / total_diagrams * 100 if total_diagrams else 0

    print(f"\n{'=' * 60}")
    print(f"  Results: {total_diagrams} diagrams, {total_pins} pins")
    print(f"{'=' * 60}")
    print(f"  Per-pin accuracy:     {correct_pins}/{total_pins} ({pin_acc:.1f}%)")
    print(f"  Per-diagram accuracy: {correct_diagrams}/{total_diagrams} ({diag_acc:.1f}%)")

    print(f"\n  Per-position accuracy:")
    print(f"  {'Pin':>4}  {'Correct':>8}  {'Total':>6}  {'Acc':>7}")
    print(f"  {'-' * 30}")
    for i in range(NUM_PINS):
        acc = per_pin_correct[i] / per_pin_total[i] * 100 if per_pin_total[i] else 0
        marker = "" if acc >= 95 else " ←" if acc >= 80 else " ← LOW"
        print(
            f"  {i:>4}  {per_pin_correct[i]:>8}  {per_pin_total[i]:>6}  {acc:>6.1f}%{marker}"
        )

    if mismatches:
        print(f"\n  Mismatches ({len(mismatches)}):")
        print(f"  {'Name':<20} {'Ground Truth':>12} {'Prediction':>12} {'Conf':>6}")
        print(f"  {'-' * 54}")
        for name, gt, pred, conf in sorted(mismatches):
            gt_str = "".join(str(p) for p in gt)
            pred_str = "".join(str(p) for p in pred)
            diff = "".join("^" if g != p else " " for g, p in zip(gt, pred))
            print(f"  {name:<20} {gt_str:>12} {pred_str:>12} {conf:>5.0%}")
            print(f"  {'':20} {'':12} {diff:>12}")
    else:
        print("\n  No mismatches — perfect accuracy! 🎯")

    print()
