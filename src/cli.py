"""CLI for pinsheet-scanner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="pinsheet-scanner",
    help="Extract 9-pin bowling (Kegeln) scores from scanned score sheets.",
    add_completion=False,
    no_args_is_help=True,
)

CropsOpt = Annotated[Path, typer.Option(help="Directory containing crop PNGs.")]
LabelsOpt = Annotated[Path, typer.Option(help="Ground-truth labels CSV.")]
_SEED = 42


def _require(path: Path, label: str, hint: str = "") -> None:
    if not path.exists():
        raise typer.BadParameter(f"{label} not found at {path}." + (f" {hint}" if hint else ""))


# ── Inference ─────────────────────────────────────────────────────────────


@app.command()
def scan(
    image: Annotated[Path, typer.Argument(help="Path to the scanned score sheet.")],
    classifier: Annotated[Optional[Path], typer.Option(help="CNN weights (.pt).")] = None,
    confidence: Annotated[float, typer.Option(help="YOLO confidence threshold.")] = 0.25,
) -> None:
    """Scan a score sheet and print per-throw results."""
    from pipeline import process_sheet

    result = process_sheet(image_path=image, classifier_path=classifier, confidence=confidence)
    print(f"Detected {len(result.throws)} throws across {result.columns} columns\n")
    for t in result.throws:
        pins = "".join(str(p) for p in t.pins_down)
        flag = " ⚠ OCR mismatch" if t.ocr_mismatch else ""
        print(f"C{t.column:>2} | R{t.row:>2} | {pins} => {t.score}"
              f" | det {t.confidence:.2f} | cls {t.classification_confidence:.2f}{flag}")
    print(f"\nTotal pins knocked down: {result.total_pins}")


# ── Training ──────────────────────────────────────────────────────────────


@app.command()
def train(
    crops: CropsOpt = Path("debug_crops/raw"),
    labels: LabelsOpt = Path("debug_crops/labels.csv"),
    output: Annotated[Path, typer.Option(help="Output weights path.")] = Path("models/pin_classifier.pt"),
    folds: Annotated[int, typer.Option(help="Cross-validation folds.")] = 5,
    epochs: Annotated[int, typer.Option(help="Training epochs per fold.")] = 60,
) -> None:
    """K-fold cross-validate then retrain the CNN classifier."""
    import numpy as np
    from sklearn.model_selection import KFold

    from classify import resolve_device
    from labels import load_labels_as_list
    from training import HYPERPARAMS_PATH, load_defaults, log_experiment, retrain_all, train_and_evaluate

    _require(labels, "Labels", "Run `pinsheet-scanner label` first.")
    _require(crops, "Crops directory")

    defaults = load_defaults()
    hp = dict(lr=float(defaults["lr"]), weight_decay=float(defaults["weight_decay"]),
              dropout=float(defaults["dropout"]), batch_size=int(defaults["batch_size"]),
              scheduler_name=str(defaults["scheduler"]))

    all_entries = load_labels_as_list(labels)
    if len(all_entries) < folds:
        raise typer.BadParameter(f"Only {len(all_entries)} images — need ≥ {folds}.")

    dev = resolve_device(None)
    if HYPERPARAMS_PATH.exists():
        print(f"Loaded hyperparams from {HYPERPARAMS_PATH}")
    print(f"Device: {dev}  |  {len(all_entries)} images  |  {folds} folds  |  {epochs} epochs\n")

    kf = KFold(n_splits=folds, shuffle=True, random_state=_SEED)
    losses, accs = [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_entries)):
        train_e = [all_entries[i] for i in train_idx]
        val_e = [all_entries[i] for i in val_idx]
        print(f"Fold {fold + 1}/{folds}  (train={len(train_e)}, val={len(val_e)}) ", end="", flush=True)
        vl, va = train_and_evaluate(train_e, val_e, crops, epochs, dev, _SEED + fold, **hp)
        losses.append(vl)
        accs.append(va)
        print(f"→ loss={vl:.4f}  acc={va:.2%}")

    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    mean_loss = float(np.mean(losses))
    print(f"\nMean: loss={mean_loss:.4f}  acc={mean_acc:.2%} ±{std_acc:.2%}")

    print(f"\nRetraining on all {len(all_entries)} images...")
    extra = {"folds": folds, "epochs": epochs, **{k: v for k, v in hp.items() if k != "scheduler_name"}}
    best_loss = retrain_all(all_entries, crops, output, dev, epochs, _SEED,
                            val_accuracy=mean_acc, extra_bundle=extra, **hp)
    print(f"Final loss: {best_loss:.4f}  →  {output.resolve()}")

    log_experiment({"command": "train", "folds": folds, "epochs": epochs,
                    "val_acc_mean": round(mean_acc, 6), "val_loss_mean": round(mean_loss, 6),
                    "hyperparams": hp, "output": str(output)})


@app.command("train-detector")
def train_detector(
    data: Annotated[Path, typer.Option(help="YOLO dataset YAML.")] = Path("data/dataset.yaml"),
    model: Annotated[str, typer.Option(help="Pretrained base model.")] = "yolo11n.pt",
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 50,
    imgsz: Annotated[int, typer.Option(help="Training image size.")] = 640,
) -> None:
    """Train a YOLOv11n model to detect pin diagrams."""
    _require(data, "Dataset config", "Populate data/train/ and data/val/ first.")
    from ultralytics import YOLO  # type: ignore[attr-defined]

    yolo = YOLO(model)
    yolo.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=-1,
               project="runs", name="pin_diagram",
               hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, degrees=5.0, translate=0.05,
               scale=0.2, flipud=0.0, fliplr=0.0, mosaic=0.5)
    m = yolo.val()
    print(f"\nmAP50: {m.box.map50:.4f}  |  mAP50-95: {m.box.map:.4f}")
    best = Path("runs/pin_diagram/weights/best.pt")
    if best.exists():
        print(f"Best weights: {best}")


# ── Tuning ────────────────────────────────────────────────────────────────


@app.command()
def tune(
    crops: CropsOpt = Path("debug_crops/raw"),
    labels: LabelsOpt = Path("debug_crops/labels.csv"),
    trials: Annotated[int, typer.Option(help="Number of Optuna trials.")] = 20,
    epochs: Annotated[int, typer.Option(help="Epochs per trial.")] = 40,
) -> None:
    """Hyperparameter search with Optuna."""
    import optuna

    from classify import resolve_device
    from labels import load_labels_as_list
    from training import HYPERPARAMS_PATH, log_experiment, split_entries, train_and_evaluate

    _require(labels, "Labels", "Run `pinsheet-scanner label` first.")
    _require(crops, "Crops directory")

    all_entries = load_labels_as_list(labels)
    val_count = max(1, len(all_entries) // 5)
    if len(all_entries) <= val_count:
        raise typer.BadParameter(f"Only {len(all_entries)} images — need more.")

    train_entries, val_entries = split_entries(all_entries, val_count, _SEED)
    dev = resolve_device(None)
    print(f"Train: {len(train_entries)}  |  Val: {len(val_entries)}  |  {trials} trials\n")

    def objective(trial: optuna.Trial) -> float:
        hp = dict(
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
            scheduler_name=trial.suggest_categorical("scheduler", ["plateau", "cosine", "onecycle", "step"]),
        )
        vl, va = train_and_evaluate(train_entries, val_entries, crops, epochs, dev, _SEED, **hp)
        trial.set_user_attr("val_acc", va)
        return vl

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=_SEED))
    study.optimize(objective, n_trials=trials)

    best = study.best_trial
    print(f"\nBest: loss={best.value:.4f}  acc={best.user_attrs['val_acc']:.2%}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    hp_dict = {k: best.params[k] for k in ("lr", "weight_decay", "dropout", "batch_size", "scheduler")}
    HYPERPARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    HYPERPARAMS_PATH.write_text(json.dumps(hp_dict, indent=2) + "\n")
    print(f"Saved to {HYPERPARAMS_PATH}")

    log_experiment({"command": "tune", "trials": trials, "epochs_per_trial": epochs,
                    "best_val_loss": round(best.value, 6), "best_hyperparams": hp_dict})


# ── Extract & Label ───────────────────────────────────────────────────────


def _detect_and_crop(image_path: Path, confidence: float = 0.25):
    """Shared: rectify → detect → crop.  Returns (rectified, sorted_dets, crops)."""
    import cv2

    from detect import crop_detections, detect_pin_diagrams, load_model, sort_detections
    from pipeline import DEFAULT_DETECTOR_PATH
    from preprocess import rectify_sheet

    raw = cv2.imread(str(image_path))
    if raw is None:
        raise typer.BadParameter(f"Could not load image: {image_path}")
    rectified = rectify_sheet(raw)
    yolo = load_model(DEFAULT_DETECTOR_PATH) if DEFAULT_DETECTOR_PATH.exists() else None
    dets = sort_detections(detect_pin_diagrams(yolo, rectified, confidence))
    return rectified, dets, crop_detections(rectified, dets)


@app.command()
def extract(
    image: Annotated[Path, typer.Argument(help="Path to the scanned score sheet.")],
    output: Annotated[Path, typer.Option(help="Output directory.")] = Path("debug_crops"),
    confidence: Annotated[float, typer.Option(help="Detection confidence.")] = 0.25,
) -> None:
    """Extract and classify pin-diagram crops from a sheet."""
    import csv
    import cv2

    from classify import classify_pins_batch, load_classifier
    from detect import draw_detections
    from pipeline import DEFAULT_CLASSIFIER_PATH

    _require(image, "Image")
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rectified, dets, crops = _detect_and_crop(image, confidence)
    print(f"Detected {len(dets)} pin diagrams")

    vis = cv2.cvtColor(rectified, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(output / "annotated_full.jpg"), draw_detections(vis, dets))

    names = [f"c{d.column:02d}_r{d.row:02d}" for d in dets]
    for name, crop in zip(names, crops):
        cv2.imwrite(str(raw_dir / f"{name}.png"), crop)

    classifications = []
    if DEFAULT_CLASSIFIER_PATH.exists():
        cnn, dev = load_classifier(DEFAULT_CLASSIFIER_PATH)
        classifications = classify_pins_batch(cnn, crops, device=dev)

    csv_rows: list[dict] = []
    for i, (det, crop) in enumerate(zip(dets, crops)):
        row: dict = {"name": names[i], "column": det.column, "row": det.row}
        if i < len(classifications):
            pins, conf = classifications[i]
            row.update(pins="".join(str(p) for p in pins), score=sum(pins), confidence=round(conf, 4))
        csv_rows.append(row)
        line = f"{names[i]:<14} {crop.shape[1]}x{crop.shape[0]}"
        if "pins" in row:
            line += f"  {row['pins']}  score={row['score']}  conf={row['confidence']:.2f}"
        print(line)

    if csv_rows:
        csv_path = output / "predictions.csv"
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=list(csv_rows[0].keys())).writeheader()
            csv.DictWriter(f, fieldnames=list(csv_rows[0].keys())).writerows(csv_rows)
        print(f"\nSaved to {csv_path}")


@app.command()
def label(crops: CropsOpt = Path("debug_crops/raw")) -> None:
    """Open browser labeling UI (crops sorted by ascending CNN confidence)."""
    import json as _json
    import webbrowser
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    from urllib.parse import urlparse

    import cv2

    from classify import classify_pins_batch, load_classifier
    from labels import load_labels_as_dict, save_labels
    from pipeline import DEFAULT_CLASSIFIER_PATH

    _require(crops, "Crops directory", "Run `pinsheet-scanner extract <image>` first.")
    all_names = sorted(p.name for p in crops.glob("*.png"))
    if not all_names:
        raise typer.BadParameter(f"No .png files in {crops}.")

    labels_path = crops.parent / "labels.csv"
    existing = load_labels_as_dict(labels_path)
    print(f"{len(all_names)} crops, {sum(1 for n in all_names if n in existing)} labeled")

    # CNN suggestions + confidence-based sort
    predictions: dict[str, dict] = {}
    if DEFAULT_CLASSIFIER_PATH.exists():
        cnn, dev = load_classifier(DEFAULT_CLASSIFIER_PATH)
        imgs = [img for n in all_names if (img := cv2.imread(str(crops / n), cv2.IMREAD_GRAYSCALE)) is not None]
        for name, (pins, conf) in zip(all_names, classify_pins_batch(cnn, imgs, device=dev)):
            predictions[name] = {"pins": pins, "conf": round(conf, 4)}

    crop_names = sorted(all_names, key=lambda n: (n in existing, predictions.get(n, {}).get("conf", 1.0)))
    html = (Path(__file__).parent / "labeler.html").read_text()

    def make_handler():
        class H(SimpleHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_GET(self):
                p = urlparse(self.path).path
                if p == "/":
                    page = html.replace("/*CROPS_JSON*/", _json.dumps(crop_names))
                    page = page.replace("/*PREDICTIONS_JSON*/", _json.dumps(predictions))
                    page = page.replace("/*LABELS_JSON*/", _json.dumps(existing))
                    self._send(200, "text/html", page.encode())
                elif p.startswith("/crop/"):
                    f = crops / p[6:]
                    if f.exists() and f.parent == crops:
                        self._send(200, "image/png", f.read_bytes())
                    else:
                        self._send(404, "text/plain", b"Not found")
                else:
                    self._send(404, "text/plain", b"Not found")

            def do_POST(self):
                if self.path == "/save":
                    body = _json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    if body["filename"] not in set(crop_names):
                        self._send(400, "text/plain", b"Unknown crop")
                        return
                    existing[body["filename"]] = body["pins"]
                    save_labels(labels_path, existing)
                    self._send(200, "application/json", b'{"ok":true}')
                else:
                    self._send(404, "text/plain", b"Not found")

            def _send(self, code, ct, data):
                self.send_response(code)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        return H

    port = 8787
    srv = HTTPServer(("127.0.0.1", port), make_handler())
    url = f"http://127.0.0.1:{port}"
    print(f"Labeler: {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"\nSaved to {labels_path}  ({sum(1 for n in all_names if n in existing)}/{len(all_names)} labeled)")


# ── Validation ────────────────────────────────────────────────────────────


@app.command()
def accuracy(
    crops: CropsOpt = Path("debug_crops/raw"),
    labels: LabelsOpt = Path("debug_crops/labels.csv"),
    classifier: Annotated[Optional[Path], typer.Option(help="CNN weights (.pt).")] = None,
) -> None:
    """Validate CNN predictions against ground-truth labels."""
    import cv2

    from classify import classify_pins_batch, load_classifier
    from constants import NUM_PINS
    from labels import load_labels_as_dict
    from pipeline import DEFAULT_CLASSIFIER_PATH

    _require(labels, "Labels", "Run `pinsheet-scanner label` first.")
    _require(crops, "Crops directory")

    label_map = load_labels_as_dict(labels)
    if not label_map:
        print("No labels found.")
        return

    names = sorted(label_map)
    images, valid = [], []
    for n in names:
        img = cv2.imread(str(crops / n), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            valid.append(n)

    model, dev = load_classifier(classifier or DEFAULT_CLASSIFIER_PATH)
    results = classify_pins_batch(model, images, device=dev)

    correct_pins = correct_diags = total_pins = 0
    mismatches = []
    for name, (pred, conf) in zip(valid, results):
        gt = label_map[name]
        ok = pred == gt
        correct_pins += sum(p == g for p, g in zip(pred, gt))
        correct_diags += int(ok)
        total_pins += NUM_PINS
        if not ok:
            mismatches.append((name, gt, pred, conf))

    n = len(valid)
    print(f"\n{n} diagrams  |  pin acc: {correct_pins}/{total_pins} ({correct_pins/total_pins:.1%})"
          f"  |  diagram acc: {correct_diags}/{n} ({correct_diags/n:.1%})")

    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for name, gt, pred, conf in sorted(mismatches):
            print(f"  {name:<18} gt={''.join(str(p) for p in gt)}  pred={''.join(str(p) for p in pred)}  conf={conf:.0%}")
    else:
        print("Perfect accuracy! 🎯")
