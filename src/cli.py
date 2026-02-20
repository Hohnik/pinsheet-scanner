"""CLI for pinsheet-scanner."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
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
SEED = 42


def _require(path: Path, label: str, hint: str = "") -> None:
    if not path.exists():
        raise typer.BadParameter(f"{label} not found at {path}." + (f" {hint}" if hint else ""))


def _hp_keys(hp: dict) -> dict:
    """Extract only the training-relevant keys from a hyperparams dict."""
    return {k: hp[k] for k in ("lr", "weight_decay", "dropout", "batch_size")}


# ── Inference ─────────────────────────────────────────────────────────────


@app.command()
def scan(
    image: Annotated[Path, typer.Argument(help="Path to the scanned score sheet.")],
    classifier: Annotated[Optional[Path], typer.Option(help="CNN weights (.pt).")] = None,
    confidence: Annotated[float, typer.Option(help="YOLO confidence threshold.")] = 0.25,
    ocr: Annotated[bool, typer.Option("--ocr/--no-ocr",
        help="Cross-validate CNN scores with Tesseract OCR (~5 s extra).")] = False,
) -> None:
    """Scan a score sheet and print per-throw results.

    OCR is disabled by default (saves ~5 s/sheet).  Enable with --ocr to add
    ⚠ markers where the CNN score disagrees with the printed digit.
    """
    from pipeline import process_sheet

    result = process_sheet(
        image_path=image, classifier_path=classifier,
        confidence=confidence, use_ocr=ocr,
    )
    n = len(result.throws)
    print(f"Detected {n} throws across {result.columns} columns\n")

    if not result.throws:
        print("No pin diagrams found.")
        return

    for t in result.throws:
        pins = "".join(str(p) for p in t.pins_down)
        flag = " ⚠ OCR mismatch" if t.ocr_mismatch else ""
        print(f"C{t.column:>2} | R{t.row:>2} | {pins} => {t.score}"
              f" | det {t.confidence:.2f} | cls {t.classification_confidence:.2f}{flag}")

    # Per-column summaries (Volle/Abr pairs make up a Bahn)
    cols: dict[int, list[int]] = {}
    for t in result.throws:
        cols.setdefault(t.column, []).append(t.score)
    print()
    bahn_pairs = list(zip(sorted(cols)[::2], sorted(cols)[1::2]))
    if bahn_pairs and all(len(cols[v]) == len(cols[a]) for v, a in bahn_pairs):
        for v, a in bahn_pairs:
            vt, at = sum(cols[v]), sum(cols[a])
            print(f"Bahn (C{v}+C{a}):  Volle={vt}  Abr={at}  Total={vt + at}")
    else:
        for c in sorted(cols):
            print(f"Col {c}: {sum(cols[c])}")
    print(f"\nTotal pins knocked down: {result.total_pins}")


@app.command()
def collect(
    image: Annotated[Path, typer.Argument(help="Sheet image to harvest crops from.")],
    crops: CropsOpt = Path("debug_crops/raw"),
    labels: LabelsOpt = Path("debug_crops/labels.csv"),
    confidence_threshold: Annotated[float, typer.Option(
        help="Minimum CNN confidence to auto-accept a prediction.")] = 0.85,
    confidence: Annotated[float, typer.Option(help="YOLO detection confidence.")] = 0.25,
    overwrite: Annotated[bool, typer.Option("--overwrite",
        help="Replace existing pseudo-labels with fresh predictions.")] = False,
) -> None:
    """Harvest high-confidence crops from a sheet into the training set.

    Runs detection + CNN on the sheet, then upserts any crop where the
    classifier confidence exceeds --confidence-threshold into labels.csv.
    Use --overwrite to refresh existing pseudo-labels after retraining.
    Review low-confidence crops with `pinsheet-scanner label` afterwards.
    """
    import cv2

    from classify import classify_pins_batch, load_classifier
    from labels import load_labels_as_dict, save_labels
    from pipeline import DEFAULT_CLASSIFIER_PATH

    _require(image, "Image")
    _require(DEFAULT_CLASSIFIER_PATH, "Classifier weights", "Train a model first.")
    crops.mkdir(parents=True, exist_ok=True)

    existing = load_labels_as_dict(labels)
    n_before = len(existing)

    rectified, dets, raw_crops = _detect_and_crop(image, confidence)
    if not dets:
        print("No pin diagrams detected.")
        return

    cnn, dev = load_classifier(DEFAULT_CLASSIFIER_PATH)
    classifications = classify_pins_batch(cnn, raw_crops, device=dev)

    stem = image.stem
    added = updated = skipped_low = 0

    for det, crop, (pins, conf) in zip(dets, raw_crops, classifications):
        name = f"{stem}_c{det.column:02d}_r{det.row:02d}.png"
        already_labeled = name in existing
        if conf < confidence_threshold:
            skipped_low += 1
            continue
        if already_labeled and not overwrite:
            continue
        cv2.imwrite(str(crops / name), crop)
        if already_labeled:
            existing[name] = pins
            updated += 1
        else:
            existing[name] = pins
            added += 1

    # Always write through save_labels so CSV is sorted consistently.
    save_labels(labels, existing)

    print(f"Sheet: {image.name}  |  {len(dets)} diagrams detected")
    print(f"  Added:    {added}  (new, conf ≥ {confidence_threshold:.2f})")
    if overwrite:
        print(f"  Updated:  {updated}  (refreshed, conf ≥ {confidence_threshold:.2f})")
    print(f"  Skipped:  {skipped_low}  (conf < {confidence_threshold:.2f}) — review with `just label`")
    print(f"\nTotal labeled: {len(existing)}  (was {n_before})")


# ── Training ──────────────────────────────────────────────────────────────


@app.command()
def train(
    crops: CropsOpt = Path("debug_crops/raw"),
    labels: LabelsOpt = Path("debug_crops/labels.csv"),
    output: Annotated[Path, typer.Option(help="Output weights path.")] = Path("models/pin_classifier.pt"),
    folds: Annotated[int, typer.Option(help="Cross-validation folds.")] = 5,
    epochs: Annotated[int, typer.Option(help="Training epochs per fold.")] = 200,
) -> None:
    """K-fold cross-validate then retrain the CNN classifier."""
    import numpy as np
    import torch

    from classify import resolve_device
    from labels import load_labels_as_list
    from training import HYPERPARAMS_PATH, load_hyperparams, train_new_model

    _require(labels, "Labels", "Run `pinsheet-scanner label` first.")
    _require(crops, "Crops directory")

    hp = _hp_keys(load_hyperparams())
    all_entries = load_labels_as_list(labels)
    if len(all_entries) < folds:
        raise typer.BadParameter(f"Only {len(all_entries)} images — need ≥ {folds}.")

    dev = resolve_device(None)
    if HYPERPARAMS_PATH.exists():
        print(f"Loaded hyperparams from {HYPERPARAMS_PATH}")
    print(f"Device: {dev}  |  {len(all_entries)} images  |  {folds} folds  |  {epochs} epochs\n")

    indices = np.random.default_rng(SEED).permutation(len(all_entries))
    fold_splits = np.array_split(indices, folds)
    losses, accs = [], []
    for fold in range(folds):
        val_idx = set(fold_splits[fold].tolist())
        train_e = [all_entries[i] for i in range(len(all_entries)) if i not in val_idx]
        val_e = [all_entries[i] for i in fold_splits[fold]]
        _, vl, va = train_new_model(
            train_e, crops, epochs, dev, SEED + fold, val_entries=val_e,
            desc=f"Fold {fold + 1}/{folds}  (train={len(train_e)}, val={len(val_e)})",
            **hp,
        )
        losses.append(vl)
        accs.append(va)
        from tqdm import tqdm as _tqdm
        _tqdm.write(f"  ↳ loss={vl:.4f}  acc={va:.2%}")

    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    print(f"\nMean: loss={np.mean(losses):.4f}  acc={mean_acc:.2%} ±{std_acc:.2%}")

    # Backup existing weights before overwriting so a bad retrain is recoverable.
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = output.with_name(f"{output.stem}.{ts}.bak.pt")
        shutil.copy2(output, backup)
        print(f"Backed up old weights → {backup.name}")

    print(f"\nRetraining on all {len(all_entries)} images...")
    model, loss, _ = train_new_model(all_entries, crops, epochs, dev, SEED,
                                     desc="Final retrain", **hp)
    torch.save(model.state_dict(), output)
    print(f"Final loss: {loss:.4f}  →  {output.resolve()}")


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
    epochs: Annotated[int, typer.Option(help="Epochs per trial (short — for ranking only).")] = 20,
) -> None:
    """Hyperparameter search with Optuna."""
    import optuna

    from classify import resolve_device
    from labels import load_labels_as_list
    from model import PinClassifier
    from training import HYPERPARAMS_PATH, split_entries, train_new_model

    _require(labels, "Labels", "Run `pinsheet-scanner label` first.")
    _require(crops, "Crops directory")

    all_entries = load_labels_as_list(labels)
    if len(all_entries) < 5:
        raise typer.BadParameter(f"Only {len(all_entries)} images — need more.")

    train_entries, val_entries = split_entries(all_entries, 0.2, SEED)
    dev = resolve_device(None)

    from tqdm import tqdm as _tqdm
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trial_bar = _tqdm(total=trials, desc="Tuning", unit="trial", dynamic_ncols=True)

    def objective(trial: optuna.Trial) -> float:
        hp = dict(
            lr=trial.suggest_float("lr", 5e-5, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
        )
        _, vl, va = train_new_model(train_entries, crops, epochs, dev, SEED,
                                    val_entries=val_entries,
                                    desc=f"  T{trial.number + 1}/{trials}", leave=False, **hp)
        trial.set_user_attr("val_acc", va)
        trial_bar.update(1)
        trial_bar.set_postfix(loss=f"{vl:.4f}", acc=f"{va:.1%}")
        return vl

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    trial_bar.close()

    best = study.best_trial
    print(f"\nBest trial #{best.number + 1}: loss={best.value:.4f}  acc={best.user_attrs['val_acc']:.2%}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Save hyperparams with provenance so stale configs are detectable.
    n_params = sum(p.numel() for p in PinClassifier().parameters())
    meta = {
        **best.params,
        "_tuned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_samples":  len(all_entries),
        "_model_params": n_params,
    }
    HYPERPARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    HYPERPARAMS_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved to {HYPERPARAMS_PATH}  (samples={len(all_entries)}, model_params={n_params})")


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
    manual_only: Annotated[bool, typer.Option("--manual-only",
        help="Evaluate only manually labeled crops (prefix = 'original_').")] = False,
    prefix: Annotated[Optional[str], typer.Option(
        help="Sheet prefix filter for --manual-only (default: 'original_').")] = None,
) -> None:
    """Validate CNN predictions against ground-truth labels.

    By default all labeled crops are evaluated (593 samples).
    Use --manual-only to evaluate only the hand-labeled 'original' sheet,
    which gives a less inflated accuracy (pseudo-labels were set by the model).
    """
    import cv2

    from classify import classify_pins_batch, load_classifier
    from labels import load_labels_as_dict
    from pipeline import DEFAULT_CLASSIFIER_PATH

    _require(labels, "Labels", "Run `pinsheet-scanner label` first.")
    _require(crops, "Crops directory")

    label_map = load_labels_as_dict(labels)
    if not label_map:
        print("No labels found.")
        return

    if manual_only:
        sheet_prefix = prefix or "original_"
        label_map = {k: v for k, v in label_map.items() if k.startswith(sheet_prefix)}
        if not label_map:
            raise typer.BadParameter(
                f"No crops with prefix '{sheet_prefix}' found. "
                "Pass --prefix to specify a different sheet name."
            )
        print(f"Evaluating {len(label_map)} manually labeled crops (prefix='{sheet_prefix}')")

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
        total_pins += len(gt)
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
