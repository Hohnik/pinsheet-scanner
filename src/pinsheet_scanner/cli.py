"""Unified CLI for pinsheet-scanner.

All commands are registered on a single ``typer.Typer`` app and exposed
via the ``pinsheet-scanner`` console entry-point.
"""

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

# ---------------------------------------------------------------------------
# Shared option types
# ---------------------------------------------------------------------------

CropsOpt = Annotated[Path, typer.Option(help="Directory containing crop PNGs.")]
LabelsOpt = Annotated[Path, typer.Option(help="Ground-truth labels CSV.")]
DeviceOpt = Annotated[Optional[str], typer.Option(help="Device: cpu / cuda / mps.")]
SeedOpt = Annotated[int, typer.Option(help="Random seed.")]


def _require_path(path: Path, label: str, hint: str = "") -> None:
    """Abort with a clear message when *path* is missing."""
    if not path.exists():
        msg = f"{label} not found at {path}."
        if hint:
            msg += f" {hint}"
        raise typer.BadParameter(msg)


# ── Inference ─────────────────────────────────────────────────────────────


@app.command()
def scan(
    image: Annotated[
        Path, typer.Argument(help="Path to the scanned score sheet image.")
    ],
    model: Annotated[
        Optional[Path], typer.Option(help="YOLO detector weights (.pt).")
    ] = None,
    classifier_model: Annotated[
        Optional[Path], typer.Option(help="CNN classifier weights (.pt).")
    ] = None,
    confidence: Annotated[
        float, typer.Option(help="Minimum detection confidence.")
    ] = 0.5,
) -> None:
    """Scan a score sheet and print per-throw results."""
    from .pipeline import process_sheet

    result = process_sheet(
        image_path=image,
        model_path=model,
        classifier_path=classifier_model,
        confidence=confidence,
    )

    print(f"Detected {len(result.throws)} throws across {result.columns} columns\n")
    for t in result.throws:
        pins = "".join(str(p) for p in t.pins_down)
        print(
            f"  Col {t.column:>2} | Row {t.row:>2} | Score {t.score:>1} "
            f"| Pins {pins} | Det {t.confidence:.2f} | Cls {t.classification_confidence:.2f}"
        )
    print(f"\nTotal pins knocked down: {result.total_pins}")


# ── Training ──────────────────────────────────────────────────────────────


@app.command("train-classifier")
def train_classifier(
    crops: CropsOpt = Path("debug_crops/raw"),
    labels: LabelsOpt = Path("debug_crops/labels.csv"),
    output: Annotated[Path, typer.Option(help="Output weights path.")] = Path(
        "models/pin_classifier.pt"
    ),
    val_count: Annotated[
        int, typer.Option(help="Images held out for validation.")
    ] = 20,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 60,
    batch_size: Annotated[Optional[int], typer.Option(help="Batch size.")] = None,
    lr: Annotated[Optional[float], typer.Option(help="Learning rate.")] = None,
    weight_decay: Annotated[
        Optional[float], typer.Option(help="Adam weight decay.")
    ] = None,
    dropout: Annotated[
        Optional[float], typer.Option(help="Dropout before head.")
    ] = None,
    scheduler: Annotated[
        Optional[str], typer.Option(help="LR scheduler: plateau|cosine|onecycle|step.")
    ] = None,
    device: DeviceOpt = None,
    seed: SeedOpt = 42,
) -> None:
    """Train the PinClassifier CNN on real labeled crops."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from .classify import resolve_device
    from .labels import load_labels_as_list
    from .model import PinClassifier
    from .training import (
        HYPERPARAMS_PATH,
        TRAIN_AUGMENT,
        RealCropDataset,
        evaluate,
        load_defaults,
        make_scheduler,
        split_entries,
        step_scheduler,
        train_one_epoch,
    )

    _require_path(labels, "Labels", "Run `pinsheet-scanner label` first.")
    _require_path(
        crops, "Crops directory", "Run `pinsheet-scanner debug-crops <image>` first."
    )

    defaults = load_defaults()
    batch_size = batch_size or int(defaults["batch_size"])
    lr = lr if lr is not None else float(defaults["lr"])
    weight_decay = (
        weight_decay if weight_decay is not None else float(defaults["weight_decay"])
    )
    dropout = dropout if dropout is not None else float(defaults["dropout"])
    scheduler = scheduler or str(defaults["scheduler"])

    all_entries = load_labels_as_list(labels)
    if len(all_entries) <= val_count:
        raise typer.BadParameter(
            f"Only {len(all_entries)} labeled images — need more than --val-count={val_count}."
        )

    train_entries, val_entries = split_entries(all_entries, val_count, seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = resolve_device(device)
    if HYPERPARAMS_PATH.exists():
        print(f"Loaded tuned defaults from {HYPERPARAMS_PATH}")
    print(f"Device: {dev}")
    print(f"Total labeled: {len(all_entries)}")
    print(f"Train: {len(train_entries)}  |  Val: {len(val_entries)}")
    print(
        f"Hyperparams: lr={lr:.0e}  wd={weight_decay:.0e}  "
        f"dropout={dropout}  bs={batch_size}  sched={scheduler}"
    )

    train_ds = RealCropDataset(
        crops, train_entries, augment_cfg=TRAIN_AUGMENT, seed=seed
    )
    val_ds = RealCropDataset(crops, val_entries, augment_cfg=None)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = PinClassifier(dropout=dropout).to(dev)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = make_scheduler(scheduler, optimizer, epochs, len(train_loader))
    step_per_batch = scheduler == "onecycle"

    best_val_loss = float("inf")
    best_val_acc = 0.0
    output.parent.mkdir(parents=True, exist_ok=True)

    header = f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val Acc':>10}  {'LR':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            dev,
            scheduler=sched if step_per_batch else None,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, dev)
        step_scheduler(sched, scheduler, metric=val_loss)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss, best_val_acc = val_loss, val_acc
            torch.save(model.state_dict(), output)
            marker = " ✓"

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:5d}  {train_loss:10.4f}  {val_loss:10.4f}  {val_acc:9.2%}  {cur_lr:10.1e}{marker}"
        )

    print(f"\nBest val loss: {best_val_loss:.4f}  |  Best val acc: {best_val_acc:.2%}")
    print(f"Saved to {output.resolve()}")


@app.command("train-detector")
def train_detector(
    data: Annotated[Path, typer.Option(help="YOLO dataset YAML config.")] = Path(
        "data/dataset.yaml"
    ),
    model: Annotated[
        str, typer.Option(help="Pretrained model to fine-tune.")
    ] = "yolo11n.pt",
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 50,
    imgsz: Annotated[int, typer.Option(help="Training image size.")] = 640,
    batch: Annotated[int, typer.Option(help="Batch size (-1 = auto).")] = -1,
    device: DeviceOpt = None,
    project: Annotated[
        str, typer.Option(help="Project directory for results.")
    ] = "runs",
    name: Annotated[str, typer.Option(help="Experiment name.")] = "pin_diagram",
) -> None:
    """Train a YOLOv11n model to detect pin diagrams on score sheets."""
    _require_path(data, "Dataset config", "Populate data/train/ and data/val/ first.")

    from ultralytics import YOLO  # type: ignore[attr-defined]

    yolo = YOLO(model)
    yolo.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.5,
    )
    metrics = yolo.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    best_weights = Path(project) / name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\nBest weights: {best_weights}")
        print(f"  cp {best_weights} models/pin_diagram.pt")
    else:
        print("\nTraining complete. Check runs/ for weights.")


# ── Tuning & cross-validation ────────────────────────────────────────────


@app.command()
def tune(
    crops: CropsOpt = Path("debug_crops/raw"),
    labels: LabelsOpt = Path("debug_crops/labels.csv"),
    val_count: Annotated[int, typer.Option(help="Validation hold-out count.")] = 20,
    trials: Annotated[int, typer.Option(help="Number of Optuna trials.")] = 20,
    epochs: Annotated[int, typer.Option(help="Epochs per trial.")] = 40,
    device: DeviceOpt = None,
    seed: SeedOpt = 42,
) -> None:
    """Hyperparameter tuning with Optuna (TPE sampler)."""
    import optuna

    from .classify import resolve_device
    from .labels import load_labels_as_list
    from .training import HYPERPARAMS_PATH, split_entries, train_and_evaluate

    _require_path(labels, "Labels", "Run `pinsheet-scanner label` first.")
    _require_path(crops, "Crops directory")

    all_entries = load_labels_as_list(labels)
    if len(all_entries) <= val_count:
        raise typer.BadParameter(
            f"Only {len(all_entries)} images — need more than --val-count={val_count}."
        )

    train_entries, val_entries = split_entries(all_entries, val_count, seed)
    dev = resolve_device(device)

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
            train_entries=train_entries,
            val_entries=val_entries,
            crops_dir=crops,
            epochs=epochs,
            device=dev,
            seed=seed,
            **hp,
        )
        trial.set_user_attr("val_acc", val_acc)
        return val_loss

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials)

    best = study.best_trial
    print(f"\n{'=' * 60}")
    print("  BEST TRIAL")
    print(f"{'=' * 60}")
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
    print("  (train-classifier will use these as defaults on next run)")


@app.command()
def kfold(
    crops: CropsOpt = Path("debug_crops/raw"),
    labels: LabelsOpt = Path("debug_crops/labels.csv"),
    output: Annotated[Path, typer.Option(help="Output weights path.")] = Path(
        "models/pin_classifier.pt"
    ),
    folds: Annotated[int, typer.Option(help="Number of folds.")] = 5,
    epochs: Annotated[int, typer.Option(help="Epochs per fold.")] = 60,
    batch_size: Annotated[Optional[int], typer.Option(help="Batch size.")] = None,
    lr: Annotated[Optional[float], typer.Option(help="Learning rate.")] = None,
    weight_decay: Annotated[
        Optional[float], typer.Option(help="Adam weight decay.")
    ] = None,
    dropout: Annotated[
        Optional[float], typer.Option(help="Dropout before head.")
    ] = None,
    scheduler: Annotated[Optional[str], typer.Option(help="LR scheduler.")] = None,
    device: DeviceOpt = None,
    seed: SeedOpt = 42,
    retrain: Annotated[
        bool, typer.Option(help="Retrain final model on all data.")
    ] = True,
) -> None:
    """K-fold cross-validation for the PinClassifier CNN."""
    import numpy as np
    from sklearn.model_selection import KFold

    from .classify import resolve_device
    from .labels import load_labels_as_list
    from .training import (
        HYPERPARAMS_PATH,
        load_defaults,
        retrain_all,
        train_and_evaluate,
    )

    _require_path(labels, "Labels", "Run `pinsheet-scanner label` first.")
    _require_path(crops, "Crops directory")

    defaults = load_defaults()
    batch_size = batch_size or int(defaults["batch_size"])
    lr = lr if lr is not None else float(defaults["lr"])
    weight_decay = (
        weight_decay if weight_decay is not None else float(defaults["weight_decay"])
    )
    dropout = dropout if dropout is not None else float(defaults["dropout"])
    scheduler = scheduler or str(defaults["scheduler"])

    all_entries = load_labels_as_list(labels)
    if len(all_entries) < folds:
        raise typer.BadParameter(
            f"Only {len(all_entries)} images — need at least {folds}."
        )

    dev = resolve_device(device)
    hp_kwargs = dict(
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        batch_size=batch_size,
        scheduler_name=scheduler,
    )

    if HYPERPARAMS_PATH.exists():
        print(f"Loaded tuned defaults from {HYPERPARAMS_PATH}")
    print(f"Device: {dev}")
    print(f"Total: {len(all_entries)}  |  Folds: {folds}  |  Epochs: {epochs}")
    print(
        f"Hyperparams: lr={lr:.0e}  wd={weight_decay:.0e}  "
        f"dropout={dropout}  bs={batch_size}  sched={scheduler}\n"
    )

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    losses: list[float] = []
    accs: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_entries)):
        train = [all_entries[i] for i in train_idx]
        val = [all_entries[i] for i in val_idx]
        print(
            f"Fold {fold + 1}/{folds}  (train={len(train)}, val={len(val)}) ",
            end="",
            flush=True,
        )
        val_loss, val_acc = train_and_evaluate(
            train_entries=train,
            val_entries=val,
            crops_dir=crops,
            epochs=epochs,
            device=dev,
            seed=seed + fold,
            **hp_kwargs,
        )
        losses.append(val_loss)
        accs.append(val_acc)
        print(f"→ loss={val_loss:.4f}  acc={val_acc:.2%}")

    mean_loss, std_loss = float(np.mean(losses)), float(np.std(losses))
    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))

    print(f"\n{'=' * 50}")
    header = f"{'Fold':>5}  {'Val Loss':>9}  {'Val Acc':>8}"
    print(header)
    print("-" * len(header))
    for i, (loss, acc) in enumerate(zip(losses, accs)):
        print(f"{i + 1:>5}  {loss:>9.4f}  {acc:>7.2%}")
    print("-" * len(header))
    print(f"{'Mean':>5}  {mean_loss:>9.4f}  {mean_acc:>7.2%}")
    print(f"{'±Std':>5}  {std_loss:>9.4f}  {std_acc:>7.2%}")
    print(f"\nPer-pin accuracy: {mean_acc:.2%} ± {std_acc:.2%}")

    if not retrain:
        print("\nSkipping final retrain (--no-retrain).")
        return

    print(f"\nRetraining on all {len(all_entries)} images for {epochs} epochs...")
    best_loss = retrain_all(all_entries, crops, output, dev, epochs, seed, **hp_kwargs)
    print(f"  Final train loss: {best_loss:.4f}")
    print(f"  Saved to {output.resolve()}")


# ── Debugging ─────────────────────────────────────────────────────────────


@app.command("debug-crops")
def debug_crops(
    image: Annotated[
        Path, typer.Argument(help="Path to the scanned score sheet image.")
    ],
    model: Annotated[
        Optional[Path], typer.Option(help="YOLO detector weights (.pt).")
    ] = None,
    classifier_model: Annotated[
        Optional[Path], typer.Option(help="CNN classifier weights (.pt).")
    ] = None,
    confidence: Annotated[
        float, typer.Option(help="Minimum detection confidence.")
    ] = 0.5,
    output: Annotated[Path, typer.Option(help="Output directory.")] = Path(
        "debug_crops"
    ),
) -> None:
    """Save cropped pin diagrams and run CNN classification for debugging."""
    import csv

    import cv2

    from .classify import classify_pins_batch_with_confidence, load_classifier
    from .detect import (
        crop_detections,
        detect_pin_diagrams,
        draw_detections,
        load_model,
        sort_detections,
    )
    from .pipeline import DEFAULT_CLASSIFIER_PATH, DEFAULT_DETECTOR_PATH

    model_path = model or DEFAULT_DETECTOR_PATH
    classifier_path = classifier_model or DEFAULT_CLASSIFIER_PATH

    _require_path(model_path, "Detector weights")
    _require_path(image, "Image")

    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image))
    if img is None:
        raise typer.BadParameter(f"Could not load image: {image}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sorted_dets = sort_detections(
        detect_pin_diagrams(
            load_model(model_path), img, confidence_threshold=confidence
        )
    )
    print(f"Detected {len(sorted_dets)} pin diagrams")

    cv2.imwrite(str(output / "annotated_full.jpg"), draw_detections(img, sorted_dets))
    print(f"Saved annotated image to {output / 'annotated_full.jpg'}")

    crops = crop_detections(gray, sorted_dets)
    names = [f"c{d.column:02d}_r{d.row:02d}" for d in sorted_dets]
    for name, crop in zip(names, crops):
        cv2.imwrite(str(raw_dir / f"{name}.png"), crop)

    # Classify if weights exist
    has_classifier = classifier_path.exists()
    classifications: list[tuple[list[int], float]] = []
    if has_classifier:
        cnn, dev = load_classifier(classifier_path)
        classifications = classify_pins_batch_with_confidence(cnn, crops, device=dev)
    else:
        print(f"\nClassifier not found at {classifier_path} — skipping classification.")

    header_line = f"{'Name':<20} {'Size':<12}"
    if has_classifier:
        header_line += f" {'Pins':>10} {'Score':>6} {'Conf':>6}"
    print(f"\n{header_line}")
    print("-" * len(header_line))

    csv_rows: list[dict[str, str | int | float]] = []
    for i, (det, crop) in enumerate(zip(sorted_dets, crops)):
        row_data: dict[str, str | int | float] = {
            "name": names[i],
            "width": crop.shape[1],
            "height": crop.shape[0],
            "column": det.column,
            "row": det.row,
            "det_confidence": round(det.confidence, 4),
        }
        line = f"{names[i]:<20} {crop.shape[1]}x{crop.shape[0]:<12}"
        if has_classifier and i < len(classifications):
            pins, cls_conf = classifications[i]
            pins_str = "".join(str(p) for p in pins)
            line += f" {pins_str:>10} {sum(pins):>6} {cls_conf:>6.2f}"
            row_data.update(
                pins=pins_str, score=sum(pins), cls_confidence=round(cls_conf, 4)
            )
        print(line)
        csv_rows.append(row_data)

    if csv_rows:
        csv_path = output / "predictions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nPredictions saved to {csv_path}")

    print(f"\nAll crops saved to {output.resolve()}")


# ── Labeling & Validation ────────────────────────────────────────────────


@app.command()
def label(
    crops: CropsOpt = Path("debug_crops/raw"),
    classifier_model: Annotated[
        Optional[Path], typer.Option(help="CNN weights for suggestions.")
    ] = None,
    port: Annotated[int, typer.Option(help="Server port.")] = 8787,
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Don't auto-open browser.")
    ] = False,
) -> None:
    """Open the labeling UI to annotate ground-truth pin states."""
    import json as _json
    import webbrowser
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    from urllib.parse import urlparse

    from .classify import classify_pins_batch_with_confidence, load_classifier
    from .labels import load_labels_as_dict, save_labels
    from .pipeline import DEFAULT_CLASSIFIER_PATH

    classifier_path = classifier_model or DEFAULT_CLASSIFIER_PATH

    _require_path(
        crops, "Crops directory", "Run `pinsheet-scanner debug-crops <image>` first."
    )

    crop_names = sorted(p.name for p in crops.glob("*.png"))
    if not crop_names:
        raise typer.BadParameter(f"No .png files found in {crops}.")

    labels_path = crops.parent / "labels.csv"
    print(f"Found {len(crop_names)} crops in {crops}")

    existing_labels = load_labels_as_dict(labels_path)
    labeled = sum(1 for n in crop_names if n in existing_labels)
    print(f"Already labeled: {labeled}/{len(crop_names)}")

    # CNN predictions as suggestions
    predictions: dict[str, dict] = {}
    if classifier_path.exists():
        import cv2

        print("Running CNN for initial suggestions...")
        cnn, dev = load_classifier(classifier_path)
        images = [
            img
            for name in crop_names
            if (img := cv2.imread(str(crops / name), cv2.IMREAD_GRAYSCALE)) is not None
        ]
        results = classify_pins_batch_with_confidence(cnn, images, device=dev)
        for name, (pins, conf) in zip(crop_names, results):
            predictions[name] = {"pins": pins, "conf": round(conf, 4)}
        print(f"  Got predictions for {len(predictions)} crops")
    else:
        print(f"  Classifier not found at {classifier_path} — no suggestions.")

    def make_handler():
        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    page = _LABEL_HTML
                    page = page.replace("/*CROPS_JSON*/", _json.dumps(crop_names))
                    page = page.replace(
                        "/*PREDICTIONS_JSON*/", _json.dumps(predictions)
                    )
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

    server = HTTPServer(("127.0.0.1", port), make_handler())
    url = f"http://127.0.0.1:{port}"
    print(f"\nLabeler running at {url}")
    print("Press Ctrl+C to stop.\n")

    if not no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        labeled = sum(1 for n in crop_names if n in existing_labels)
        print(f"\n\nLabels saved to {labels_path}")
        print(f"Progress: {labeled}/{len(crop_names)} labeled")


@app.command()
def accuracy(
    crops: CropsOpt = Path("debug_crops/raw"),
    labels: LabelsOpt = Path("debug_crops/labels.csv"),
    classifier_model: Annotated[
        Optional[Path], typer.Option(help="CNN classifier weights (.pt).")
    ] = None,
) -> None:
    """Compare ground-truth labels against CNN predictions."""
    import cv2

    from .classify import classify_pins_batch_with_confidence, load_classifier
    from .constants import NUM_PINS
    from .labels import load_labels_as_dict
    from .pipeline import DEFAULT_CLASSIFIER_PATH

    classifier_path = classifier_model or DEFAULT_CLASSIFIER_PATH

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
    print(f"  Per-pin accuracy:    {correct_pins}/{total_pins} ({pin_acc:.1f}%)")
    print(
        f"  Per-diagram accuracy: {correct_diagrams}/{total_diagrams} ({diag_acc:.1f}%)"
    )

    print("\n  Per-position accuracy:")
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


# ── Labeler HTML (kept as a module-level constant) ────────────────────────

_LABEL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pin Labeler</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; display: flex; flex-direction: column;
         align-items: center; min-height: 100vh; padding: 20px; }
  h1 { font-size: 1.3rem; margin-bottom: 8px; color: #a0a0c0; font-weight: 500; }
  .progress { font-size: 0.85rem; color: #707090; margin-bottom: 16px; }
  .progress-bar { width: 320px; height: 4px; background: #2a2a4a; border-radius: 2px;
                  margin-bottom: 20px; }
  .progress-fill { height: 100%; background: #6c63ff; border-radius: 2px; transition: width 0.3s; }
  .card { background: #16213e; border-radius: 12px; padding: 24px; width: 360px;
          box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
  .crop-name { text-align: center; font-size: 0.9rem; color: #8888aa; margin-bottom: 12px;
               font-family: monospace; }
  .crop-container { display: flex; justify-content: center; margin-bottom: 20px; }
  .crop-container img { width: 200px; height: auto; image-rendering: pixelated;
                        border-radius: 6px; border: 2px solid #2a2a4a; }
  .pin-grid { position: relative; width: 220px; height: 220px; margin: 0 auto 20px; }
  .pin { position: absolute; width: 40px; height: 40px; border-radius: 50%;
         border: 3px solid #4a4a6a; background: transparent; cursor: pointer;
         display: flex; align-items: center; justify-content: center;
         font-size: 0.75rem; font-weight: 600; color: #6a6a8a;
         transition: all 0.15s ease; transform: translate(-50%, -50%); }
  .pin:hover { border-color: #6c63ff; transform: translate(-50%, -50%) scale(1.1); }
  .pin.down { background: #6c63ff; border-color: #6c63ff; color: #fff; }
  .pin .key-hint { position: absolute; bottom: -16px; font-size: 0.6rem; color: #505070; }
  .nav { display: flex; gap: 10px; justify-content: center; margin-bottom: 12px; }
  .nav button { padding: 8px 20px; border: none; border-radius: 6px; cursor: pointer;
                font-size: 0.85rem; font-weight: 500; transition: background 0.15s; }
  .btn-prev, .btn-next { background: #2a2a4a; color: #c0c0d0; }
  .btn-prev:hover, .btn-next:hover { background: #3a3a5a; }
  .btn-save { background: #6c63ff; color: #fff; }
  .btn-save:hover { background: #5a52dd; }
  .btn-prev:disabled, .btn-next:disabled { opacity: 0.3; cursor: default; }
  .score { text-align: center; font-size: 1.1rem; color: #6c63ff; margin-bottom: 12px;
           font-weight: 600; }
  .conf { text-align: center; font-size: 0.75rem; color: #505070; margin-bottom: 16px; }
  .status { text-align: center; font-size: 0.8rem; margin-top: 12px; min-height: 1.2em; }
  .status.saved { color: #4caf50; }
  .status.error { color: #f44336; }
  .done-banner { text-align: center; padding: 40px; }
  .done-banner h2 { color: #4caf50; margin-bottom: 12px; }
  .labeled-mark { color: #4caf50; font-size: 0.75rem; }
  .keyboard-help { font-size: 0.7rem; color: #404060; text-align: center;
                   margin-top: 16px; line-height: 1.6; }
</style>
</head>
<body>
<h1>Pin Labeler</h1>
<div class="progress" id="progress-text"></div>
<div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
<div class="card" id="card"></div>
<div class="keyboard-help">
  <b>Keys:</b> 1-9 toggle pins &middot; &larr;&rarr; navigate &middot; Enter save &amp; next
</div>

<script>
const CROPS = /*CROPS_JSON*/;
const PREDICTIONS = /*PREDICTIONS_JSON*/;
const LABELS = /*LABELS_JSON*/;

let idx = 0;
for (let i = 0; i < CROPS.length; i++) {
  if (!LABELS[CROPS[i]]) { idx = i; break; }
}

const PIN_POS = [
  [50, 8], [30, 30], [70, 30], [10, 52], [50, 52], [90, 52], [30, 74], [70, 74], [50, 96]
];

let pins = [0,0,0,0,0,0,0,0,0];

function getInitialPins(name) {
  if (LABELS[name]) return [...LABELS[name]];
  if (PREDICTIONS[name]) return [...PREDICTIONS[name].pins];
  return [0,0,0,0,0,0,0,0,0];
}

function render() {
  const total = CROPS.length;
  const labeled = Object.keys(LABELS).length;
  document.getElementById("progress-text").textContent =
    `${labeled} / ${total} labeled` + (labeled === total ? " ✓" : "");
  document.getElementById("progress-fill").style.width =
    `${(labeled / total * 100).toFixed(1)}%`;

  if (total === 0) {
    document.getElementById("card").innerHTML =
      '<div class="done-banner"><h2>No crops found</h2></div>';
    return;
  }

  const name = CROPS[idx];
  pins = getInitialPins(name);
  const pred = PREDICTIONS[name];
  const isLabeled = !!LABELS[name];

  let html = `<div class="crop-name">${name} ${isLabeled ? '<span class="labeled-mark">✓ labeled</span>' : ''}</div>`;
  html += `<div class="crop-container"><img src="/crop/${name}" alt="${name}"></div>`;
  html += `<div class="score" id="score">Score: ${pins.reduce((a,b)=>a+b,0)}</div>`;
  if (pred) {
    html += `<div class="conf">CNN confidence: ${(pred.conf * 100).toFixed(0)}%</div>`;
  }
  html += `<div class="pin-grid">`;
  for (let i = 0; i < 9; i++) {
    const [x, y] = PIN_POS[i];
    html += `<div class="pin ${pins[i] ? 'down' : ''}" style="left:${x}%;top:${y}%"
                  onclick="togglePin(${i})" id="pin${i}">
               ${i}<span class="key-hint">${i+1}</span>
             </div>`;
  }
  html += `</div>`;
  html += `<div class="nav">`;
  html += `<button class="btn-prev" onclick="go(-1)" ${idx===0?'disabled':''}>← Prev</button>`;
  html += `<button class="btn-save" onclick="saveAndNext()">Save & Next</button>`;
  html += `<button class="btn-next" onclick="go(1)" ${idx>=total-1?'disabled':''}>Next →</button>`;
  html += `</div>`;
  html += `<div class="status" id="status"></div>`;

  document.getElementById("card").innerHTML = html;
}

function togglePin(i) {
  pins[i] = pins[i] ? 0 : 1;
  const el = document.getElementById("pin" + i);
  el.classList.toggle("down", !!pins[i]);
  document.getElementById("score").textContent = "Score: " + pins.reduce((a,b)=>a+b,0);
}

function go(delta) {
  const next = idx + delta;
  if (next >= 0 && next < CROPS.length) { idx = next; render(); }
}

async function saveAndNext() {
  const name = CROPS[idx];
  const body = JSON.stringify({ filename: name, pins: [...pins] });
  try {
    const resp = await fetch("/save", { method: "POST", headers: {"Content-Type":"application/json"}, body });
    if (!resp.ok) throw new Error(resp.statusText);
    LABELS[name] = [...pins];
    const el = document.getElementById("status");
    el.textContent = "Saved ✓";
    el.className = "status saved";
    setTimeout(() => {
      if (idx < CROPS.length - 1) { idx++; render(); }
      else { render(); el.textContent = "All done!"; el.className = "status saved"; }
    }, 250);
  } catch (e) {
    const el = document.getElementById("status");
    el.textContent = "Error: " + e.message;
    el.className = "status error";
  }
}

document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") go(-1);
  else if (e.key === "ArrowRight") go(1);
  else if (e.key === "Enter") saveAndNext();
  else if (e.key >= "1" && e.key <= "9") togglePin(parseInt(e.key) - 1);
});

render();
</script>
</body>
</html>"""
