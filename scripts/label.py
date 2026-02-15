"""Local labeling server for ground-truth pin annotation.

Opens a browser UI where you can view each crop from debug_crops/raw/,
toggle which pins are knocked down, and save labels to debug_crops/labels.csv.

CNN predictions are pre-loaded as starting suggestions so you only need
to correct mistakes.

Usage:
    uv run python -m scripts.label
    uv run python -m scripts.label --crops debug_crops/raw --port 8787
"""

from __future__ import annotations

import argparse
import csv
import json
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from pinsheet_scanner.classify import (
    classify_pins_batch_with_confidence,
    load_classifier,
)
from pinsheet_scanner.constants import NUM_PINS
from pinsheet_scanner.pipeline import DEFAULT_CLASSIFIER_PATH

LABELS_FILENAME = "labels.csv"

HTML_PAGE = r"""<!DOCTYPE html>
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
  .info { text-align: center; font-size: 0.75rem; color: #505070; margin-top: 8px; }
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
// find first unlabeled
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


def _load_existing_labels(labels_path: Path) -> dict[str, list[int]]:
    """Load existing labels from CSV, if it exists."""
    labels: dict[str, list[int]] = {}
    if not labels_path.exists():
        return labels
    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            pins = [int(row[f"p{i}"]) for i in range(NUM_PINS)]
            labels[row["filename"]] = pins
    return labels


def _save_labels(labels_path: Path, labels: dict[str, list[int]]) -> None:
    """Write all labels to CSV (atomic overwrite)."""
    tmp = labels_path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + [f"p{i}" for i in range(NUM_PINS)])
        for filename in sorted(labels):
            writer.writerow([filename] + labels[filename])
    tmp.replace(labels_path)


def _get_predictions(
    crops_dir: Path, crop_names: list[str], classifier_path: Path
) -> dict[str, dict]:
    """Run CNN on all crops and return {name: {pins, conf}}."""
    if not classifier_path.exists():
        print(f"  Classifier not found at {classifier_path} — no suggestions.")
        return {}

    import cv2

    model, device = load_classifier(classifier_path)
    images = []
    for name in crop_names:
        img = cv2.imread(str(crops_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    results = classify_pins_batch_with_confidence(model, images, device=device)

    preds: dict[str, dict] = {}
    for name, (pins, conf) in zip(crop_names, results):
        preds[name] = {"pins": pins, "conf": round(conf, 4)}
    return preds


def _make_handler(
    crops_dir: Path,
    labels_path: Path,
    labels: dict[str, list[int]],
    crop_names: list[str],
    predictions: dict[str, dict],
):
    """Create a request handler class with the data baked in."""

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # silence request logs

        def do_GET(self):
            parsed = urlparse(self.path)

            if parsed.path == "/":
                page = HTML_PAGE
                page = page.replace("/*CROPS_JSON*/", json.dumps(crop_names))
                page = page.replace("/*PREDICTIONS_JSON*/", json.dumps(predictions))
                page = page.replace("/*LABELS_JSON*/", json.dumps(labels))
                self._respond(200, "text/html", page.encode())

            elif parsed.path.startswith("/crop/"):
                name = parsed.path[6:]
                img_path = crops_dir / name
                if img_path.exists() and img_path.parent == crops_dir:
                    self._respond(200, "image/png", img_path.read_bytes())
                else:
                    self._respond(404, "text/plain", b"Not found")
            else:
                self._respond(404, "text/plain", b"Not found")

        def do_POST(self):
            if self.path == "/save":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                filename = body["filename"]
                pins = body["pins"]

                if filename not in {n for n in crop_names}:
                    self._respond(400, "text/plain", b"Unknown crop")
                    return

                labels[filename] = pins
                _save_labels(labels_path, labels)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Label pin crops for ground-truth validation."
    )
    p.add_argument(
        "--crops", type=Path, default=Path("debug_crops/raw"), help="Crops directory."
    )
    p.add_argument(
        "--classifier-model",
        type=Path,
        default=None,
        help="CNN classifier weights for pre-populating suggestions.",
    )
    p.add_argument("--port", type=int, default=8787, help="Server port.")
    p.add_argument(
        "--no-open", action="store_true", help="Don't auto-open the browser."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    crops_dir: Path = args.crops
    classifier_path: Path = args.classifier_model or DEFAULT_CLASSIFIER_PATH
    labels_path = crops_dir.parent / LABELS_FILENAME

    if not crops_dir.exists():
        raise FileNotFoundError(
            f"Crops directory not found at {crops_dir}. "
            "Run `just debug-crops <image>` first."
        )

    crop_names = sorted(p.name for p in crops_dir.glob("*.png"))
    if not crop_names:
        raise FileNotFoundError(f"No .png files found in {crops_dir}.")

    print(f"Found {len(crop_names)} crops in {crops_dir}")

    # Load existing labels
    labels = _load_existing_labels(labels_path)
    labeled = sum(1 for n in crop_names if n in labels)
    print(f"Already labeled: {labeled}/{len(crop_names)}")

    # Get CNN predictions as suggestions
    print("Running CNN for initial suggestions...")
    predictions = _get_predictions(crops_dir, crop_names, classifier_path)
    if predictions:
        print(f"  Got predictions for {len(predictions)} crops")

    handler = _make_handler(crops_dir, labels_path, labels, crop_names, predictions)
    server = HTTPServer(("127.0.0.1", args.port), handler)

    url = f"http://127.0.0.1:{args.port}"
    print(f"\nLabeler running at {url}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n\nLabels saved to {labels_path}")
        labeled = sum(1 for n in crop_names if n in labels)
        print(f"Progress: {labeled}/{len(crop_names)} labeled")


if __name__ == "__main__":
    main()
