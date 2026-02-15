# Roadmap

## Current Status

The project has been refactored to use a unified CLI architecture:
- **Unified CLI**: All commands now live under `pinsheet-scanner <subcommand>` using Typer (replacing individual scripts).
- **Eliminated scripts/ directory**: Training, tuning, and utility scripts consolidated into `src/pinsheet_scanner/cli.py`.
- **Shared training module**: `src/pinsheet_scanner/training.py` contains reusable training primitives (dataset, scheduler, hyperparams).
- **Improved maintainability**: No duplicated argparse boilerplate, auto-generated help, cleaner separation of concerns.

## Accuracy Metrics

Trained on 120 real labeled crops (100 train / 20 val) with heavy online augmentation for 60 epochs. Tested on the same 120 crops (overfitting expected, but indicates model capacity).

| Metric               | Value               |
| -------------------- | ------------------- |
| Per-pin accuracy     | 100.0 % (1080/1080) |
| Per-diagram accuracy | 100.0 % (120/120)   |
| Worst pin (8)        | 100.0 %             |
| Best pin (3)         | 100.0 %             |

**Note**: Perfect accuracy on training data; real-world performance needs validation on unseen sheets.

## Next Steps

### More labeled data

The single biggest accuracy lever. Scan additional score sheets, extract crops, and label them:

```sh
just debug-crops new_sheet.jpg
just label
just tune --trials 20
just kfold --folds 5
just accuracy
```

Target: 300+ labeled crops for a meaningful train/val split and robust generalization.

### Threshold tuning

Many remaining errors are borderline calls at 50–70 % sigmoid confidence.
Options to explore:

- **Per-pin thresholds** — calibrate a separate threshold for each of the 9 pin positions instead of a single 0.5 cutoff.
- **Test-time augmentation (TTA)** — classify each crop multiple times with random augmentation, average the probabilities.

### Pin 0 / Pin 8 investigation

These two positions (top and bottom of the diamond) account for most errors. Investigate whether Otsu binarisation loses information at the extremes of the diamond layout. Try:

- Training with and without binarisation
- Alternative preprocessing (adaptive threshold, CLAHE)
- Larger crop padding around the detection box

### Model improvements

- Experiment with input resolution (96×96 or 128×128 instead of 64×64)
- Try a slightly deeper architecture or residual connections
- Add learning rate warmup to the training schedule

### Software Quality

- Add CI/CD pipeline (GitHub Actions for tests, lint, typecheck)
- Expand test coverage (integration tests, CLI testing)
- Add logging/experiment tracking (wandb/mlflow)
- Improve CLI UX (progress bars with rich, configuration files)