# Roadmap

## Current accuracy

Trained on 120 real labeled crops (100 train / 20 val) with heavy online
augmentation for 60 epochs.

| Metric               | Value               |
| -------------------- | ------------------- |
| Per-pin accuracy     | 96.9 % (1047/1080)  |
| Per-diagram accuracy | 75.0 % (90/120)     |
| Worst pin (8)        | 93.3 %              |
| Best pin (3)         | 100.0 %             |

## Next steps

### More labeled data

The single biggest accuracy lever. Scan additional score sheets, extract
crops, and label them:

```sh
just debug-crops new_sheet.jpg
just label
just retrain
just accuracy
```

Target: 300+ labeled crops for a meaningful train/val split.

### Threshold tuning

Many remaining errors are borderline calls at 50–70 % sigmoid confidence.
Options to explore:

- **Per-pin thresholds** — calibrate a separate threshold for each of the
  9 pin positions instead of a single 0.5 cutoff.
- **Test-time augmentation (TTA)** — classify each crop multiple times
  with random augmentation, average the probabilities.

### Pin 0 / Pin 8 investigation

These two positions (top and bottom of the diamond) account for most
errors. Investigate whether Otsu binarisation loses information at the
extremes of the diamond layout. Try:

- Training with and without binarisation
- Alternative preprocessing (adaptive threshold, CLAHE)
- Larger crop padding around the detection box

### Model improvements

- Experiment with input resolution (96×96 or 128×128 instead of 64×64)
- Try a slightly deeper architecture or residual connections
- Add learning rate warmup to the training schedule