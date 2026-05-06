# Dataset Generation — 8-State Block Detection

## Image Format

Each image is **16 pixels tall × 128 pixels wide**, divided into 8 blocks of 16×16 pixels:

| Block | Columns | Meaning (active = white) |
|-------|---------|--------------------------|
| 0 | 0–15 | State 0 |
| 1 | 16–31 | State 1 |
| 2 | 32–47 | State 2 |
| 3 | 48–63 | State 3 |
| 4 | 64–79 | State 4 |
| 5 | 80–95 | State 5 |
| 6 | 96–111 | State 6 |
| 7 | 112–127 | Normal (exclusive) |

- Active block = pure white `(255, 255, 255)`
- Inactive block = pure black `(0, 0, 0)`
- States 0–6 can combine freely (2^7 = 128 combinations)
- State 7 is active ONLY when all states 0–6 are inactive

## Class System

129 classes total:
- 128 classes for combinations of states 0–6, named like `s0_s2_s4`
- 1 class `normal` for state 7 only (blocks 0–6 black, block 7 white)
- Plus `none` (all blocks black — no states active)

The output `labels.txt` lists all class names in alphabetical order. **Line number = class index** (0-indexed).

## Usage

```bash
cd image_classification

# Generate dataset
python generate_states_dataset.py \
    --train-count 12900 \     # 100 images/class for training
    --val-count 2580 \        # 20 images/class for validation
    --test-count 2580 \       # 20 images/class for testing
    --output-dir ./datasets/states_dataset \
    --seed 42
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-count` | 200 | Total training images (divided evenly across 129 classes) |
| `--val-count` | 50 | Total validation images |
| `--test-count` | 50 | Total test images |
| `--output-dir` | `./datasets/states_dataset` | Output directory |
| `--seed` | 42 | Random seed for reproducibility |

## Output Structure

```
datasets/states_dataset/
├── labels.txt                  # 129 class names, one per line
├── train/
│   ├── none/img_0000.png ...   # All 8 blocks black
│   ├── normal/img_0000.png ... # Only block 7 white
│   ├── s0/img_0000.png ...     # Only block 0 white
│   ├── s0_s1/img_0000.png ...
│   └── ... (129 directories)
├── val/    (same structure)
└── test/   (same structure)
```

## Script Location

```
image_classification/generate_states_dataset.py
```

No external dependencies beyond `numpy` and `Pillow`. Call it directly with any Python 3.8+.
