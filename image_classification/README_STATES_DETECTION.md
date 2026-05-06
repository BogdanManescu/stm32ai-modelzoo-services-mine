# States Detection — Image Classification on STM32N6

## Overview

This project trains an AI model to detect 8 human-condition states from a 16×128 pixel sensor image. The image is divided into 8 blocks of 16×16 pixels, each block encoding one state:

| Block | Column Range | State |
|-------|-------------|-------|
| 0 | 0–15 | High Body Temperature |
| 1 | 16–31 | Muscle Fatigue |
| 2 | 32–47 | EMG Fatigue |
| 3 | 48–63 | Poor Posture |
| 4 | 64–79 | High Environmental Temperature |
| 5 | 80–95 | High Stress |
| 6 | 96–111 | Eye Fatigue |
| 7 | 112–127 | Normal State |

- **Active state** = white block (255,255,255)
- **Inactive state** = black block (0,0,0)
- States 0–6 can be active in any combination (2^7 = 128 combos)
- State 7 (Normal) is active **only** when all states 0–6 are inactive

## Class Mapping

Since image classification outputs a single class label, each combination of active states 0–6 maps to a unique class, plus one class for Normal. Total: **129 classes**.

Class names encode which states are active:
- `none` — no states 0–6 active (all black)
- `s0` — only state 0 active
- `s0_s2_s4` — states 0, 2, and 4 active
- `normal` — only state 7 active (blocks 0–6 black, block 7 white)

The file `datasets/states_dataset/labels.txt` lists all 129 class names in alphabetical order. **Line number = class index** (0-indexed). For example:
```
line 0: none          → class index 0
line 1: normal        → class index 1
line 2: s0            → class index 2
...
line 128: s6          → class index 128
```

## File Reference

```
image_classification/
├── generate_states_dataset.py          # Dataset generation script
├── config_file_examples/
│   └── training_states_config.yaml     # Training configuration
└── datasets/
    └── states_dataset/                 # Generated dataset (you create this)
        ├── labels.txt                  # Class names (1 per line, sorted)
        ├── train/<class>/img_XXXX.png
        ├── val/<class>/img_XXXX.png
        └── test/<class>/img_XXXX.png
```

## Step 1: Generate the Dataset

```bash
cd image_classification

# Delete any previous dataset
rm -rf ./datasets/states_dataset

# Generate: 12900 train + 2580 val + 2580 test = 100 per class train, 20 per class val/test
python generate_states_dataset.py \
    --train-count 12900 \
    --val-count 2580 \
    --test-count 2580 \
    --output-dir ./datasets/states_dataset \
    --seed 42
```

## Step 2: Train the Model

```bash
cd image_classification

python stm32ai_main.py \
    --config-path ./config_file_examples/ \
    --config-name training_states_config.yaml
```

Override training parameters as needed:
```bash
python stm32ai_main.py \
    --config-path ./config_file_examples/ \
    --config-name training_states_config.yaml \
    training.epochs=100 \
    training.batch_size=64
```

The trained model is saved under:
```
tf/src/experiments_outputs/<timestamp>/saved_models/st_mnistv1/
```

## Step 3: Interpret Model Output

The model outputs an integer class index (0–128). Look up the active states in `datasets/states_dataset/labels.txt`:

```python
# Load labels
with open("datasets/states_dataset/labels.txt") as f:
    labels = [line.strip() for line in f]

# Model prediction
class_idx = model.predict(image)  # integer 0-128
class_name = labels[class_idx]

if class_name == "normal":
    print("Normal state")
elif class_name == "none":
    print("No states active")
else:
    active_states = [int(s[1:]) for s in class_name.split("_")]
    print(f"Active states: {active_states}")
```

## Model Info

- **Architecture**: `st_mnistv1` (~17K parameters)
- **Input**: 16×128×3 (height × width × channels), RGB
- **Output**: 129-class softmax
- **Framework**: TensorFlow 2.18 / Keras 3.8
