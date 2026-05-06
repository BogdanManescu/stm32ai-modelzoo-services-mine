# Training — TensorFlow (CPU / Intel oneDNN)

Uses TensorFlow with Intel oneDNN optimizations for CPU inference. The `st_mnistv1` model (~17K params) is a compact CNN already registered in the model zoo.

## Prerequisites

1. Python 3.10+ with packages installed:

```bash
pip install tensorflow==2.18.0 tensorflow-intel==2.18.0 keras==3.8.0
pip install tf2onnx hydra-core omegaconf neural_compressor mlflow clearml munch
pip install numpy pillow pyyaml tqdm onnx onnxruntime opencv-python pandas
pip install scipy scikit-learn scikit-image matplotlib h5py protobuf
```

Or use the bundled requirements file:
```bash
pip install -r requirements_states_detection.txt
```

2. The repository root must be on `PYTHONPATH` so the `common` and `api` modules are importable. This is handled automatically by `stm32ai_main.py`.

## Verify TensorFlow

```bash
python -c "import tensorflow as tf; print('TF version:', tf.__version__)"
```

GPU is not required for this path — TensorFlow will use CPU with Intel oneDNN optimizations.

## Config File

`config_file_examples/training_states_config.yaml` — key settings:

| Setting | Value | Notes |
|---------|-------|-------|
| `model.model_name` | `st_mnistv1` | Built-in model zoo model |
| `model.input_shape` | `(16, 128, 3)` | H×W×C format |
| `model.pretrained` | `False` | Train from scratch |
| `dataset.dataset_name` | `custom_dataset` | Uses class-subdirectory format |
| `dataset.classes_file_path` | `./datasets/states_dataset/labels.txt` | |
| `dataset.training_path` | `./datasets/states_dataset/train` | |
| `dataset.validation_path` | `./datasets/states_dataset/val` | |
| `preprocessing.rescaling.scale` | `1/255` | Scale [0,255] → [0,1] |
| `training.dropout` | `0.2` | |
| `training.batch_size` | `32` | |
| `training.epochs` | `50` | |

> Unlike PyTorch, TF model names do NOT need a special suffix. `st_mnistv1` works as-is.

## Run Training

```bash
cd image_classification

# Quick test (1 epoch)
python stm32ai_main.py \
    --config-path ./config_file_examples/ \
    --config-name training_states_config.yaml \
    training.epochs=1 training.batch_size=16

# Full training
python stm32ai_main.py \
    --config-path ./config_file_examples/ \
    --config-name training_states_config.yaml

# Full training with overrides
python stm32ai_main.py \
    --config-path ./config_file_examples/ \
    --config-name training_states_config.yaml \
    training.epochs=100 training.batch_size=64
```

## Output

The trained model is saved at:
```
tf/src/experiments_outputs/<timestamp>/saved_models/st_mnistv1/
├── st_mnistv1.keras          # Keras model
└── ...
```

## Data Augmentation Note

The TF pipeline requires a `data_augmentation` section even if you don't want augmentation. A minimal brightness-only augmentation is included in the config:

```yaml
data_augmentation:
  random_brightness:
    factor: 0.05
```

This adds trivial variation to pure black/white images and can be disabled by removing the section if you update the pipeline code to handle the missing section.
