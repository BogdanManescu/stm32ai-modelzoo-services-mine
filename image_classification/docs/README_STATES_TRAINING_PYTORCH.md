# Training — PyTorch (GPU)

Uses NVIDIA GPU via CUDA. The custom model `states_cnn_pt` is a tiny CNN (~32K params) designed specifically for 16×128 input images.

## Prerequisites

1. NVIDIA GPU with CUDA 12.x drivers
2. Python 3.10+ with packages installed:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install timm hydra-core omegaconf neural_compressor mlflow clearml munch
pip install numpy pillow pyyaml tqdm onnx onnxruntime opencv-python pandas
pip install scipy scikit-learn scikit-image matplotlib h5py protobuf
```

Or use the bundled requirements file:
```bash
pip install -r requirements_states_detection.txt
```

3. The repository root must be on `PYTHONPATH` so the `common` and `api` modules are importable. This is handled automatically by `stm32ai_main.py`.

## Verify GPU Access

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA: True
Device: NVIDIA GeForce RTX 5070  (or similar)
```

## Config File

`config_file_examples_pt/training_states_config.yaml` — key settings:

| Setting | Value | Notes |
|---------|-------|-------|
| `model.model_name` | `states_cnn_pt` | **Must** end with `_pt` for PyTorch |
| `model.input_shape` | `[3, 16, 128]` | C×H×W format |
| `model.pretrained` | `False` | No pretrained weights for custom model |
| `dataset.dataset_name` | `custom` | Uses class-subdirectory format |
| `dataset.num_classes` | `129` | |
| `dataset.training_path` | `./datasets/states_dataset/train` | |
| `dataset.validation_path` | `./datasets/states_dataset/val` | |
| `training.trainer_name` | `ic_trainer` | **Required** for PyTorch |
| `training.batch_size` | `32` | Adjust based on GPU memory |
| `training.epochs` | `50` | |
| `preprocessing.normalization.mean` | `[0, 0, 0]` | No ImageNet normalization |
| `preprocessing.normalization.std` | `[1, 1, 1]` | |

> **Critical**: PyTorch model names must end with `_pt`. The pipeline auto-detects framework from the suffix — without it, the config parser forces `framework: tf` regardless of the `framework` setting.

## Run Training

```bash
cd image_classification

# Quick test (1 epoch)
python stm32ai_main.py \
    --config-path ./config_file_examples_pt/ \
    --config-name training_states_config.yaml \
    training.epochs=1 training.batch_size=16

# Full training
python stm32ai_main.py \
    --config-path ./config_file_examples_pt/ \
    --config-name training_states_config.yaml

# Full training with overrides
python stm32ai_main.py \
    --config-path ./config_file_examples_pt/ \
    --config-name training_states_config.yaml \
    training.epochs=100 training.batch_size=64
```

## Output

The trained model is saved at:
```
pt/src/experiments_outputs/<timestamp>/saved_models/
├── checkpoint-0.pth.tar     # Training checkpoint
├── last.pth.tar              # Last epoch
├── model_best.pth.tar        # Best validation accuracy
├── summary.csv               # Training metrics
└── states_cnn_pt/            # Model files
```

**Note**: The post-training evaluation step may fail on ONNX export (16×128 is an unusual input size). This does not affect the saved `.pth.tar` checkpoint — the model trained correctly.

## Custom Model

The model is defined at `pt/src/models/states_cnn.py` and registered in `pt/wrappers/models/custom_models/custom_models.py`. Architecture:

```
Conv2d(3→16, stride 2) → BN → ReLU        # 16×128 → 16×8×64
Conv2d(16→32, stride 2) → BN → ReLU       # 16×8×64 → 32×4×32
Conv2d(32→64, stride 2) → BN → ReLU       # 32×4×32 → 64×2×16
AdaptiveAvgPool2d(1) → Flatten → Dropout → Linear(64→129)
```

~32K parameters in float32; ~8K after int8 quantization. Well within NPU limits.
