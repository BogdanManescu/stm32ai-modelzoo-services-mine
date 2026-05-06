# Migration Guide — Replicating on Another Machine

## Files to Copy

These are the files you need to replicate the full pipeline on another machine:

### Core Scripts (your custom code)

```
image_classification/
├── generate_states_dataset.py              # Dataset generation
├── requirements_states_detection.txt        # Python dependencies
├── config_file_examples/
│   └── training_states_config.yaml          # TF training config
├── config_file_examples_pt/
│   └── training_states_config.yaml          # PyTorch training config
├── pt/src/models/
│   └── states_cnn.py                        # Custom PyTorch model
└── docs/
    ├── README_STATES_DATASET.md
    ├── README_STATES_TRAINING_PYTORCH.md
    └── README_STATES_TRAINING_TENSORFLOW.md
```

### Modified Pipeline Files

You modified ONE existing file. You must apply this change on the target machine:

**`image_classification/pt/wrappers/models/custom_models/custom_models.py`**

Two additions:
1. **Import** (top of file, after existing imports):
   ```python
   from image_classification.pt.src.models.states_cnn import StatesCNN
   ```

2. **Model registration** (in the `MODEL_FNS` dict):
   ```python
   'states_cnn_pt': (StatesCNN, {'dropout': 0.2}),
   ```

### The Full Repository

Everything else comes from the base STM32 model zoo repository. On the target machine:

```bash
git clone https://github.com/STMicroelectronics/stm32ai-modelzoo-services.git
cd stm32ai-modelzoo-services
```

Then copy your custom files into the `image_classification/` directory, apply the one-line change to `custom_models.py`, and install dependencies.

## Setup Checklist

```bash
# 1. Clone the repo
git clone https://github.com/STMicroelectronics/stm32ai-modelzoo-services.git
cd stm32ai-modelzoo-services/image_classification

# 2. Create conda environment (recommended)
conda create -n stmai2 python=3.10 -y
conda activate stmai2

# 3. Install PyTorch with CUDA (for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Install remaining dependencies
pip install -r requirements_states_detection.txt

# 5. Copy your custom files into place
#    - generate_states_dataset.py → image_classification/
#    - training_states_config.yaml → image_classification/config_file_examples/
#    - training_states_config.yaml → image_classification/config_file_examples_pt/
#    - states_cnn.py → image_classification/pt/src/models/
#    - Apply the custom_models.py edit (see above)

# 6. Generate dataset
python generate_states_dataset.py \
    --train-count 12900 --val-count 2580 --test-count 2580 \
    --output-dir ./datasets/states_dataset

# 7. Train (PyTorch GPU)
python stm32ai_main.py \
    --config-path ./config_file_examples_pt/ \
    --config-name training_states_config.yaml

# or: Train (TensorFlow CPU)
python stm32ai_main.py \
    --config-path ./config_file_examples/ \
    --config-name training_states_config.yaml
```

## Package Version Reference

Installed versions in the working `stmai2` conda environment:

| Package | Version | Required for |
|---------|---------|-------------|
| Python | 3.12 | Both |
| torch | 2.7.1+cu128 | PyTorch GPU |
| torchvision | 0.22.1+cu128 | PyTorch GPU |
| timm | 1.0.19 | PyTorch data loading |
| tensorflow | 2.18.0 | TF CPU |
| tensorflow-intel | 2.18.0 | TF CPU (Intel oneDNN) |
| keras | 3.8.0 | TF |
| tf2onnx | 1.16.1 | TF → ONNX conversion |
| hydra-core | 1.3.2 | Config management |
| omegaconf | 2.3.0 | Config parsing |
| neural_compressor | 3.4.1 | Quantization |
| mlflow | 2.20.0 | Experiment tracking |
| clearml | 1.16.5 | Experiment tracking |
| onnx | 1.16.1 | Model exchange |
| onnxruntime | 1.20.1 | ONNX inference |
| opencv-python | 4.11.0.86 | Image I/O |
| pillow | 12.1.1 | Image I/O |
| numpy | 1.26.4 | Math |
| munch | 4.0.0 | Attribute dict access |
