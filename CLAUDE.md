# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HelloFresh Recipe AI is a YOLOv8-based computer vision project for food ingredient detection and recipe analysis. The project uses Python with the Ultralytics YOLOv8 framework for object detection, custom model training, and batch processing of food images and videos.

## Technology Stack

- **Python 3.8+** - Main programming language
- **YOLOv8 (Ultralytics)** - Object detection framework
- **PyTorch** - Deep learning backend
- **OpenCV** - Image processing
- **NumPy, Pandas** - Data manipulation
- **Matplotlib, Seaborn** - Visualization
- **Jupyter** - Experimentation and analysis

## Development Commands

### Environment Setup
```bash
# Windows
setup_env.bat

# Linux/macOS
./setup_env.sh

# Manual activation
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### Core Operations
```bash
# Object detection on single image
python src/detect.py image.jpg --output results/detected.jpg

# Train custom model
python src/train.py --data configs/food_data.yaml --epochs 100

# Validate trained model
python src/train.py --data configs/food_data.yaml --validate-only --model-path models/custom/best.pt

# Export model to ONNX
python src/train.py --export onnx --model-path models/custom/best.pt
```

### Development Tools
```bash
# Run Jupyter notebooks
jupyter notebook notebooks/

# View training metrics
tensorboard --logdir logs/

# Validate dataset structure
python -c "from src.utils.data_utils import validate_dataset_structure; validate_dataset_structure('data/')"
```

## Project Architecture

### Core Components
- `src/detect.py` - Main detection script with FoodDetector class for inference
- `src/train.py` - Training script with FoodModelTrainer class for custom model training
- `src/utils/data_utils.py` - Dataset utilities for preprocessing and validation

### Data Flow
1. **Raw data** → `data/raw/` (original images/videos)
2. **Preprocessing** → `data/processed/` (resized, augmented)
3. **Annotations** → `data/annotations/` (YOLO format labels)
4. **Training** → `models/custom/` (trained models)
5. **Inference** → `results/` (detection outputs)

### Configuration System
- `configs/food_data.yaml` - Dataset configuration with class names and paths
- `configs/training_config.yaml` - Training hyperparameters and augmentation settings

## Important Patterns

### Dataset Structure
YOLO format with train/val/test splits:
```
data/
├── train/images/ + train/labels/
├── val/images/ + val/labels/
└── test/images/ + test/labels/
```

### Model Management
- Pretrained models auto-download to `models/pretrained/`
- Custom trained models save to `models/custom/`
- Best model saved as `best.pt`, last checkpoint as `last.pt`

### Error Handling
- Always validate dataset structure before training
- Check CUDA availability for GPU training
- Verify model paths exist before validation/export

## Development Notes

- Use `FoodDetector` class for inference workflows
- Use `FoodModelTrainer` class for training workflows  
- Data utilities in `src/utils/data_utils.py` for preprocessing
- All paths should be absolute or relative to project root
- Training automatically saves to `runs/detect/` or `runs/train/` subdirectories