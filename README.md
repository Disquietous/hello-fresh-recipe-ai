# HelloFresh Recipe AI - YOLOv8 Image Analysis

A complete YOLOv8-based computer vision project for food ingredient detection and recipe analysis.

## Features

- **Object Detection**: Detect food ingredients in images and videos using YOLOv8
- **Custom Training**: Train custom models on your own food datasets
- **Batch Processing**: Process multiple images and videos efficiently
- **Model Export**: Export trained models to various formats (ONNX, TensorRT, etc.)
- **Data Utilities**: Tools for dataset preparation and validation

## Project Structure

```
hello-fresh-recipe-ai/
├── data/                    # Dataset storage
│   ├── raw/                # Original images and videos
│   ├── processed/          # Preprocessed data
│   └── annotations/        # YOLO format labels
├── models/                 # Model storage
│   ├── pretrained/         # Downloaded YOLOv8 models
│   └── custom/            # Trained custom models
├── results/               # Output results
│   ├── images/            # Processed images
│   ├── videos/            # Processed videos
│   └── metrics/           # Training metrics
├── src/                   # Source code
│   ├── detect.py          # Main detection script
│   ├── train.py           # Training script
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── notebooks/             # Jupyter notebooks
└── logs/                  # Training logs
```

## Setup Instructions

### 1. Environment Setup

**Windows:**
```bash
# Run the setup script
setup_env.bat
```

**Linux/macOS:**
```bash
# Make script executable and run
chmod +x setup_env.sh
./setup_env.sh
```

**Manual Setup:**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pretrained Models

YOLOv8 models will be automatically downloaded when first used. Available models:
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, most accurate)

## Usage

### Object Detection

**Single Image:**
```bash
python src/detect.py path/to/image.jpg --output results/detected_image.jpg
```

**Video Processing:**
```bash
python src/detect.py path/to/video.mp4 --output results/detected_video.mp4
```

**Batch Processing:**
```bash
python src/detect.py data/raw/images/ --output results/batch_results/ --batch
```

**Advanced Options:**
```bash
python src/detect.py image.jpg --model yolov8s.pt --conf 0.5 --save-crops
```

### Custom Model Training

**Prepare Dataset:**
1. Organize your dataset:
   ```
   data/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

2. Create data configuration:
   ```bash
   python src/train.py --data data/ --create-config --classes apple banana orange
   ```

**Train Model:**
```bash
python src/train.py --data configs/food_data.yaml --epochs 100 --batch-size 16
```

**Train with Custom Parameters:**
```bash
python src/train.py --data configs/food_data.yaml --model-size s --epochs 200 --batch-size 32 --lr 0.001
```

**Validate Model:**
```bash
python src/train.py --data configs/food_data.yaml --validate-only --model-path models/custom/best.pt
```

**Export Model:**
```bash
python src/train.py --export onnx --model-path models/custom/best.pt
```

## Configuration

### Data Configuration (`configs/food_data.yaml`)
Defines dataset paths and class names for training.

### Training Configuration (`configs/training_config.yaml`)
Contains default training parameters and hyperparameters.

## Development

### Data Preparation
```python
from src.utils.data_utils import split_dataset, validate_dataset_structure

# Split dataset into train/val/test
split_dataset('raw_images/', 'raw_labels/', 'data/', (0.7, 0.2, 0.1))

# Validate dataset structure
validate_dataset_structure('data/')
```

### Jupyter Notebooks
Use the `notebooks/` directory for experimentation and analysis:
```bash
jupyter notebook notebooks/
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support recommended for training
- **RAM**: 8GB+ for training, 4GB+ for inference
- **Storage**: 10GB+ for datasets and models

## Model Performance

Training metrics and validation results are saved in:
- `results/metrics/` - Training logs and metrics
- `logs/` - TensorBoard logs

View training progress:
```bash
tensorboard --logdir logs/
```

## Troubleshooting

### Common Issues

1. **CUDA not available**: Install PyTorch with CUDA support
2. **Out of memory**: Reduce batch size or image size
3. **Dataset not found**: Check paths in data configuration file

### Getting Help

- Check the logs in the `logs/` directory
- Verify dataset structure with validation tools
- Ensure all dependencies are installed correctly

## License

This project is for educational and development purposes.