# HelloFresh Recipe AI - Text Detection & Ingredient Recognition

A complete YOLOv8-based computer vision project for detecting and recognizing ingredient names, amounts, and units from recipe images and text.

## Features

- **Text Detection**: Detect text regions containing ingredient information using YOLOv8
- **OCR Integration**: Extract text using multiple OCR engines (EasyOCR, Tesseract, PaddleOCR)
- **Ingredient Parsing**: Parse ingredient names, amounts, and units from extracted text
- **Custom Training**: Train custom models for text detection in recipe contexts
- **Batch Processing**: Process multiple recipe images efficiently
- **Data Validation**: Validate and clean extracted ingredient data
- **Model Export**: Export trained models to various formats (ONNX, TensorRT, etc.)

## Project Structure

```
hello-fresh-recipe-ai/
├── data/                    # Dataset storage
│   ├── raw/                # Original recipe images
│   ├── processed/          # Preprocessed images
│   └── annotations/        # YOLO format text detection labels
├── models/                 # Model storage
│   ├── pretrained/         # Downloaded YOLOv8 models
│   └── custom/            # Trained custom models
├── results/               # Output results
│   ├── images/            # Processed images
│   ├── videos/            # Processed videos
│   └── metrics/           # Training metrics
├── src/                   # Source code
│   ├── text_detect.py     # Text detection and ingredient recognition
│   ├── detect.py          # Legacy object detection script
│   ├── train.py           # Training script for text detection
│   └── utils/             # Utility functions
│       ├── data_utils.py  # Data preprocessing utilities
│       └── text_utils.py  # Text processing and parsing utilities
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

### Text Detection and Ingredient Recognition

**Process Recipe Image:**
```bash
python src/text_detect.py recipe_image.jpg --output-image results/annotated_recipe.jpg --output-json results/ingredients.json
```

**Using Different OCR Engines:**
```bash
# Using EasyOCR (default)
python src/text_detect.py recipe.jpg --ocr-engine easyocr

# Using Tesseract
python src/text_detect.py recipe.jpg --ocr-engine tesseract

# Using PaddleOCR
python src/text_detect.py recipe.jpg --ocr-engine paddleocr
```

**Custom Text Detection Model:**
```bash
python src/text_detect.py recipe.jpg --model models/custom/text_model.pt --output-json ingredients.json
```

### Legacy Object Detection (for reference)

**Single Image:**
```bash
python src/detect.py path/to/image.jpg --output results/detected_image.jpg
```

### Custom Model Training

**Prepare Text Detection Dataset:**
1. Organize your dataset with recipe images and text annotations:
   ```
   data/
   ├── train/
   │   ├── images/        # Recipe images
   │   └── labels/        # YOLO format text region annotations
   ├── val/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

2. Create data configuration:
   ```bash
   python src/train.py --data data/ --create-config --classes ingredient_name amount unit text_line
   ```

**Train Text Detection Model:**
```bash
python src/train.py --data configs/text_data.yaml --epochs 100 --batch-size 16
```

**Train with Custom Parameters:**
```bash
python src/train.py --data configs/text_data.yaml --model-size s --epochs 200 --batch-size 32 --lr 0.001
```

**Validate Model:**
```bash
python src/train.py --data configs/text_data.yaml --validate-only --model-path models/custom/best.pt
```

**Export Model:**
```bash
python src/train.py --export onnx --model-path models/custom/best.pt
```

## Configuration

### Data Configuration (`configs/text_data.yaml`)
Defines dataset paths and class names for text detection training.

### Training Configuration (`configs/training_config.yaml`)
Contains default training parameters and hyperparameters.

## Development

### Text Processing and Data Preparation
```python
from src.utils.text_utils import IngredientParser, TextPreprocessor
from src.utils.data_utils import split_dataset, validate_dataset_structure

# Parse ingredient text
parser = IngredientParser()
ingredient_data = parser.parse_ingredient_line("2 cups flour")
print(ingredient_data)

# Preprocess images for better OCR
preprocessor = TextPreprocessor()
enhanced_image = preprocessor.enhance_contrast(image)

# Split dataset into train/val/test
split_dataset('raw_images/', 'raw_labels/', 'data/', (0.7, 0.2, 0.1))
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

1. **OCR not working**: Install Tesseract OCR system package
2. **CUDA not available**: Install PyTorch with CUDA support  
3. **Out of memory**: Reduce batch size or image size
4. **Poor text recognition**: Try different OCR engines or image preprocessing
5. **Dataset not found**: Check paths in data configuration file

### Getting Help

- Check the logs in the `logs/` directory
- Verify dataset structure with validation tools
- Ensure all dependencies are installed correctly

## License

This project is for educational and development purposes.