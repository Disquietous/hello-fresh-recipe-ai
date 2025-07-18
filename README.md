# HelloFresh Recipe AI - Ingredient Extraction Pipeline

A complete computer vision pipeline for extracting structured ingredient data from recipe images using YOLOv8 text detection, multiple OCR engines, and intelligent parsing.

## 🚀 Pipeline Overview

The system processes recipe images through a 5-stage pipeline:

1. **📋 Text Region Detection** - YOLOv8 identifies ingredient text blocks
2. **🔍 OCR Text Extraction** - Multiple engines extract text from regions  
3. **⚙️ Text Preprocessing** - Enhanced image processing for better OCR
4. **🧠 Ingredient Parsing** - NLP-based parsing of quantities, units, and names
5. **📊 Structured Output** - Validated JSON with confidence scores

## ✨ Features

- **Multi-Stage Pipeline**: Complete workflow from image → structured data
- **Multiple OCR Engines**: EasyOCR, PaddleOCR, Tesseract with fallback support
- **Intelligent Parsing**: Extract quantities, units, and ingredient names
- **Quality Validation**: Confidence scoring and data quality assessment
- **Flexible Configuration**: JSON-based configuration system
- **Batch Processing**: Process multiple images efficiently
- **Custom Training**: Train models for specific recipe formats

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
├── src/                     # Source code
│   ├── ingredient_pipeline.py  # Main pipeline orchestrator
│   ├── text_detect.py          # Individual text detection script
│   ├── train.py                # Training script for text detection
│   └── utils/                  # Utility functions
│       ├── text_utils.py       # Text processing and parsing
│       └── data_utils.py       # Data preprocessing utilities
├── configs/                 # Configuration files
│   ├── pipeline_config.json    # Main pipeline configuration
│   ├── text_data.yaml          # Training data configuration
│   ├── training_config.yaml    # Training parameters
│   └── ingredients.json        # Ingredient database
├── examples/               # Usage examples and sample scripts
├── notebooks/             # Jupyter notebooks
└── logs/                  # Training and pipeline logs
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

## 📋 Usage

### Main Ingredient Extraction Pipeline

**Basic Usage:**
```bash
# Process a single recipe image
python src/ingredient_pipeline.py recipe_image.jpg --output-dir results/

# With custom configuration
python src/ingredient_pipeline.py recipe.jpg --config configs/pipeline_config.json --output-dir results/

# Using different OCR engine
python src/ingredient_pipeline.py recipe.jpg --ocr-engine paddleocr --confidence 0.3
```

**Batch Processing:**
```bash
# Process multiple images
for image in *.jpg; do
    python src/ingredient_pipeline.py "$image" --output-dir "results/$(basename "$image" .jpg)"
done
```

**Python API Usage:**
```python
from src.ingredient_pipeline import IngredientExtractionPipeline

# Initialize pipeline
pipeline = IngredientExtractionPipeline()

# Process image
results = pipeline.process_image('recipe.jpg', 'results/')

# Access structured data
for ingredient in results['ingredients']:
    print(f"{ingredient['quantity']} {ingredient['unit']} {ingredient['ingredient_name']}")
```

### Individual Components

**Text Detection Only:**
```bash
python src/text_detect.py recipe.jpg --output-image results/annotated.jpg --output-json results/raw_text.json
```

**Ingredient Parsing Only:**
```python
from src.utils.text_utils import IngredientParser

parser = IngredientParser()
result = parser.parse_ingredient_line("2 cups all-purpose flour")
print(result)  # {'ingredient_name': 'Flour', 'amount': '2', 'unit': 'cups', ...}
```

### Custom Model Training

**Prepare Text Detection Dataset:**
1. Organize your dataset with recipe images and text region annotations:
   ```
   data/
   ├── train/
   │   ├── images/          # Recipe images
   │   └── labels/          # YOLO format annotations for ingredient text blocks
   ├── val/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

2. Annotation format for ingredient text blocks:
   ```
   # Each line in label file: class_id x_center y_center width height
   0 0.5 0.3 0.4 0.05  # ingredient_line: "2 cups flour"
   1 0.5 0.4 0.6 0.15  # ingredient_block: multi-line ingredient section
   ```

**Train Text Detection Model:**
```bash
python src/train.py --data configs/text_data.yaml --epochs 100 --batch-size 16
```

**Advanced Training:**
```bash
python src/train.py --data configs/text_data.yaml --model-size s --epochs 200 --batch-size 32 --lr 0.001
```

**Export Model:**
```bash
python src/train.py --export onnx --model-path models/custom/best.pt
```

## ⚙️ Configuration

### Pipeline Configuration (`configs/pipeline_config.json`)
Main configuration file controlling all pipeline stages:

```json
{
  "text_detection": {
    "model_path": "yolov8n.pt",
    "confidence_threshold": 0.25
  },
  "ocr": {
    "engine": "easyocr",
    "fallback_engines": ["paddleocr", "tesseract"]
  },
  "parsing": {
    "min_confidence": 0.5,
    "normalize_ingredients": true
  },
  "output": {
    "save_annotated_image": true,
    "output_format": "json"
  }
}
```

### Training Configuration (`configs/text_data.yaml`)
Dataset configuration for training text detection models:

```yaml
path: ./data
train: train/images
val: val/images
nc: 2
names:
  0: ingredient_line
  1: ingredient_block
```

## 📊 Output Format

The pipeline produces structured JSON output:

```json
{
  "source_image": "recipe.jpg",
  "pipeline_version": "1.0",
  "detection_summary": {
    "total_regions_detected": 5,
    "high_confidence_ingredients": 3,
    "medium_confidence_ingredients": 2,
    "low_confidence_ingredients": 0
  },
  "ingredients": [
    {
      "ingredient_name": "All-Purpose Flour",
      "quantity": "2",
      "unit": "cups",
      "unit_category": "volume",
      "confidence_scores": {
        "overall": 0.85,
        "ingredient_recognition": 0.9,
        "text_detection": 0.8,
        "ocr_quality": 0.85
      },
      "raw_text": "2 cups all-purpose flour",
      "bounding_box": {
        "x1": 100, "y1": 150, "x2": 300, "y2": 180
      }
    }
  ],
  "validation_results": {
    "validation_score": 0.85,
    "overall_quality": "good"
  }
}
```

## 🛠️ Development

### Examples and Testing
```bash
# Run comprehensive examples
python examples/basic_usage.py

# Test individual components
python -c "from src.utils.text_utils import IngredientParser; parser = IngredientParser(); print(parser.parse_ingredient_line('2 cups flour'))"
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

1. **No ingredients detected**
   - Check image quality and text visibility
   - Adjust confidence thresholds
   - Try different OCR engines

2. **Poor ingredient parsing**
   - Verify text is in English
   - Check ingredient database coverage
   - Enable text preprocessing

3. **OCR errors**
   - Install Tesseract system package: `apt install tesseract-ocr` (Linux) or `brew install tesseract` (macOS)
   - For Windows: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Performance issues**
   - Use GPU acceleration when available
   - Reduce image size for faster processing
   - Use smaller YOLO models (yolov8n vs yolov8s)

### Getting Help

- Check the logs in the `logs/` directory
- Verify dataset structure with validation tools
- Ensure all dependencies are installed correctly

## License

This project is for educational and development purposes.