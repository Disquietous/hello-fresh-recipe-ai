# HelloFresh Recipe AI - Ingredient Extraction Pipeline

A complete computer vision pipeline for extracting structured ingredient data from recipe images using YOLOv8 text detection, multiple OCR engines, and intelligent parsing. Specializes in detecting and parsing text from handwritten, printed, and digital recipe formats.

## 🚀 Pipeline Overview

The system processes recipe images through a 5-stage pipeline:

1. **📋 Text Region Detection** - YOLOv8 identifies ingredient text blocks with format-aware preprocessing
2. **🔍 OCR Text Extraction** - Multiple engines (EasyOCR, PaddleOCR, Tesseract) with adaptive settings
3. **⚙️ Text Preprocessing** - Format-specific enhancement for handwritten, printed, and digital recipes
4. **🧠 Ingredient Parsing** - NLP-based parsing of quantities, units, and ingredient names
5. **📊 Structured Output** - Validated JSON with confidence scores and quality metrics

## ✨ Features

- **Multi-Format Support**: Handwritten, printed, and digital recipe formats with adaptive processing
- **Comprehensive Data Pipeline**: Complete dataset preparation tools with external dataset downloaders
- **Text-Specific Augmentation**: Specialized augmentations for text detection training
- **Multiple OCR Engines**: EasyOCR, PaddleOCR, Tesseract with format-optimized settings
- **Intelligent Parsing**: Extract quantities, units, and ingredient names with context awareness
- **Quality Validation**: Confidence scoring, annotation validation, and data quality assessment
- **Flexible Configuration**: JSON-based configuration system with format-specific parameters
- **Batch Processing**: Efficient processing of multiple images with parallel execution
- **Custom Training**: Complete training pipeline for recipe-specific text detection models
- **External Dataset Integration**: Automated downloading and conversion of ICDAR, TextOCR, and other text detection datasets

## Project Structure

```
hello-fresh-recipe-ai/
├── data/                           # Dataset storage and management
│   ├── recipe_cards/              # Recipe images organized by format
│   │   ├── handwritten/           # Handwritten recipe cards
│   │   ├── printed/              # Printed cookbooks and cards
│   │   └── digital/              # Digital recipe screenshots
│   ├── external_datasets/         # Downloaded public datasets
│   │   ├── icdar2015/            # ICDAR text detection datasets
│   │   ├── textocr/              # TextOCR dataset
│   │   └── synthtext/            # SynthText synthetic dataset
│   ├── processed/                 # Preprocessed and augmented data
│   │   ├── train/                # Training split
│   │   ├── val/                  # Validation split
│   │   └── test/                 # Test split
│   └── annotations/               # Original annotation files
├── models/                        # Model storage
│   ├── pretrained/               # Downloaded YOLOv8 models
│   └── custom/                   # Trained custom models
├── results/                      # Output results
│   ├── images/                   # Processed images
│   ├── datasets/                 # Dataset analysis results
│   └── metrics/                  # Training and validation metrics
├── src/                          # Source code
│   ├── ingredient_pipeline.py    # Main pipeline orchestrator
│   ├── text_detect.py           # Individual text detection script
│   ├── train.py                 # Training script for text detection
│   └── utils/                   # Utility functions
│       ├── text_utils.py        # Text processing and parsing
│       ├── data_utils.py        # Data preprocessing and validation
│       ├── annotation_utils.py   # Annotation format conversion
│       ├── dataset_downloader.py # External dataset downloaders
│       ├── text_augmentation.py  # Text-specific data augmentation
│       └── recipe_format_handler.py # Recipe format classification
├── scripts/                     # Command-line tools
│   └── prepare_dataset.py       # Dataset preparation and management
├── configs/                     # Configuration files
│   ├── pipeline_config.json     # Main pipeline configuration
│   ├── text_data.yaml          # Training data configuration
│   ├── training_config.yaml    # Training parameters
│   └── ingredients.json        # Ingredient database
├── examples/                    # Usage examples and sample scripts
├── notebooks/                   # Jupyter notebooks
└── logs/                       # Training and pipeline logs
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

### Data Preparation and Management

**Dataset Preparation Tool:**
The `scripts/prepare_dataset.py` script provides comprehensive dataset management:

```bash
# Download external text detection datasets
python scripts/prepare_dataset.py download --datasets icdar2015 textocr

# Prepare recipe dataset from raw images
python scripts/prepare_dataset.py prepare data/recipe_cards/ data/processed/

# Convert external dataset to YOLO format
python scripts/prepare_dataset.py convert data/external_datasets/icdar2015/ icdar data/converted/

# Validate dataset structure and quality
python scripts/prepare_dataset.py validate data/processed/ --type yolo

# Create annotation template for new images
python scripts/prepare_dataset.py template recipe_image.jpg annotation_template.json

# Merge multiple datasets
python scripts/prepare_dataset.py merge data/processed/ data/external_converted/ data/merged/
```

**Dataset Organization:**
1. Organize your recipe images by format:
   ```
   data/recipe_cards/
   ├── handwritten/         # Handwritten recipe cards and notes
   ├── printed/            # Printed cookbooks, magazines, cards
   └── digital/            # Screenshots from websites and apps
   ```

2. Annotation format for text detection:
   ```
   # YOLO format: class_id x_center y_center width height (normalized 0-1)
   0 0.5 0.3 0.4 0.05  # ingredient_line: single ingredient line
   1 0.5 0.4 0.6 0.15  # ingredient_block: multi-line ingredient section
   2 0.5 0.6 0.8 0.10  # instruction_text: cooking instructions
   3 0.5 0.1 0.6 0.08  # recipe_title: recipe title or heading
   4 0.8 0.9 0.3 0.04  # metadata_text: serving size, time, etc.
   ```

**Automatic Data Augmentation:**
The system applies format-specific augmentations:
- **Handwritten**: Ink variation, paper texture, rotation, perspective
- **Printed**: Print artifacts, scan lines, slight rotation
- **Digital**: Screen glare, compression artifacts, font rendering

### Custom Model Training

**Prepare Text Detection Dataset:**
```bash
# Use the dataset preparation tool to create training data
python scripts/prepare_dataset.py prepare data/recipe_cards/ data/processed/ --augmentations 5
```

**Train Text Detection Model:**
```bash
python src/train.py --data configs/text_data.yaml --epochs 100 --batch-size 16
```

**Advanced Training with Format-Specific Data:**
```bash
python src/train.py --data configs/text_data.yaml --model-size s --epochs 200 --batch-size 32 --lr 0.001 --format-specific
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
path: ./data/processed
train: train/images
val: val/images
test: test/images
nc: 5
names:
  0: ingredient_line    # Single ingredient lines
  1: ingredient_block   # Multi-line ingredient sections
  2: instruction_text   # Cooking instructions
  3: recipe_title      # Recipe titles and headings
  4: metadata_text     # Serving size, time, difficulty, etc.
```

### Recipe Format Configuration
The system automatically detects and processes different recipe formats:

```json
{
  "format_detection": {
    "handwritten": {
      "preprocessing": ["noise_reduction", "adaptive_threshold", "rotation_correction"],
      "ocr_settings": {"confidence_threshold": 0.3, "psm_mode": 6}
    },
    "printed": {
      "preprocessing": ["moderate_noise_reduction", "global_threshold"],
      "ocr_settings": {"confidence_threshold": 0.6, "psm_mode": 4}
    },
    "digital": {
      "preprocessing": ["minimal_processing", "ui_element_removal"],
      "ocr_settings": {"confidence_threshold": 0.7, "psm_mode": 3}
    }
  }
}
```

## 📊 Output Format

The pipeline produces structured JSON output:

```json
{
  "source_image": "recipe.jpg",
  "pipeline_version": "2.0",
  "format_analysis": {
    "detected_format": "handwritten",
    "confidence": 0.87,
    "characteristics": {
      "ink_type": "pen",
      "has_lined_paper": true,
      "text_slant": 2.3
    }
  },
  "detection_summary": {
    "total_regions_detected": 8,
    "ingredient_lines": 5,
    "ingredient_blocks": 1,
    "instruction_text": 2,
    "high_confidence_ingredients": 4,
    "medium_confidence_ingredients": 2,
    "low_confidence_ingredients": 0
  },
  "ingredients": [
    {
      "ingredient_name": "All-Purpose Flour",
      "quantity": "2",
      "unit": "cups",
      "unit_category": "volume",
      "normalized_ingredient": "flour_all_purpose",
      "confidence_scores": {
        "overall": 0.85,
        "ingredient_recognition": 0.9,
        "text_detection": 0.8,
        "ocr_quality": 0.85,
        "parsing_confidence": 0.88
      },
      "raw_text": "2 cups all-purpose flour",
      "processed_text": "2 cups all-purpose flour",
      "bounding_box": {
        "x1": 100, "y1": 150, "x2": 300, "y2": 180
      },
      "text_region_type": "ingredient_line"
    }
  ],
  "other_text_regions": [
    {
      "type": "recipe_title",
      "text": "Chocolate Chip Cookies",
      "bounding_box": {"x1": 50, "y1": 20, "x2": 400, "y2": 60},
      "confidence": 0.92
    }
  ],
  "validation_results": {
    "validation_score": 0.85,
    "overall_quality": "good",
    "format_specific_quality": "excellent_for_handwritten"
  },
  "processing_metadata": {
    "ocr_engine_used": "easyocr",
    "preprocessing_applied": ["noise_reduction", "adaptive_threshold", "rotation_correction"],
    "processing_time_ms": 3420
  }
}
```

## 🛠️ Development

### Examples and Testing
```bash
# Run comprehensive examples
python examples/basic_usage.py

# Test dataset preparation tools
python scripts/prepare_dataset.py validate data/processed/ --type yolo

# Test format classification
python -c "from src.utils.recipe_format_handler import RecipeFormatClassifier; import cv2; classifier = RecipeFormatClassifier(); image = cv2.imread('recipe.jpg'); metadata = classifier.classify_recipe_format(image); print(f'Format: {metadata.format_type.value}, Confidence: {metadata.confidence:.2f}')"

# Test text augmentation
python -c "from src.utils.text_augmentation import TextAugmentationPipeline; pipeline = TextAugmentationPipeline(); print('Augmentation pipeline initialized successfully')"

# Test individual components
python -c "from src.utils.text_utils import IngredientParser; parser = IngredientParser(); print(parser.parse_ingredient_line('2 cups flour'))"
```

### Jupyter Notebooks
Use the `notebooks/` directory for experimentation and analysis:
```bash
jupyter notebook notebooks/
```

### Dataset Management Examples
```bash
# Download and prepare external datasets
python scripts/prepare_dataset.py download --data-dir data/external_datasets/

# Create training dataset from your recipe cards
python scripts/prepare_dataset.py prepare data/recipe_cards/ data/training_ready/ --augmentations 3

# Analyze dataset quality
python scripts/prepare_dataset.py validate data/training_ready/ --type yolo

# Merge multiple datasets for comprehensive training
python scripts/prepare_dataset.py merge data/training_ready/ data/external_datasets/icdar2015/yolo_format/ data/final_training/
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