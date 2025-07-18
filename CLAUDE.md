# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HelloFresh Recipe AI is a YOLOv8-based computer vision project for text detection and ingredient recognition from recipe images. The project uses Python with the Ultralytics YOLOv8 framework for text detection, multiple OCR engines for text extraction, and NLP techniques for parsing ingredient names, amounts, and units.

## Technology Stack

- **Python 3.8+** - Main programming language
- **YOLOv8 (Ultralytics)** - Text detection framework
- **PyTorch** - Deep learning backend
- **EasyOCR, Tesseract, PaddleOCR** - Text recognition engines
- **OpenCV** - Image processing
- **NLTK** - Natural language processing
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
# Main ingredient extraction pipeline
python src/ingredient_pipeline.py recipe.jpg --output-dir results/

# With custom configuration
python src/ingredient_pipeline.py recipe.jpg --config configs/pipeline_config.json

# Different OCR engines
python src/ingredient_pipeline.py recipe.jpg --ocr-engine paddleocr --confidence 0.3

# Individual text detection
python src/text_detect.py recipe.jpg --output-image results/annotated.jpg

# Train custom text detection model
python src/train.py --data configs/text_data.yaml --epochs 100

# Export trained model
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
- `src/ingredient_pipeline.py` - Main pipeline orchestrator with IngredientExtractionPipeline class
- `src/text_detect.py` - Individual text detection script with IngredientTextDetector class
- `src/train.py` - Training script with TextModelTrainer class for custom models
- `src/utils/text_utils.py` - Text processing, parsing, and validation (IngredientParser, TextPreprocessor)
- `src/utils/data_utils.py` - Dataset utilities for preprocessing and validation
- `configs/pipeline_config.json` - Main pipeline configuration
- `configs/ingredients.json` - Ingredient database and units

### Data Flow
1. **Raw data** → `data/raw/` (original recipe images)
2. **Text Detection** → YOLOv8 model detects text regions
3. **OCR Processing** → Multiple OCR engines extract text from regions
4. **Text Parsing** → NLP techniques parse ingredients, amounts, units
5. **Validation** → Data quality checks and confidence scoring
6. **Output** → `results/` (annotated images and JSON ingredient data)

### Configuration System
- `configs/text_data.yaml` - Dataset configuration for text detection classes
- `configs/training_config.yaml` - Training hyperparameters and augmentation settings

## Important Patterns

### Dataset Structure
YOLO format for text detection with train/val/test splits:
```
data/
├── train/images/ + train/labels/  # Recipe images with text region annotations
├── val/images/ + val/labels/      # Validation set
└── test/images/ + test/labels/    # Test set
```

### Text Detection Classes
- `ingredient_name` - Text regions containing ingredient names
- `amount` - Numerical amounts (1, 2.5, 1/2, etc.)
- `unit` - Measurement units (cups, tbsp, oz, etc.)
- `text_line` - General text lines in recipes

### Model Management
- Pretrained models auto-download to `models/pretrained/`
- Custom trained models save to `models/custom/`
- Best model saved as `best.pt`, last checkpoint as `last.pt`

### Error Handling
- Always validate dataset structure before training
- Check CUDA availability for GPU training
- Verify model paths exist before validation/export

## Development Notes

- Use `IngredientExtractionPipeline` class for complete end-to-end processing
- Use `IngredientTextDetector` class for individual text detection tasks
- Use `TextModelTrainer` class for training custom text detection models  
- Use `IngredientParser` class for parsing ingredient text
- Pipeline supports multiple OCR engines with automatic fallback
- Configuration-driven approach - modify `configs/pipeline_config.json` for different settings
- All paths should be absolute or relative to project root
- Training saves to `models/custom/` directory
- Results include confidence scores and validation metrics
- Text preprocessing significantly improves OCR accuracy
- Examples in `examples/` directory show comprehensive usage patterns