# Text Detection Training System

A comprehensive training system for YOLOv8 text detection optimized for recipe images. This system includes annotation tools, training scripts, evaluation metrics, validation pipelines, and multi-language support.

## 🚀 Features

- **YOLOv8 Text Detection**: Specialized for recipe text regions (ingredients, instructions, titles)
- **Interactive Annotation Tool**: Easy-to-use GUI for creating text bounding boxes
- **Text-Specific Training**: Optimized augmentations and hyperparameters for text detection
- **Comprehensive Evaluation**: Precision, recall, F1-score, and mAP metrics for text regions
- **End-to-End Validation**: Tests both detection accuracy and OCR quality
- **Multi-Language Support**: Process recipes in 11+ languages
- **Batch Processing**: Efficient handling of large datasets
- **Visualization Tools**: Training plots, evaluation charts, and error analysis

## 📋 System Components

### 1. **Text Detection Classes**
- `ingredient_line` - Single ingredient lines (e.g., "2 cups flour")
- `ingredient_block` - Multi-line ingredient sections
- `instruction_text` - Cooking instructions and method steps
- `recipe_title` - Recipe names and headings
- `metadata_text` - Serving size, time, difficulty, etc.

### 2. **Core Modules**
- `train_text_detection.py` - Main training script
- `text_detection_evaluator.py` - Comprehensive evaluation metrics
- `validation_pipeline.py` - End-to-end validation pipeline
- `annotation_tool.py` - Interactive annotation interface
- `multilingual_support.py` - Multi-language processing

### 3. **Configuration**
- `configs/text_detection_training.yaml` - Training configuration
- Text-specific augmentations (reduced rotation, no flipping)
- Optimized hyperparameters for text detection

## 🛠️ Installation

### 1. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Additional dependencies for training
pip install ultralytics opencv-python torch torchvision
pip install matplotlib seaborn pandas numpy

# For multilingual support
pip install langdetect easyocr

# For annotation tool
pip install tkinter
```

### 2. System Dependencies

**Tesseract OCR (optional, for validation):**
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## 📊 Quick Start

### 1. Create Annotations

```bash
# Annotate single image
python src/annotation_tool.py --image recipe.jpg --output-dir annotations/

# Batch annotation mode
python src/annotation_tool.py --image-dir recipe_images/ --batch --output-dir annotations/

# Interactive file selection (no arguments)
python src/annotation_tool.py
```

**Annotation Controls:**
- Left click + drag: Draw bounding box
- Right click: Delete box at cursor
- 1-5: Change annotation class
- S: Save annotations
- R: Reset view
- Q: Quit

### 2. Train Text Detection Model

```bash
# Train with default configuration
python src/train_text_detection.py --config configs/text_detection_training.yaml

# Train with custom name
python src/train_text_detection.py --config configs/text_detection_training.yaml --name recipe_text_v2

# Resume training
python src/train_text_detection.py --config configs/text_detection_training.yaml --resume

# Export model after training
python src/train_text_detection.py --config configs/text_detection_training.yaml --export
```

### 3. Evaluate Model

```bash
# Evaluate trained model
python src/text_detection_evaluator.py --model runs/text_detection/recipe_text_v1/weights/best.pt --dataset data/processed

# Enable OCR evaluation
python src/text_detection_evaluator.py --model best.pt --dataset data/processed --enable-ocr

# Custom configuration
python src/text_detection_evaluator.py --model best.pt --dataset data/processed --confidence 0.3 --iou 0.5
```

### 4. Validate Full Pipeline

```bash
# Comprehensive validation
python src/validation_pipeline.py --model best.pt --dataset data/processed

# Parallel processing
python src/validation_pipeline.py --model best.pt --dataset data/processed --parallel --workers 8

# Custom configuration
python src/validation_pipeline.py --model best.pt --dataset data/processed --config validation_config.json
```

### 5. Multi-Language Processing

```bash
# Process multilingual recipe
python src/multilingual_support.py --image recipe_spanish.jpg --language es

# Show supported languages
python src/multilingual_support.py --supported-languages

# Save results
python src/multilingual_support.py --image recipe.jpg --output results.json
```

## 📁 Dataset Structure

```
data/
├── processed/
│   ├── train/
│   │   ├── images/
│   │   │   ├── recipe_001.jpg
│   │   │   └── recipe_002.jpg
│   │   └── labels/
│   │       ├── recipe_001.txt
│   │       └── recipe_002.txt
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   ├── ground_truth_text.json      # Optional: OCR ground truth
│   └── expected_ingredients.json   # Optional: Ingredient ground truth
```

**YOLO Label Format:**
```
# class_id x_center y_center width height (normalized 0-1)
0 0.5 0.3 0.4 0.1    # ingredient_line
1 0.5 0.7 0.6 0.2    # ingredient_block
```

## ⚙️ Configuration

### Training Configuration

```yaml
# configs/text_detection_training.yaml
path: ./data/processed
nc: 5  # Number of classes
names:
  0: ingredient_line
  1: ingredient_block
  2: instruction_text
  3: recipe_title
  4: metadata_text

# Training parameters
batch: 16
epochs: 200
imgsz: 640
model: yolov8n.pt

# Text-specific augmentations
degrees: 5.0      # Reduced rotation
flipud: 0.0       # No vertical flip
fliplr: 0.0       # No horizontal flip
perspective: 0.0001  # Minimal perspective

# Optimization
optimizer: AdamW
lr0: 0.001
patience: 50
```

### Evaluation Configuration

```json
{
  "confidence_threshold": 0.25,
  "iou_thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
  "enable_ocr_evaluation": true,
  "class_names": {
    "0": "ingredient_line",
    "1": "ingredient_block", 
    "2": "instruction_text",
    "3": "recipe_title",
    "4": "metadata_text"
  }
}
```

## 📈 Evaluation Metrics

### Detection Metrics
- **Precision**: Accuracy of detected text regions
- **Recall**: Coverage of actual text regions
- **F1-Score**: Harmonic mean of precision and recall
- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision at IoU=0.5:0.95

### OCR Quality Metrics
- **Character Accuracy**: Percentage of correctly recognized characters
- **Word Accuracy**: Percentage of correctly recognized words
- **Sequence Accuracy**: Percentage of perfectly recognized text sequences

### End-to-End Metrics
- **Pipeline Accuracy**: Overall success rate of the complete pipeline
- **Ingredient Extraction Rate**: Percentage of images with extracted ingredients
- **Processing Time**: Average time per image

## 🔧 Advanced Usage

### Custom Training

```python
from src.train_text_detection import TextDetectionTrainer

# Initialize trainer
trainer = TextDetectionTrainer('configs/text_detection_training.yaml')

# Validate dataset
trainer.validate_dataset()

# Train model
results = trainer.train()

# Validate model
validation_results = trainer.validate_model()

# Export model
export_paths = trainer.export_model()
```

### Custom Evaluation

```python
from src.text_detection_evaluator import TextDetectionEvaluator

# Initialize evaluator
evaluator = TextDetectionEvaluator(config)

# Evaluate model
result = evaluator.evaluate_model(
    model_path='best.pt',
    test_dataset_path='data/processed',
    output_dir='evaluation_results'
)

# Access metrics
print(f"Precision: {result.overall_metrics.precision:.3f}")
print(f"Recall: {result.overall_metrics.recall:.3f}")
print(f"F1-Score: {result.overall_metrics.f1_score:.3f}")
```

### Multi-Language Processing

```python
from src.multilingual_support import MultilingualRecipeProcessor

# Initialize processor
processor = MultilingualRecipeProcessor()

# Process multilingual recipe
result = processor.process_multilingual_recipe(image, language_hint='es')

# Get ingredients with translations
for ingredient in result['multilingual_ingredients']:
    print(f"{ingredient['ingredient_name']} ({ingredient['language']})")
    print(f"Translations: {ingredient['translations']}")
```

## 🎯 Performance Optimization

### Training Optimization

```yaml
# For faster training
batch: 32          # Increase batch size
workers: 8         # More data loading workers
cache: true        # Cache images in memory
amp: true          # Automatic Mixed Precision

# For better accuracy
epochs: 300        # More training epochs
patience: 100      # More patience for early stopping
mosaic: 0.9        # More mosaic augmentation
```

### Inference Optimization

```python
# Export optimized model
trainer.export_model(formats=['onnx', 'engine'])

# Use optimized model
model = YOLO('model.onnx')  # ONNX for CPU
model = YOLO('model.engine')  # TensorRT for GPU
```

### Memory Optimization

```yaml
# Reduce memory usage
batch: 8           # Smaller batch size
imgsz: 512         # Smaller image size
workers: 4         # Fewer workers
cache: false       # Don't cache images
```

## 📊 Results Analysis

### Training Results
- Training/validation loss curves
- Precision-recall curves
- Class-wise performance metrics
- Confusion matrices

### Evaluation Results
- Overall detection metrics
- Class-wise performance
- Error analysis and failure cases
- Processing time analysis

### Validation Results
- End-to-end pipeline accuracy
- OCR quality assessment
- Ingredient extraction success rates
- Performance bottleneck analysis

## 🌍 Multi-Language Support

### Supported Languages
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Dutch (nl)
- Russian (ru)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)

### Language-Specific Features
- Automatic language detection
- Multilingual OCR processing
- Language-specific text cleaning
- Ingredient name translations
- Unit conversion between languages

## 🐛 Troubleshooting

### Common Issues

**Low Detection Accuracy:**
- Check annotation quality and consistency
- Increase training epochs
- Adjust confidence threshold
- Review class balance in dataset

**Poor OCR Performance:**
- Verify image quality and resolution
- Check text region accuracy
- Enable multilingual OCR
- Adjust OCR confidence thresholds

**Training Errors:**
- Verify dataset structure
- Check YOLO label format
- Ensure sufficient training data
- Review configuration parameters

### Debug Mode

```bash
# Enable debug logging
python src/train_text_detection.py --config configs/text_detection_training.yaml --verbose

# Validate dataset structure
python src/utils/data_utils.py --dataset data/processed --validate

# Check annotations
python src/annotation_tool.py --image recipe.jpg --load-existing
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Add comprehensive tests
4. Update documentation
5. Commit changes (`git commit -am 'Add new feature'`)
6. Push to branch (`git push origin feature/new-feature`)
7. Create Pull Request

## 📚 Example Workflows

### Complete Training Pipeline

```bash
# 1. Annotate training data
python src/annotation_tool.py --image-dir raw_images/ --batch --output-dir annotations/

# 2. Organize dataset
python src/utils/data_utils.py --organize --input annotations/ --output data/processed/

# 3. Train model
python src/train_text_detection.py --config configs/text_detection_training.yaml --name recipe_text_v1

# 4. Evaluate model
python src/text_detection_evaluator.py --model runs/text_detection/recipe_text_v1/weights/best.pt --dataset data/processed

# 5. Validate pipeline
python src/validation_pipeline.py --model runs/text_detection/recipe_text_v1/weights/best.pt --dataset data/processed

# 6. Export for production
python src/train_text_detection.py --config configs/text_detection_training.yaml --export --model-path best.pt
```

### Hyperparameter Tuning

```bash
# Experiment with different configurations
python src/train_text_detection.py --config configs/text_detection_training.yaml --name exp_lr001
python src/train_text_detection.py --config configs/text_detection_training_lr0001.yaml --name exp_lr0001
python src/train_text_detection.py --config configs/text_detection_training_batch32.yaml --name exp_batch32

# Compare results
python src/utils/compare_experiments.py --experiments exp_lr001 exp_lr0001 exp_batch32
```

## 📄 License

This project is for educational and development purposes.

## 🔗 Related Documentation

- [OCR Pipeline Documentation](README_OCR_Pipeline.md)
- [Main Project README](README.md)
- [Requirements](requirements.txt)
- [Training Configuration](configs/text_detection_training.yaml)