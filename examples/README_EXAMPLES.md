# Recipe Processing Examples

This directory contains comprehensive examples demonstrating how to use the Recipe Processing API for different types of recipe images and use cases.

## 📁 Example Files

### 🚀 [quick_start_examples.py](quick_start_examples.py)
**Perfect for beginners** - Simple, ready-to-run examples with minimal setup.

**What's included:**
- Basic image processing example
- Cookbook page processing
- Handwritten recipe processing
- Digital screenshot processing
- Error handling basics
- API usage examples
- Performance tips
- Troubleshooting guide

**Run it:**
```bash
python examples/quick_start_examples.py
```

### 🔧 [comprehensive_usage_examples.py](comprehensive_usage_examples.py)
**Advanced examples** - Detailed demonstrations of all features and capabilities.

**What's included:**
- 8 comprehensive examples covering all major use cases
- Batch processing with async operations
- Performance optimization techniques
- Advanced error handling and validation
- Memory-efficient processing
- Intelligent caching strategies
- API integration patterns
- Result formatting and export

**Run it:**
```bash
python examples/comprehensive_usage_examples.py
```

### 📊 [recipe_type_examples.py](recipe_type_examples.py)
**Type-specific processing** - Optimized settings and examples for different recipe image types.

**What's included:**
- Cookbook page processing (clean printed text)
- Handwritten recipe card processing
- Digital screenshot processing (apps/websites)
- Recipe blog image processing
- Foreign language recipe processing (Spanish, French, German, Italian)
- UI element filtering
- Decorative text filtering
- Language-specific optimizations

**Run it:**
```bash
python examples/recipe_type_examples.py
```

## 🎯 Quick Reference by Use Case

### I want to process a single cookbook page
```python
from src.ingredient_pipeline import IngredientExtractionPipeline

pipeline = IngredientExtractionPipeline()
result = pipeline.process_image(
    'cookbook_page.jpg',
    format_hint='cookbook',
    ocr_engine='tesseract',
    confidence_threshold=0.3
)
```

### I want to process handwritten recipes
```python
result = pipeline.process_image(
    'handwritten_recipe.jpg',
    format_hint='handwritten',
    ocr_engine='easyocr',
    confidence_threshold=0.2,
    enable_preprocessing=True
)
```

### I want to process app screenshots
```python
result = pipeline.process_image(
    'app_screenshot.png',
    format_hint='digital',
    ocr_engine='paddleocr',
    confidence_threshold=0.4
)
```

### I want to process multiple images
```python
image_paths = ['recipe1.jpg', 'recipe2.jpg', 'recipe3.jpg']
results = []

for image_path in image_paths:
    try:
        result = pipeline.process_image(image_path)
        results.append(result)
        print(f'✅ {image_path}: {len(result["ingredients"])} ingredients')
    except Exception as e:
        print(f'❌ {image_path}: Error - {e}')
```

### I want to use the REST API
```python
import requests

with open('recipe.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process',
        files={'file': f},
        data={'format_hint': 'cookbook', 'confidence_threshold': 0.3}
    )

if response.status_code == 200:
    result = response.json()
    print(f"Found {len(result['ingredients'])} ingredients")
```

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
# Install core requirements
pip install -r requirements.txt

# Install performance testing dependencies (optional)
pip install -r requirements-performance.txt
```

### 2. Download Models
```bash
# Download YOLOv8 model (required for text detection)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3. Prepare Test Images
The examples will create sample images automatically, but you can also use your own:

```
examples/
├── sample_images/           # Auto-generated test images
│   ├── cookbook_sample.jpg
│   └── handwritten_sample.jpg
├── your_images/            # Your own recipe images
│   ├── recipe1.jpg
│   ├── recipe2.png
│   └── handwritten_card.jpg
└── results/                # Processing results
    ├── example_1_cookbook.json
    └── batch_summary_*.json
```

## 📝 Expected Results Format

All processing functions return results in this format:

```json
{
  "filename": "recipe.jpg",
  "format_type": "printed_cookbook",
  "quality_score": 0.85,
  "confidence_score": 0.78,
  "processing_time": 8.5,
  "language": "en",
  "ingredients": [
    {
      "ingredient_name": "flour",
      "quantity": "2",
      "unit": "cups",
      "preparation": "sifted",
      "confidence": 0.92,
      "bbox": {"x1": 120, "y1": 45, "x2": 280, "y2": 70}
    }
  ]
}
```

## 🎨 Recipe Image Types & Settings

| Recipe Type | Best OCR Engine | Confidence Threshold | Notes |
|-------------|----------------|---------------------|--------|
| **Cookbook Pages** | `tesseract` | `0.3` | Clean printed text, high accuracy expected |
| **Handwritten** | `easyocr` | `0.2` | Lower threshold needed, enable preprocessing |
| **Digital Screenshots** | `paddleocr` | `0.4` | Modern fonts, watch for UI elements |
| **Recipe Blogs** | `paddleocr` | `0.35` | Mixed layouts, filter decorative text |
| **Foreign Languages** | `easyocr` | `0.25` | Language-specific optimizations |

## 🚨 Common Issues & Solutions

### No ingredients found
- **Solution**: Lower confidence threshold to 0.1
- **Try**: Different OCR engine (easyocr → paddleocr → tesseract)
- **Enable**: Image preprocessing
- **Check**: Image contains readable text

### Low confidence scores
- **Improve**: Image quality (lighting, focus, resolution)
- **Try**: Different OCR engine
- **Enable**: Preprocessing for poor quality images
- **Consider**: Manual review for critical recipes

### Processing is slow
- **Resize**: Large images to 1500-2500px max dimension
- **Use**: Faster OCR engine (tesseract for clean text)
- **Enable**: Caching for repeated processing
- **Process**: In smaller batches

### Wrong ingredients detected
- **Increase**: Confidence threshold
- **Use**: Format hints (`cookbook`, `handwritten`, etc.)
- **Crop**: Image to ingredients section only
- **Implement**: Post-processing filters

## 🔧 Performance Tips

### Image Optimization
- Resize to 1500-2500px maximum dimension
- Use good lighting and avoid blurry images
- Crop to ingredients section when possible
- Convert to JPEG for smaller file sizes

### OCR Engine Selection
- **Tesseract**: Best for clean printed text (cookbooks)
- **EasyOCR**: Best for handwritten text and multiple languages
- **PaddleOCR**: Best for digital text and mixed layouts

### Processing Optimization
- Use format hints when you know the image type
- Enable caching for repeated processing
- Process images in batches for better performance
- Use appropriate confidence thresholds per image type

## 📚 Additional Resources

- **[Usage Guide](../docs/USAGE_GUIDE.md)** - Comprehensive usage documentation
- **[API Documentation](../docs/API_DOCUMENTATION.md)** - Complete API reference
- **[Performance Guide](../docs/PERFORMANCE_OPTIMIZATION.md)** - Performance tuning guide
- **[Configuration Reference](../configs/performance_config.yaml)** - Configuration options

## 🆘 Getting Help

1. **Check the examples** - Most common use cases are covered
2. **Read the troubleshooting sections** - Common issues and solutions
3. **Check the logs** - Enable debug logging for detailed information
4. **Test with different settings** - Try different OCR engines and thresholds
5. **Validate your images** - Ensure images are readable and well-lit

## 🚀 Next Steps

1. **Start with quick_start_examples.py** for basic usage
2. **Try recipe_type_examples.py** for your specific image types
3. **Explore comprehensive_usage_examples.py** for advanced features
4. **Read the full documentation** for complete feature coverage
5. **Optimize settings** based on your specific use case

Happy recipe processing! 🍳✨