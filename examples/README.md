# Examples - HelloFresh Recipe AI

This directory contains examples and sample scripts demonstrating how to use the HelloFresh Recipe AI ingredient extraction pipeline.

## Files

### `basic_usage.py`
Comprehensive examples showing different ways to use the pipeline:

1. **Basic Usage** - Simple pipeline with default configuration
2. **Custom Configuration** - Using custom settings for better accuracy
3. **Configuration File** - Loading settings from JSON config file
4. **Batch Processing** - Processing multiple images at once
5. **Parsing Only** - Using just the ingredient parser component
6. **Validation** - Recipe data validation and quality assessment

### Running Examples

```bash
# Run all examples
python examples/basic_usage.py

# Or run specific functions by importing them
python -c "from examples.basic_usage import example_1_basic_usage; example_1_basic_usage()"
```

## Sample Recipe Images

To test the examples, you'll need sample recipe images. Place them in:

- `examples/sample_recipe.jpg` - Single recipe image
- `examples/recipe1.jpg`, `examples/recipe2.jpg` - Multiple recipe images  
- `examples/recipe_images/` - Directory for batch processing
- `examples/complex_recipe.jpg` - Complex recipe with multiple ingredients

### Image Requirements

For best results, use images that:
- Show ingredient lists clearly
- Have good contrast between text and background
- Are not overly blurry or distorted
- Contain English text
- Have ingredients formatted as lists (e.g., "2 cups flour")

## Configuration Examples

### Basic Configuration
```python
config = {
    'text_detection': {
        'confidence_threshold': 0.25
    },
    'ocr': {
        'engine': 'easyocr'
    }
}
```

### Advanced Configuration
```python
config = {
    'text_detection': {
        'model_path': 'yolov8s.pt',
        'confidence_threshold': 0.3,
        'iou_threshold': 0.45
    },
    'ocr': {
        'engine': 'paddleocr',
        'gpu': True,
        'fallback_engines': ['easyocr', 'tesseract']
    },
    'preprocessing': {
        'enhance_contrast': True,
        'remove_noise': True,
        'correct_skew': True
    },
    'parsing': {
        'min_confidence': 0.6,
        'fuzzy_match_threshold': 0.8,
        'normalize_ingredients': True
    },
    'output': {
        'save_annotated_image': True,
        'save_cropped_regions': True,
        'output_format': 'json'
    }
}
```

## Expected Output

### JSON Structure
```json
{
  "source_image": "path/to/image.jpg",
  "detection_summary": {
    "total_regions_detected": 5,
    "high_confidence_ingredients": 3,
    "medium_confidence_ingredients": 2,
    "low_confidence_ingredients": 0
  },
  "ingredients": [
    {
      "ingredient_name": "Flour",
      "quantity": "2",
      "unit": "cups",
      "unit_category": "volume",
      "confidence_scores": {
        "overall": 0.85,
        "ingredient_recognition": 0.9,
        "text_detection": 0.8,
        "ocr_quality": 0.85
      },
      "bounding_box": {
        "x1": 100,
        "y1": 150,
        "x2": 300,
        "y2": 180
      }
    }
  ]
}
```

### Console Output
```
✅ Processed: recipe.jpg
📊 Found 5 ingredients:
   - 2 cups Flour (confidence: 0.85)
   - 1 tsp Salt (confidence: 0.78)
   - 3 tbsp Olive Oil (confidence: 0.82)
   - 1 lb Ground Beef (confidence: 0.90)
   - 2 cloves Garlic (confidence: 0.75)
📁 Results saved to: results/example1/
```

## Troubleshooting

### Common Issues

1. **No ingredients detected**
   - Check image quality and contrast
   - Try different OCR engines
   - Adjust confidence thresholds

2. **Poor ingredient recognition**
   - Use higher resolution images
   - Enable text preprocessing
   - Check ingredient database coverage

3. **Slow processing**
   - Use smaller YOLO models (yolov8n vs yolov8s)
   - Disable GPU if causing issues
   - Reduce image size

### Debug Mode

Enable verbose logging for debugging:
```python
import logging
logging.getLogger('IngredientPipeline').setLevel(logging.DEBUG)
```

## Performance Tips

1. **Use GPU acceleration** when available
2. **Batch process** multiple images for efficiency
3. **Cache models** for repeated processing
4. **Optimize image size** - don't process unnecessarily large images
5. **Use appropriate YOLO model** - balance speed vs accuracy

## Next Steps

After running the examples:

1. Try with your own recipe images
2. Experiment with different configurations
3. Train custom models on your specific data
4. Integrate the pipeline into your applications
5. Contribute improvements back to the project