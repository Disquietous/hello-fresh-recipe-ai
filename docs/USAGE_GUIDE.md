# Recipe Processing API - Comprehensive Usage Guide

This guide provides practical examples of how to extract ingredients from different types of recipe images using the Recipe Processing API. Whether you're working with cookbook pages, handwritten recipe cards, digital screenshots, or blog images, this guide will help you get the best results.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Recipe Image Types](#recipe-image-types)
3. [Basic Usage Examples](#basic-usage-examples)
4. [Advanced Configuration](#advanced-configuration)
5. [Batch Processing](#batch-processing)
6. [Error Handling](#error-handling)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

Ensure your environment is set up:

```bash
# Clone the repository
git clone https://github.com/your-org/hello-fresh-recipe-ai.git
cd hello-fresh-recipe-ai

# Set up the environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download required models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Basic Example

```python
from src.ingredient_pipeline import IngredientExtractionPipeline

# Initialize the pipeline
pipeline = IngredientExtractionPipeline()

# Process a recipe image
result = pipeline.process_image('path/to/recipe.jpg')

# Display results
for ingredient in result['ingredients']:
    print(f"{ingredient['quantity']} {ingredient['unit']} {ingredient['ingredient_name']}")
```

### API Example

```bash
# Start the API server
python -m uvicorn src.api.recipe_processing_api:app --host 0.0.0.0 --port 8000

# Process an image via API
curl -X POST "http://localhost:8000/process" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@recipe.jpg" \
  -F "format_hint=cookbook"
```

## Recipe Image Types

The system is optimized for different types of recipe images, each with specific characteristics and recommended settings.

### 1. Printed Cookbook Pages

**Characteristics:**
- Clean, professional typography
- High contrast black text on white background
- Consistent formatting and layout
- Multiple columns or structured sections

**Recommended Settings:**
```python
cookbook_config = {
    'format_hint': 'cookbook',
    'ocr_engine': 'tesseract',
    'confidence_threshold': 0.3,
    'quality_threshold': 0.7,
    'language_hint': 'en'
}
```

**Example:**
```python
# Process cookbook page
result = pipeline.process_image(
    'examples/cookbook_page.jpg',
    format_hint='cookbook',
    ocr_engine='tesseract',
    confidence_threshold=0.3
)

print(f"Detected format: {result['format_type']}")
print(f"Quality score: {result['quality_score']:.2f}")
print(f"Found {len(result['ingredients'])} ingredients")
```

### 2. Handwritten Recipe Cards

**Characteristics:**
- Personal handwriting (varying legibility)
- Informal abbreviations and notes
- Possible stains or aging
- Inconsistent text size and spacing

**Recommended Settings:**
```python
handwritten_config = {
    'format_hint': 'handwritten',
    'ocr_engine': 'easyocr',
    'confidence_threshold': 0.2,  # Lower threshold for handwriting
    'quality_threshold': 0.5,
    'enable_preprocessing': True
}
```

**Example:**
```python
# Process handwritten recipe
result = pipeline.process_image(
    'examples/handwritten_card.jpg',
    format_hint='handwritten',
    ocr_engine='easyocr',
    confidence_threshold=0.2,
    enable_preprocessing=True
)

# Handwritten text often needs manual review
for ingredient in result['ingredients']:
    if ingredient['confidence'] < 0.5:
        print(f"Low confidence: {ingredient['ingredient_name']} (confidence: {ingredient['confidence']:.2f})")
```

### 3. Digital Recipe Screenshots

**Characteristics:**
- Clean digital fonts
- Possible compression artifacts
- Varied background colors
- May include UI elements (buttons, ads)

**Recommended Settings:**
```python
digital_config = {
    'format_hint': 'digital',
    'ocr_engine': 'paddleocr',
    'confidence_threshold': 0.4,
    'quality_threshold': 0.6,
    'filter_ui_elements': True
}
```

**Example:**
```python
# Process digital screenshot
result = pipeline.process_image(
    'examples/app_screenshot.png',
    format_hint='digital',
    ocr_engine='paddleocr',
    confidence_threshold=0.4
)

# Filter out likely UI elements
ingredients = [
    ing for ing in result['ingredients']
    if not any(word in ing['ingredient_name'].lower() 
              for word in ['button', 'click', 'share', 'save'])
]
```

### 4. Recipe Blog Images

**Characteristics:**
- Mixed content (text + images)
- Decorative fonts and styling
- Variable layouts
- Possible watermarks or overlays

**Recommended Settings:**
```python
blog_config = {
    'format_hint': 'blog',
    'ocr_engine': 'paddleocr',
    'confidence_threshold': 0.35,
    'quality_threshold': 0.6,
    'ignore_decorative_text': True
}
```

**Example:**
```python
# Process blog recipe
result = pipeline.process_image(
    'examples/blog_recipe.jpg',
    format_hint='blog',
    ocr_engine='paddleocr',
    confidence_threshold=0.35
)

# Blog images often have decorative text to filter
filtered_ingredients = []
for ingredient in result['ingredients']:
    # Skip decorative or promotional text
    text = ingredient['ingredient_name'].lower()
    if not any(word in text for word in ['recipe', 'blog', 'website', 'copyright']):
        filtered_ingredients.append(ingredient)
```

## Basic Usage Examples

### Example 1: Simple Ingredient Extraction

```python
from src.ingredient_pipeline import IngredientExtractionPipeline

def extract_ingredients_simple(image_path):
    """Simple ingredient extraction with default settings"""
    pipeline = IngredientExtractionPipeline()
    
    try:
        result = pipeline.process_image(image_path)
        
        print(f"Successfully processed: {result['filename']}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Quality score: {result['quality_score']:.2f}")
        print(f"Confidence: {result['confidence_score']:.2f}")
        print("\nIngredients found:")
        print("-" * 40)
        
        for i, ingredient in enumerate(result['ingredients'], 1):
            quantity = ingredient.get('quantity', '')
            unit = ingredient.get('unit', '')
            name = ingredient['ingredient_name']
            confidence = ingredient['confidence']
            
            print(f"{i:2d}. {quantity} {unit} {name} (confidence: {confidence:.2f})")
        
        return result
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Usage
result = extract_ingredients_simple('examples/recipe1.jpg')
```

### Example 2: Multi-Language Recipe Processing

```python
def process_multilingual_recipe(image_path, language='auto'):
    """Process recipes in different languages"""
    pipeline = IngredientExtractionPipeline()
    
    # Language-specific configurations
    language_configs = {
        'en': {'ocr_engine': 'tesseract', 'confidence_threshold': 0.3},
        'es': {'ocr_engine': 'easyocr', 'confidence_threshold': 0.25},
        'fr': {'ocr_engine': 'easyocr', 'confidence_threshold': 0.25},
        'de': {'ocr_engine': 'paddleocr', 'confidence_threshold': 0.3},
        'it': {'ocr_engine': 'easyocr', 'confidence_threshold': 0.25},
    }
    
    # Auto-detect language if not specified
    if language == 'auto':
        # Quick language detection pass
        quick_result = pipeline.process_image(
            image_path,
            ocr_engine='easyocr',
            confidence_threshold=0.1,
            enable_caching=False
        )
        language = quick_result.get('language', 'en')
        print(f"Detected language: {language}")
    
    # Use language-specific configuration
    config = language_configs.get(language, language_configs['en'])
    
    result = pipeline.process_image(
        image_path,
        language_hint=language,
        **config
    )
    
    print(f"Processing language: {language}")
    print(f"Ingredients in {language}:")
    for ingredient in result['ingredients']:
        # Show both original and English names if available
        original_name = ingredient['ingredient_name']
        english_name = ingredient.get('ingredient_name_en', original_name)
        
        if original_name != english_name:
            print(f"  {original_name} ({english_name})")
        else:
            print(f"  {original_name}")
    
    return result

# Usage examples
spanish_result = process_multilingual_recipe('examples/receta_spanish.jpg', 'es')
auto_detect_result = process_multilingual_recipe('examples/recipe_unknown.jpg', 'auto')
```

### Example 3: Recipe Format Auto-Detection

```python
def process_with_auto_detection(image_path):
    """Automatically detect recipe format and optimize processing"""
    pipeline = IngredientExtractionPipeline()
    
    # First pass: format detection
    print("Analyzing image format...")
    format_result = pipeline.analyze_format(image_path)
    
    detected_format = format_result['detected_format']
    confidence = format_result['confidence']
    characteristics = format_result['characteristics']
    
    print(f"Detected format: {detected_format} (confidence: {confidence:.2f})")
    print(f"Image characteristics:")
    for key, value in characteristics.items():
        print(f"  {key}: {value}")
    
    # Optimize settings based on detected format
    format_settings = {
        'printed_cookbook': {
            'ocr_engine': 'tesseract',
            'confidence_threshold': 0.3,
            'quality_threshold': 0.7
        },
        'handwritten': {
            'ocr_engine': 'easyocr',
            'confidence_threshold': 0.2,
            'quality_threshold': 0.5,
            'enable_preprocessing': True
        },
        'digital_screenshot': {
            'ocr_engine': 'paddleocr',
            'confidence_threshold': 0.4,
            'quality_threshold': 0.6
        },
        'recipe_blog': {
            'ocr_engine': 'paddleocr',
            'confidence_threshold': 0.35,
            'quality_threshold': 0.6
        }
    }
    
    settings = format_settings.get(detected_format, format_settings['printed_cookbook'])
    
    print(f"\nOptimized settings for {detected_format}:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
    
    # Process with optimized settings
    print("\nProcessing with optimized settings...")
    result = pipeline.process_image(
        image_path,
        format_hint=detected_format,
        **settings
    )
    
    return result

# Usage
result = process_with_auto_detection('examples/mystery_recipe.jpg')
```

## Advanced Configuration

### Custom OCR Engine Configuration

```python
def process_with_custom_ocr_config(image_path):
    """Use custom OCR engine configurations for better accuracy"""
    
    # Custom EasyOCR configuration
    easyocr_config = {
        'ocr_engine': 'easyocr',
        'ocr_config': {
            'detail': 1,  # Return bounding box coordinates
            'width_ths': 0.7,
            'height_ths': 0.7,
            'decoder': 'beamsearch',
            'beamWidth': 5
        }
    }
    
    # Custom Tesseract configuration
    tesseract_config = {
        'ocr_engine': 'tesseract',
        'ocr_config': {
            'psm': 6,  # Page segmentation mode
            'oem': 3,  # OCR Engine Mode
            'config': '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/- '
        }
    }
    
    # Custom PaddleOCR configuration
    paddleocr_config = {
        'ocr_engine': 'paddleocr',
        'ocr_config': {
            'use_angle_cls': True,
            'lang': 'en',
            'det_model_dir': None,
            'rec_model_dir': None,
            'cls_model_dir': None
        }
    }
    
    pipeline = IngredientExtractionPipeline()
    
    # Try different configurations and compare results
    configs = [
        ('EasyOCR', easyocr_config),
        ('Tesseract', tesseract_config),
        ('PaddleOCR', paddleocr_config)
    ]
    
    results = {}
    for name, config in configs:
        print(f"Testing {name} configuration...")
        try:
            result = pipeline.process_image(image_path, **config)
            results[name] = {
                'ingredient_count': len(result['ingredients']),
                'avg_confidence': sum(ing['confidence'] for ing in result['ingredients']) / len(result['ingredients']) if result['ingredients'] else 0,
                'processing_time': result['processing_time'],
                'quality_score': result['quality_score']
            }
            print(f"  {name}: {results[name]['ingredient_count']} ingredients, "
                  f"avg confidence: {results[name]['avg_confidence']:.2f}, "
                  f"time: {results[name]['processing_time']:.2f}s")
        except Exception as e:
            print(f"  {name}: Failed - {e}")
            results[name] = None
    
    # Select best configuration
    best_config = max(
        [(name, config) for name, config in configs if results[name] is not None],
        key=lambda x: results[x[0]]['avg_confidence'] * results[x[0]]['ingredient_count']
    )
    
    print(f"\nBest configuration: {best_config[0]}")
    return pipeline.process_image(image_path, **best_config[1])

# Usage
result = process_with_custom_ocr_config('examples/challenging_recipe.jpg')
```

### Image Preprocessing Options

```python
def process_with_preprocessing(image_path):
    """Apply various preprocessing techniques for better OCR accuracy"""
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
    
    pipeline = IngredientExtractionPipeline()
    
    # Load and analyze image
    with Image.open(image_path) as img:
        print(f"Original image: {img.size}, mode: {img.mode}")
    
    # Preprocessing options
    preprocessing_options = {
        'basic_cleanup': {
            'resize_factor': 2.0,
            'enhance_contrast': 1.5,
            'enhance_sharpness': 1.2,
            'denoise': True
        },
        'high_contrast': {
            'resize_factor': 1.5,
            'enhance_contrast': 2.0,
            'enhance_brightness': 1.1,
            'convert_grayscale': True
        },
        'handwriting_optimized': {
            'resize_factor': 2.5,
            'enhance_contrast': 1.8,
            'enhance_sharpness': 1.5,
            'gaussian_blur_radius': 0.5,
            'threshold_binary': True
        }
    }
    
    def apply_preprocessing(image_path, options):
        """Apply preprocessing transformations"""
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize for better OCR
            if 'resize_factor' in options:
                new_size = (
                    int(img.width * options['resize_factor']),
                    int(img.height * options['resize_factor'])
                )
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Enhance contrast
            if 'enhance_contrast' in options:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(options['enhance_contrast'])
            
            # Enhance brightness
            if 'enhance_brightness' in options:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(options['enhance_brightness'])
            
            # Enhance sharpness
            if 'enhance_sharpness' in options:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(options['enhance_sharpness'])
            
            # Convert to grayscale
            if options.get('convert_grayscale'):
                img = img.convert('L')
            
            # Apply Gaussian blur
            if 'gaussian_blur_radius' in options:
                img = img.filter(ImageFilter.GaussianBlur(
                    radius=options['gaussian_blur_radius']
                ))
            
            # Apply denoising
            if options.get('denoise'):
                # Convert to OpenCV format for advanced processing
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
                img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            
            # Binary thresholding
            if options.get('threshold_binary'):
                if img.mode != 'L':
                    img = img.convert('L')
                cv_img = np.array(img)
                _, cv_img = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img = Image.fromarray(cv_img)
            
            return img
    
    # Test different preprocessing approaches
    results = {}
    for approach_name, options in preprocessing_options.items():
        print(f"\nTesting {approach_name} preprocessing...")
        
        try:
            # Apply preprocessing
            processed_img = apply_preprocessing(image_path, options)
            
            # Save temporary processed image
            temp_path = f"temp_processed_{approach_name}.jpg"
            processed_img.save(temp_path, 'JPEG', quality=95)
            
            # Process with pipeline
            result = pipeline.process_image(
                temp_path,
                confidence_threshold=0.25,
                enable_caching=False
            )
            
            results[approach_name] = {
                'ingredients': result['ingredients'],
                'quality_score': result['quality_score'],
                'confidence_score': result['confidence_score'],
                'processing_time': result['processing_time']
            }
            
            print(f"  Found {len(result['ingredients'])} ingredients")
            print(f"  Quality: {result['quality_score']:.2f}")
            print(f"  Confidence: {result['confidence_score']:.2f}")
            
            # Clean up temp file
            import os
            os.remove(temp_path)
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[approach_name] = None
    
    # Compare results and select best
    best_approach = max(
        [(name, res) for name, res in results.items() if res is not None],
        key=lambda x: x[1]['quality_score'] * x[1]['confidence_score']
    )
    
    print(f"\nBest preprocessing approach: {best_approach[0]}")
    return best_approach[1]

# Usage
result = process_with_preprocessing('examples/low_quality_recipe.jpg')
```

## Batch Processing

### Processing Multiple Images

```python
def process_recipe_batch(image_paths, output_dir='batch_results'):
    """Process multiple recipe images efficiently"""
    import os
    import json
    from datetime import datetime
    
    pipeline = IngredientExtractionPipeline()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    batch_results = {
        'batch_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_images': len(image_paths),
        'processed_images': 0,
        'successful_images': 0,
        'failed_images': 0,
        'results': []
    }
    
    print(f"Starting batch processing of {len(image_paths)} images...")
    print(f"Results will be saved to: {output_dir}")
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\nProcessing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        try:
            # Process individual image
            result = pipeline.process_image(image_path)
            
            # Add batch metadata
            result['batch_index'] = i
            result['original_path'] = image_path
            result['status'] = 'success'
            
            batch_results['results'].append(result)
            batch_results['successful_images'] += 1
            
            print(f"  ✓ Success: {len(result['ingredients'])} ingredients found")
            
            # Save individual result
            result_filename = f"result_{i:03d}_{os.path.splitext(os.path.basename(image_path))[0]}.json"
            result_path = os.path.join(output_dir, result_filename)
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
        except Exception as e:
            error_result = {
                'batch_index': i,
                'original_path': image_path,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            batch_results['results'].append(error_result)
            batch_results['failed_images'] += 1
            
            print(f"  ✗ Failed: {e}")
        
        batch_results['processed_images'] += 1
    
    # Save batch summary
    summary_path = os.path.join(output_dir, f"batch_summary_{batch_results['batch_id']}.json")
    with open(summary_path, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    # Generate batch report
    generate_batch_report(batch_results, output_dir)
    
    print(f"\nBatch processing completed!")
    print(f"Total: {batch_results['total_images']}")
    print(f"Successful: {batch_results['successful_images']}")
    print(f"Failed: {batch_results['failed_images']}")
    print(f"Success rate: {batch_results['successful_images']/batch_results['total_images']*100:.1f}%")
    
    return batch_results

def generate_batch_report(batch_results, output_dir):
    """Generate a comprehensive batch processing report"""
    import csv
    
    # Create CSV report with all ingredients
    csv_path = os.path.join(output_dir, f"ingredients_report_{batch_results['batch_id']}.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'image_file', 'ingredient_name', 'quantity', 'unit',
            'confidence', 'preparation', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in batch_results['results']:
            if result['status'] == 'success':
                filename = os.path.basename(result['original_path'])
                for ingredient in result['ingredients']:
                    row = {
                        'image_file': filename,
                        'ingredient_name': ingredient['ingredient_name'],
                        'quantity': ingredient.get('quantity', ''),
                        'unit': ingredient.get('unit', ''),
                        'confidence': ingredient['confidence'],
                        'preparation': ingredient.get('preparation', ''),
                        'bbox_x1': ingredient['bbox']['x1'],
                        'bbox_y1': ingredient['bbox']['y1'],
                        'bbox_x2': ingredient['bbox']['x2'],
                        'bbox_y2': ingredient['bbox']['y2']
                    }
                    writer.writerow(row)
    
    print(f"CSV report saved: {csv_path}")

# Usage example
image_list = [
    'examples/cookbook1.jpg',
    'examples/cookbook2.jpg',
    'examples/handwritten1.jpg',
    'examples/blog_recipe1.jpg',
    'examples/digital_screenshot1.png'
]

batch_results = process_recipe_batch(image_list, 'my_batch_results')
```

### Async Batch Processing

```python
import asyncio
import aiofiles
from typing import List

async def process_recipe_batch_async(image_paths: List[str], max_concurrent: int = 4):
    """Process multiple recipe images asynchronously for better performance"""
    pipeline = IngredientExtractionPipeline()
    
    async def process_single_image(image_path: str, semaphore: asyncio.Semaphore):
        async with semaphore:
            try:
                print(f"Processing: {os.path.basename(image_path)}")
                result = await pipeline.process_image_async(image_path)
                result['status'] = 'success'
                return result
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                return {
                    'filename': os.path.basename(image_path),
                    'status': 'failed',
                    'error': str(e)
                }
    
    # Create semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all images
    tasks = [
        process_single_image(image_path, semaphore)
        for image_path in image_paths
    ]
    
    print(f"Starting async batch processing of {len(image_paths)} images...")
    print(f"Max concurrent: {max_concurrent}")
    
    # Process all images concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Separate successful and failed results
    successful = [r for r in results if isinstance(r, dict) and r.get('status') == 'success']
    failed = [r for r in results if isinstance(r, dict) and r.get('status') == 'failed']
    exceptions = [r for r in results if isinstance(r, Exception)]
    
    print(f"\nAsync batch processing completed!")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Exceptions: {len(exceptions)}")
    
    return {
        'successful': successful,
        'failed': failed,
        'exceptions': exceptions
    }

# Usage
async def main():
    image_paths = ['recipe1.jpg', 'recipe2.jpg', 'recipe3.jpg']
    results = await process_recipe_batch_async(image_paths, max_concurrent=2)
    return results

# Run async batch processing
# results = asyncio.run(main())
```

## Error Handling

### Robust Error Handling

```python
from src.api.error_handling import ProcessingError, ValidationError
import logging

def process_with_error_handling(image_path):
    """Process image with comprehensive error handling"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    pipeline = IngredientExtractionPipeline()
    
    try:
        # Validate input
        if not os.path.exists(image_path):
            raise ValidationError(f"Image file not found: {image_path}")
        
        # Check file size
        file_size = os.path.getsize(image_path)
        max_size = 20 * 1024 * 1024  # 20MB
        if file_size > max_size:
            raise ValidationError(f"File too large: {file_size} bytes (max: {max_size})")
        
        # Check file format
        try:
            with Image.open(image_path) as img:
                if img.format not in ['JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF']:
                    raise ValidationError(f"Unsupported format: {img.format}")
        except Exception as e:
            raise ValidationError(f"Invalid image file: {e}")
        
        logger.info(f"Starting processing of {image_path}")
        
        # Process with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = pipeline.process_image(
                    image_path,
                    confidence_threshold=0.25,
                    enable_caching=True
                )
                
                # Validate results
                if not result.get('ingredients'):
                    logger.warning("No ingredients found in image")
                    if attempt < max_retries - 1:
                        logger.info("Retrying with lower confidence threshold...")
                        result = pipeline.process_image(
                            image_path,
                            confidence_threshold=0.1,
                            enable_caching=False
                        )
                
                logger.info(f"Successfully processed {image_path}")
                return result
                
            except ProcessingError as e:
                logger.error(f"Processing error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
            
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise ProcessingError(f"Failed after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)
    
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {
            'status': 'validation_error',
            'error': str(e),
            'filename': os.path.basename(image_path)
        }
    
    except ProcessingError as e:
        logger.error(f"Processing error: {e}")
        return {
            'status': 'processing_error',
            'error': str(e),
            'filename': os.path.basename(image_path)
        }
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            'status': 'unexpected_error',
            'error': str(e),
            'filename': os.path.basename(image_path)
        }

# Usage
result = process_with_error_handling('examples/problematic_image.jpg')
if result.get('status') and 'error' in result['status']:
    print(f"Error: {result['error']}")
else:
    print(f"Success: Found {len(result['ingredients'])} ingredients")
```

### Graceful Degradation

```python
def process_with_fallback_strategies(image_path):
    """Process image with multiple fallback strategies"""
    pipeline = IngredientExtractionPipeline()
    
    # Strategy 1: Optimal settings
    strategies = [
        {
            'name': 'optimal',
            'config': {
                'ocr_engine': 'paddleocr',
                'confidence_threshold': 0.4,
                'quality_threshold': 0.7
            }
        },
        # Strategy 2: Lower thresholds
        {
            'name': 'permissive',
            'config': {
                'ocr_engine': 'easyocr',
                'confidence_threshold': 0.2,
                'quality_threshold': 0.5
            }
        },
        # Strategy 3: Different OCR engine
        {
            'name': 'alternative_ocr',
            'config': {
                'ocr_engine': 'tesseract',
                'confidence_threshold': 0.3,
                'enable_preprocessing': True
            }
        },
        # Strategy 4: Last resort
        {
            'name': 'minimal',
            'config': {
                'ocr_engine': 'easyocr',
                'confidence_threshold': 0.1,
                'quality_threshold': 0.3,
                'enable_preprocessing': True
            }
        }
    ]
    
    for strategy in strategies:
        try:
            print(f"Trying {strategy['name']} strategy...")
            
            result = pipeline.process_image(
                image_path,
                **strategy['config']
            )
            
            # Check if we got reasonable results
            if result['ingredients'] and len(result['ingredients']) >= 3:
                print(f"Success with {strategy['name']} strategy!")
                result['strategy_used'] = strategy['name']
                return result
            else:
                print(f"{strategy['name']} strategy found {len(result['ingredients'])} ingredients")
        
        except Exception as e:
            print(f"{strategy['name']} strategy failed: {e}")
            continue
    
    # If all strategies fail, return empty result
    print("All strategies failed, returning empty result")
    return {
        'filename': os.path.basename(image_path),
        'ingredients': [],
        'strategy_used': 'none',
        'status': 'all_strategies_failed'
    }

# Usage
result = process_with_fallback_strategies('examples/difficult_image.jpg')
print(f"Strategy used: {result['strategy_used']}")
```

## Performance Optimization

### Memory-Efficient Processing

```python
def process_large_images_efficiently(image_paths):
    """Process large images with memory optimization"""
    import gc
    from PIL import Image
    
    pipeline = IngredientExtractionPipeline()
    results = []
    
    def optimize_image_for_processing(image_path, max_dimension=2048):
        """Optimize image size while preserving text readability"""
        with Image.open(image_path) as img:
            width, height = img.size
            max_dim = max(width, height)
            
            if max_dim > max_dimension:
                ratio = max_dimension / max_dim
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save optimized image temporarily
                temp_path = f"temp_optimized_{os.path.basename(image_path)}"
                img.save(temp_path, 'JPEG', quality=90, optimize=True)
                return temp_path
            
            return image_path
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        try:
            # Optimize image size
            optimized_path = optimize_image_for_processing(image_path)
            
            # Process optimized image
            result = pipeline.process_image(
                optimized_path,
                enable_caching=False  # Disable caching for large batches
            )
            
            results.append(result)
            
            # Clean up temporary file if created
            if optimized_path != image_path:
                os.remove(optimized_path)
            
            # Force garbage collection every 10 images
            if (i + 1) % 10 == 0:
                gc.collect()
                print(f"Memory cleanup after {i+1} images")
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'filename': os.path.basename(image_path),
                'error': str(e),
                'status': 'failed'
            })
    
    return results

# Usage
large_image_paths = [f"large_recipe_{i}.jpg" for i in range(100)]
results = process_large_images_efficiently(large_image_paths)
```

### Caching for Performance

```python
def process_with_intelligent_caching(image_paths):
    """Use intelligent caching to improve performance"""
    import hashlib
    
    pipeline = IngredientExtractionPipeline()
    cache_hits = 0
    cache_misses = 0
    
    def get_image_hash(image_path):
        """Generate hash for image to detect duplicates"""
        hasher = hashlib.md5()
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    # Pre-process: Group images by hash to find duplicates
    image_groups = {}
    for image_path in image_paths:
        img_hash = get_image_hash(image_path)
        if img_hash not in image_groups:
            image_groups[img_hash] = []
        image_groups[img_hash].append(image_path)
    
    print(f"Found {len(image_groups)} unique images out of {len(image_paths)} total")
    
    results = {}
    
    # Process unique images only
    for img_hash, paths in image_groups.items():
        representative_path = paths[0]  # Process first image in group
        print(f"Processing {os.path.basename(representative_path)} (represents {len(paths)} images)")
        
        try:
            result = pipeline.process_image(
                representative_path,
                enable_caching=True
            )
            
            # Apply result to all images in group
            for path in paths:
                result_copy = result.copy()
                result_copy['filename'] = os.path.basename(path)
                result_copy['original_path'] = path
                results[path] = result_copy
                
                if path == representative_path:
                    cache_misses += 1
                else:
                    cache_hits += 1
        
        except Exception as e:
            print(f"Error processing group: {e}")
            for path in paths:
                results[path] = {
                    'filename': os.path.basename(path),
                    'error': str(e),
                    'status': 'failed'
                }
    
    print(f"Cache performance: {cache_hits} hits, {cache_misses} misses")
    print(f"Cache hit rate: {cache_hits/(cache_hits+cache_misses)*100:.1f}%")
    
    return list(results.values())

# Usage
image_paths_with_duplicates = [
    'recipe1.jpg', 'recipe2.jpg', 'recipe1.jpg',  # recipe1.jpg is duplicate
    'recipe3.jpg', 'recipe2.jpg', 'recipe4.jpg'   # recipe2.jpg is duplicate
]
results = process_with_intelligent_caching(image_paths_with_duplicates)
```

## Best Practices

### 1. Image Quality Guidelines

```python
def analyze_image_quality(image_path):
    """Analyze image quality and provide recommendations"""
    from PIL import Image, ImageStat
    import numpy as np
    
    with Image.open(image_path) as img:
        # Basic image information
        width, height = img.size
        mode = img.mode
        format_type = img.format
        
        # Convert to grayscale for analysis
        gray_img = img.convert('L')
        
        # Calculate image statistics
        stat = ImageStat.Stat(gray_img)
        mean_brightness = stat.mean[0]
        std_brightness = stat.stddev[0]
        
        # Convert to numpy for advanced analysis
        img_array = np.array(gray_img)
        
        # Calculate sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Analyze text regions (simplified)
        edges = cv2.Canny(img_array, 50, 150)
        text_density = np.sum(edges > 0) / (width * height)
        
        # Generate quality assessment
        quality_score = 0
        recommendations = []
        
        # Resolution check
        min_dimension = min(width, height)
        if min_dimension >= 1200:
            quality_score += 25
        elif min_dimension >= 800:
            quality_score += 20
            recommendations.append("Consider using higher resolution images (1200px+ minimum)")
        else:
            quality_score += 10
            recommendations.append("Image resolution is low. Use at least 800px minimum dimension")
        
        # Brightness check
        if 80 <= mean_brightness <= 180:
            quality_score += 25
        else:
            quality_score += 10
            if mean_brightness < 80:
                recommendations.append("Image is too dark. Increase brightness or improve lighting")
            else:
                recommendations.append("Image is too bright. Reduce exposure or improve lighting")
        
        # Contrast check
        if std_brightness >= 40:
            quality_score += 25
        else:
            quality_score += 10
            recommendations.append("Low contrast. Increase contrast or improve lighting conditions")
        
        # Sharpness check
        if sharpness >= 100:
            quality_score += 25
        else:
            quality_score += 10
            recommendations.append("Image appears blurry. Ensure focus and reduce camera shake")
        
        quality_assessment = {
            'image_path': image_path,
            'dimensions': f"{width}x{height}",
            'format': format_type,
            'quality_score': quality_score,
            'mean_brightness': mean_brightness,
            'contrast_std': std_brightness,
            'sharpness': sharpness,
            'text_density': text_density,
            'recommendations': recommendations
        }
        
        return quality_assessment

def print_quality_assessment(assessment):
    """Print formatted quality assessment"""
    print(f"\nImage Quality Assessment: {os.path.basename(assessment['image_path'])}")
    print("=" * 50)
    print(f"Dimensions: {assessment['dimensions']}")
    print(f"Format: {assessment['format']}")
    print(f"Quality Score: {assessment['quality_score']}/100")
    print(f"Brightness: {assessment['mean_brightness']:.1f}")
    print(f"Contrast: {assessment['contrast_std']:.1f}")
    print(f"Sharpness: {assessment['sharpness']:.1f}")
    print(f"Text Density: {assessment['text_density']:.3f}")
    
    if assessment['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(assessment['recommendations'], 1):
            print(f"{i}. {rec}")
    else:
        print("\n✓ Image quality is good for processing")

# Usage
assessment = analyze_image_quality('examples/recipe.jpg')
print_quality_assessment(assessment)
```

### 2. Ingredient Validation and Cleaning

```python
def validate_and_clean_ingredients(ingredients):
    """Validate and clean extracted ingredients"""
    import re
    
    # Common ingredient validation patterns
    quantity_pattern = re.compile(r'^[\d\s/.-]+$')
    unit_pattern = re.compile(r'^(cup|cups|tsp|tbsp|oz|lb|g|kg|ml|l|piece|pieces|clove|cloves)s?$', re.IGNORECASE)
    
    # Common false positives to filter out
    false_positives = {
        'ingredients', 'recipe', 'instructions', 'directions', 'method',
        'serves', 'serving', 'prep', 'time', 'cook', 'total', 'difficulty',
        'page', 'book', 'author', 'website', 'blog', 'copyright', 'photo',
        'image', 'click', 'share', 'save', 'print', 'email'
    }
    
    cleaned_ingredients = []
    
    for ingredient in ingredients:
        # Skip ingredients with very low confidence
        if ingredient['confidence'] < 0.15:
            continue
        
        name = ingredient['ingredient_name'].lower().strip()
        
        # Skip false positives
        if name in false_positives:
            continue
        
        # Skip if name is too short or too long
        if len(name) < 2 or len(name) > 50:
            continue
        
        # Skip if name contains only numbers or special characters
        if not re.search(r'[a-zA-Z]', name):
            continue
        
        # Clean quantity
        quantity = ingredient.get('quantity', '').strip()
        if quantity and not quantity_pattern.match(quantity):
            quantity = ''  # Clear invalid quantity
        
        # Clean unit
        unit = ingredient.get('unit', '').strip()
        if unit and not unit_pattern.match(unit):
            unit = ''  # Clear invalid unit
        
        # Normalize ingredient name
        normalized_name = normalize_ingredient_name(name)
        
        cleaned_ingredient = {
            'ingredient_name': normalized_name,
            'ingredient_name_original': ingredient['ingredient_name'],
            'quantity': quantity,
            'unit': unit,
            'preparation': ingredient.get('preparation', ''),
            'confidence': ingredient['confidence'],
            'bbox': ingredient['bbox']
        }
        
        cleaned_ingredients.append(cleaned_ingredient)
    
    return cleaned_ingredients

def normalize_ingredient_name(name):
    """Normalize ingredient names for consistency"""
    # Convert to lowercase
    name = name.lower()
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    # Common normalizations
    normalizations = {
        'tomatoe': 'tomato',
        'potatoe': 'potato',
        'chesse': 'cheese',
        'flowur': 'flour',
        'suggar': 'sugar',
        'oinion': 'onion',
        'carrit': 'carrot',
        'chiken': 'chicken',
        'beaf': 'beef',
        'porck': 'pork'
    }
    
    for wrong, correct in normalizations.items():
        name = name.replace(wrong, correct)
    
    # Remove common prefixes/suffixes that might be OCR artifacts
    artifacts = ['fresh', 'organic', 'chopped', 'diced', 'sliced', 'grated']
    words = name.split()
    filtered_words = [word for word in words if word not in artifacts]
    
    if filtered_words:  # Make sure we don't remove everything
        name = ' '.join(filtered_words)
    
    return name.title()  # Convert to title case

# Usage
raw_ingredients = pipeline.process_image('recipe.jpg')['ingredients']
cleaned_ingredients = validate_and_clean_ingredients(raw_ingredients)

print("Raw ingredients:", len(raw_ingredients))
print("Cleaned ingredients:", len(cleaned_ingredients))
for ingredient in cleaned_ingredients[:5]:  # Show first 5
    print(f"  {ingredient['quantity']} {ingredient['unit']} {ingredient['ingredient_name']}")
```

### 3. Result Formatting and Export

```python
def format_ingredients_for_export(ingredients, format_type='text'):
    """Format ingredients for different output formats"""
    
    if format_type == 'text':
        return format_ingredients_text(ingredients)
    elif format_type == 'json':
        return format_ingredients_json(ingredients)
    elif format_type == 'csv':
        return format_ingredients_csv(ingredients)
    elif format_type == 'markdown':
        return format_ingredients_markdown(ingredients)
    elif format_type == 'recipe_card':
        return format_ingredients_recipe_card(ingredients)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def format_ingredients_text(ingredients):
    """Format as plain text list"""
    lines = ["Ingredients:"]
    lines.append("-" * 20)
    
    for i, ingredient in enumerate(ingredients, 1):
        quantity = ingredient.get('quantity', '')
        unit = ingredient.get('unit', '')
        name = ingredient['ingredient_name']
        prep = ingredient.get('preparation', '')
        
        line_parts = [f"{i:2d}."]
        if quantity:
            line_parts.append(quantity)
        if unit:
            line_parts.append(unit)
        line_parts.append(name)
        if prep:
            line_parts.append(f"({prep})")
        
        lines.append(" ".join(line_parts))
    
    return "\n".join(lines)

def format_ingredients_json(ingredients):
    """Format as structured JSON"""
    import json
    
    formatted = {
        'ingredients': [],
        'summary': {
            'total_count': len(ingredients),
            'avg_confidence': sum(ing['confidence'] for ing in ingredients) / len(ingredients) if ingredients else 0
        }
    }
    
    for ingredient in ingredients:
        formatted_ingredient = {
            'name': ingredient['ingredient_name'],
            'amount': {
                'quantity': ingredient.get('quantity', ''),
                'unit': ingredient.get('unit', '')
            },
            'preparation': ingredient.get('preparation', ''),
            'confidence': round(ingredient['confidence'], 3),
            'position': {
                'x': (ingredient['bbox']['x1'] + ingredient['bbox']['x2']) // 2,
                'y': (ingredient['bbox']['y1'] + ingredient['bbox']['y2']) // 2
            }
        }
        formatted['ingredients'].append(formatted_ingredient)
    
    return json.dumps(formatted, indent=2)

def format_ingredients_csv(ingredients):
    """Format as CSV data"""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['Name', 'Quantity', 'Unit', 'Preparation', 'Confidence'])
    
    # Data rows
    for ingredient in ingredients:
        writer.writerow([
            ingredient['ingredient_name'],
            ingredient.get('quantity', ''),
            ingredient.get('unit', ''),
            ingredient.get('preparation', ''),
            round(ingredient['confidence'], 3)
        ])
    
    return output.getvalue()

def format_ingredients_markdown(ingredients):
    """Format as Markdown table"""
    lines = ["# Ingredients\n"]
    lines.append("| Name | Quantity | Unit | Preparation | Confidence |")
    lines.append("|------|----------|------|-------------|------------|")
    
    for ingredient in ingredients:
        quantity = ingredient.get('quantity', '')
        unit = ingredient.get('unit', '')
        name = ingredient['ingredient_name']
        prep = ingredient.get('preparation', '')
        conf = f"{ingredient['confidence']:.2f}"
        
        lines.append(f"| {name} | {quantity} | {unit} | {prep} | {conf} |")
    
    return "\n".join(lines)

def format_ingredients_recipe_card(ingredients):
    """Format as a nicely formatted recipe card"""
    lines = []
    lines.append("╔" + "═" * 48 + "╗")
    lines.append("║" + " " * 16 + "INGREDIENTS" + " " * 21 + "║")
    lines.append("╠" + "═" * 48 + "╣")
    
    for ingredient in ingredients:
        quantity = ingredient.get('quantity', '')
        unit = ingredient.get('unit', '')
        name = ingredient['ingredient_name']
        prep = ingredient.get('preparation', '')
        
        # Build ingredient line
        amount_text = f"{quantity} {unit}".strip()
        if amount_text:
            ingredient_line = f"{amount_text:<12} {name}"
        else:
            ingredient_line = name
        
        if prep:
            ingredient_line += f" ({prep})"
        
        # Truncate if too long
        if len(ingredient_line) > 46:
            ingredient_line = ingredient_line[:43] + "..."
        
        lines.append(f"║ {ingredient_line:<46} ║")
    
    lines.append("╚" + "═" * 48 + "╝")
    
    return "\n".join(lines)

# Usage examples
ingredients = pipeline.process_image('recipe.jpg')['ingredients']

print("Text format:")
print(format_ingredients_for_export(ingredients, 'text'))

print("\nMarkdown format:")
print(format_ingredients_for_export(ingredients, 'markdown'))

print("\nRecipe card format:")
print(format_ingredients_for_export(ingredients, 'recipe_card'))

# Save to file
with open('ingredients.json', 'w') as f:
    f.write(format_ingredients_for_export(ingredients, 'json'))

with open('ingredients.csv', 'w') as f:
    f.write(format_ingredients_for_export(ingredients, 'csv'))
```

## Troubleshooting

### Common Issues and Solutions

```python
def diagnose_processing_issues(image_path, result):
    """Diagnose common processing issues and suggest solutions"""
    
    issues = []
    suggestions = []
    
    # Check if any ingredients were found
    if not result.get('ingredients'):
        issues.append("No ingredients detected")
        suggestions.extend([
            "Try lowering the confidence threshold",
            "Use a different OCR engine (easyocr, paddleocr, tesseract)",
            "Enable image preprocessing",
            "Check image quality and lighting"
        ])
    
    # Check confidence scores
    if result.get('ingredients'):
        avg_confidence = sum(ing['confidence'] for ing in result['ingredients']) / len(result['ingredients'])
        if avg_confidence < 0.3:
            issues.append(f"Low average confidence: {avg_confidence:.2f}")
            suggestions.extend([
                "Improve image quality (resolution, focus, lighting)",
                "Try different OCR engine",
                "Use image preprocessing techniques"
            ])
    
    # Check processing time
    if result.get('processing_time', 0) > 30:
        issues.append(f"Slow processing: {result['processing_time']:.1f} seconds")
        suggestions.extend([
            "Resize image to smaller dimensions",
            "Use faster OCR engine",
            "Enable caching for repeated processing"
        ])
    
    # Check quality score
    if result.get('quality_score', 1) < 0.5:
        issues.append(f"Low quality score: {result['quality_score']:.2f}")
        suggestions.extend([
            "Improve image lighting and focus",
            "Increase image resolution",
            "Reduce noise and artifacts"
        ])
    
    # Analyze image characteristics
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            if min(width, height) < 600:
                issues.append("Low resolution image")
                suggestions.append("Use higher resolution images (at least 800px)")
            
            if max(width, height) > 4000:
                issues.append("Very high resolution may slow processing")
                suggestions.append("Consider resizing to 2000-3000px maximum")
    
    except Exception as e:
        issues.append(f"Cannot analyze image: {e}")
    
    # Generate diagnostic report
    print(f"\nDiagnostic Report for: {os.path.basename(image_path)}")
    print("=" * 50)
    
    if issues:
        print("Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        
        print("\nSuggested solutions:")
        for i, suggestion in enumerate(set(suggestions), 1):
            print(f"{i}. {suggestion}")
    else:
        print("✓ No issues detected")
    
    return {
        'issues': issues,
        'suggestions': list(set(suggestions))
    }

def auto_fix_common_issues(image_path):
    """Automatically try to fix common processing issues"""
    pipeline = IngredientExtractionPipeline()
    
    print(f"Auto-fixing issues for: {os.path.basename(image_path)}")
    
    # Strategy 1: Try with preprocessing
    print("Trying with image preprocessing...")
    try:
        result1 = pipeline.process_image(
            image_path,
            enable_preprocessing=True,
            confidence_threshold=0.25
        )
        if result1['ingredients'] and len(result1['ingredients']) >= 3:
            print("✓ Success with preprocessing")
            return result1
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
    
    # Strategy 2: Try different OCR engines
    for engine in ['easyocr', 'paddleocr', 'tesseract']:
        print(f"Trying with {engine}...")
        try:
            result2 = pipeline.process_image(
                image_path,
                ocr_engine=engine,
                confidence_threshold=0.2
            )
            if result2['ingredients'] and len(result2['ingredients']) >= 2:
                print(f"✓ Success with {engine}")
                return result2
        except Exception as e:
            print(f"✗ {engine} failed: {e}")
    
    # Strategy 3: Very permissive settings
    print("Trying with very permissive settings...")
    try:
        result3 = pipeline.process_image(
            image_path,
            confidence_threshold=0.1,
            quality_threshold=0.3,
            enable_preprocessing=True
        )
        print(f"✓ Found {len(result3['ingredients'])} ingredients with permissive settings")
        return result3
    except Exception as e:
        print(f"✗ Permissive settings failed: {e}")
    
    print("✗ Unable to process image with any strategy")
    return None

# Usage
result = pipeline.process_image('problematic_recipe.jpg')
diagnosis = diagnose_processing_issues('problematic_recipe.jpg', result)

if diagnosis['issues']:
    print("Attempting auto-fix...")
    fixed_result = auto_fix_common_issues('problematic_recipe.jpg')
```

This comprehensive usage guide provides practical examples for all major use cases of the Recipe Processing API. Users can refer to specific sections based on their recipe image types and requirements, with clear examples and best practices for optimal results.