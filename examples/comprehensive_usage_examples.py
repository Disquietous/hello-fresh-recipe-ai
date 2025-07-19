#!/usr/bin/env python3
"""
Comprehensive Usage Examples for Recipe Processing API
This file contains practical examples for different recipe image types and use cases.
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingredient_pipeline import IngredientExtractionPipeline

class RecipeProcessingExamples:
    """Collection of comprehensive usage examples"""
    
    def __init__(self):
        self.pipeline = IngredientExtractionPipeline()
        self.examples_dir = Path(__file__).parent
        self.results_dir = self.examples_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    def example_1_basic_cookbook_processing(self):
        """Example 1: Basic cookbook page processing"""
        print("="*60)
        print("EXAMPLE 1: Basic Cookbook Page Processing")
        print("="*60)
        
        # Simulate processing a cookbook page
        example_config = {
            'format_hint': 'cookbook',
            'ocr_engine': 'tesseract',
            'confidence_threshold': 0.3,
            'quality_threshold': 0.7,
            'language_hint': 'en'
        }
        
        print("Configuration for cookbook pages:")
        for key, value in example_config.items():
            print(f"  {key}: {value}")
        
        print("\nSimulated processing (replace with actual image path):")
        print("result = pipeline.process_image('cookbook_page.jpg', **config)")
        
        # Create mock result for demonstration
        mock_result = {
            'filename': 'cookbook_page.jpg',
            'format_type': 'printed_cookbook',
            'quality_score': 0.85,
            'confidence_score': 0.78,
            'processing_time': 8.5,
            'language': 'en',
            'ingredients': [
                {
                    'ingredient_name': 'flour',
                    'quantity': '2',
                    'unit': 'cups',
                    'preparation': 'sifted',
                    'confidence': 0.92,
                    'bbox': {'x1': 120, 'y1': 45, 'x2': 280, 'y2': 70}
                },
                {
                    'ingredient_name': 'sugar',
                    'quantity': '1',
                    'unit': 'cup',
                    'preparation': '',
                    'confidence': 0.89,
                    'bbox': {'x1': 120, 'y1': 75, 'x2': 250, 'y2': 100}
                },
                {
                    'ingredient_name': 'eggs',
                    'quantity': '3',
                    'unit': 'large',
                    'preparation': '',
                    'confidence': 0.95,
                    'bbox': {'x1': 120, 'y1': 105, 'x2': 220, 'y2': 130}
                },
                {
                    'ingredient_name': 'butter',
                    'quantity': '1/2',
                    'unit': 'cup',
                    'preparation': 'melted',
                    'confidence': 0.87,
                    'bbox': {'x1': 120, 'y1': 135, 'x2': 260, 'y2': 160}
                }
            ]
        }
        
        self._display_results(mock_result)
        self._save_example_result(mock_result, "example_1_cookbook")
        
        print("\nKey takeaways for cookbook pages:")
        print("- Use 'tesseract' OCR engine for clean printed text")
        print("- Set confidence threshold around 0.3")
        print("- Expect high quality and confidence scores")
        print("- Perfect for structured ingredient lists")
    
    def example_2_handwritten_recipe_card(self):
        """Example 2: Handwritten recipe card processing"""
        print("\n" + "="*60)
        print("EXAMPLE 2: Handwritten Recipe Card Processing")
        print("="*60)
        
        example_config = {
            'format_hint': 'handwritten',
            'ocr_engine': 'easyocr',
            'confidence_threshold': 0.2,  # Lower for handwriting
            'quality_threshold': 0.5,
            'enable_preprocessing': True
        }
        
        print("Configuration for handwritten recipes:")
        for key, value in example_config.items():
            print(f"  {key}: {value}")
        
        # Mock result with typical handwriting challenges
        mock_result = {
            'filename': 'grandmas_recipe.jpg',
            'format_type': 'handwritten',
            'quality_score': 0.62,
            'confidence_score': 0.45,
            'processing_time': 12.3,
            'language': 'en',
            'ingredients': [
                {
                    'ingredient_name': 'flour',
                    'quantity': '2',
                    'unit': 'c',  # Abbreviated units common in handwriting
                    'preparation': '',
                    'confidence': 0.68,
                    'bbox': {'x1': 85, 'y1': 120, 'x2': 180, 'y2': 145}
                },
                {
                    'ingredient_name': 'milk',
                    'quantity': '1',
                    'unit': 'cup',
                    'preparation': '',
                    'confidence': 0.42,  # Lower confidence typical
                    'bbox': {'x1': 85, 'y1': 150, 'x2': 165, 'y2': 175}
                },
                {
                    'ingredient_name': 'vanilla',
                    'quantity': '1',
                    'unit': 'tsp',
                    'preparation': '',
                    'confidence': 0.35,
                    'bbox': {'x1': 85, 'y1': 180, 'x2': 175, 'y2': 205}
                },
                {
                    'ingredient_name': 'baking powder',  # Often abbreviated as "b. powder"
                    'quantity': '2',
                    'unit': 'tsp',
                    'preparation': '',
                    'confidence': 0.28,
                    'bbox': {'x1': 85, 'y1': 210, 'x2': 220, 'y2': 235}
                }
            ]
        }
        
        self._display_results(mock_result)
        
        print("\nHandwriting-specific considerations:")
        print("- Lower confidence scores are normal (0.2-0.5)")
        print("- Enable preprocessing for better text clarity")
        print("- Watch for abbreviated units (c, tsp, T, etc.)")
        print("- Manual review recommended for low confidence items")
        
        # Show confidence analysis
        print("\nConfidence Analysis:")
        for ingredient in mock_result['ingredients']:
            conf = ingredient['confidence']
            status = "✓ Good" if conf > 0.5 else "⚠ Review" if conf > 0.3 else "✗ Manual check"
            print(f"  {ingredient['ingredient_name']}: {conf:.2f} - {status}")
        
        self._save_example_result(mock_result, "example_2_handwritten")
    
    def example_3_digital_recipe_screenshot(self):
        """Example 3: Digital recipe screenshot processing"""
        print("\n" + "="*60)
        print("EXAMPLE 3: Digital Recipe Screenshot Processing")
        print("="*60)
        
        example_config = {
            'format_hint': 'digital',
            'ocr_engine': 'paddleocr',
            'confidence_threshold': 0.4,
            'quality_threshold': 0.6,
            'filter_ui_elements': True
        }
        
        print("Configuration for digital screenshots:")
        for key, value in example_config.items():
            print(f"  {key}: {value}")
        
        # Mock result with digital-specific elements
        mock_result = {
            'filename': 'recipe_app_screenshot.png',
            'format_type': 'digital_screenshot',
            'quality_score': 0.72,
            'confidence_score': 0.81,
            'processing_time': 6.8,
            'language': 'en',
            'ingredients': [
                {
                    'ingredient_name': 'olive oil',
                    'quantity': '2',
                    'unit': 'tablespoons',
                    'preparation': 'extra virgin',
                    'confidence': 0.94,
                    'bbox': {'x1': 50, 'y1': 200, 'x2': 280, 'y2': 225}
                },
                {
                    'ingredient_name': 'garlic cloves',
                    'quantity': '3',
                    'unit': '',
                    'preparation': 'minced',
                    'confidence': 0.88,
                    'bbox': {'x1': 50, 'y1': 230, 'x2': 220, 'y2': 255}
                },
                {
                    'ingredient_name': 'tomatoes',
                    'quantity': '4',
                    'unit': 'large',
                    'preparation': 'diced',
                    'confidence': 0.91,
                    'bbox': {'x1': 50, 'y1': 260, 'x2': 240, 'y2': 285}
                },
                {
                    'ingredient_name': 'basil leaves',
                    'quantity': '1/4',
                    'unit': 'cup',
                    'preparation': 'fresh',
                    'confidence': 0.86,
                    'bbox': {'x1': 50, 'y1': 290, 'x2': 250, 'y2': 315}
                }
            ]
        }
        
        self._display_results(mock_result)
        
        print("\nDigital screenshot tips:")
        print("- PaddleOCR works well with digital fonts")
        print("- Watch for UI elements (buttons, ads) in results")
        print("- Good quality scores expected from digital sources")
        print("- Consider cropping to ingredients section only")
        
        # Demonstrate UI filtering
        print("\nFiltering UI elements (simulated):")
        ui_words = ['share', 'save', 'print', 'rate', 'comment', 'like']
        filtered_ingredients = [
            ing for ing in mock_result['ingredients']
            if not any(word in ing['ingredient_name'].lower() for word in ui_words)
        ]
        print(f"Original ingredients: {len(mock_result['ingredients'])}")
        print(f"After UI filtering: {len(filtered_ingredients)}")
        
        self._save_example_result(mock_result, "example_3_digital")
    
    def example_4_multilingual_processing(self):
        """Example 4: Multi-language recipe processing"""
        print("\n" + "="*60)
        print("EXAMPLE 4: Multi-Language Recipe Processing")
        print("="*60)
        
        # Example configurations for different languages
        language_configs = {
            'spanish': {
                'language_hint': 'es',
                'ocr_engine': 'easyocr',
                'confidence_threshold': 0.25
            },
            'french': {
                'language_hint': 'fr',
                'ocr_engine': 'easyocr',
                'confidence_threshold': 0.25
            },
            'german': {
                'language_hint': 'de',
                'ocr_engine': 'paddleocr',
                'confidence_threshold': 0.3
            }
        }
        
        # Mock results for different languages
        multilingual_results = {
            'spanish': {
                'filename': 'receta_espanola.jpg',
                'language': 'es',
                'ingredients': [
                    {
                        'ingredient_name': 'aceite de oliva',
                        'ingredient_name_en': 'olive oil',
                        'quantity': '3',
                        'unit': 'cucharadas',
                        'confidence': 0.87
                    },
                    {
                        'ingredient_name': 'ajo',
                        'ingredient_name_en': 'garlic',
                        'quantity': '2',
                        'unit': 'dientes',
                        'confidence': 0.92
                    },
                    {
                        'ingredient_name': 'tomates',
                        'ingredient_name_en': 'tomatoes',
                        'quantity': '4',
                        'unit': '',
                        'confidence': 0.89
                    }
                ]
            },
            'french': {
                'filename': 'recette_francaise.jpg',
                'language': 'fr',
                'ingredients': [
                    {
                        'ingredient_name': 'farine',
                        'ingredient_name_en': 'flour',
                        'quantity': '250',
                        'unit': 'g',
                        'confidence': 0.94
                    },
                    {
                        'ingredient_name': 'œufs',
                        'ingredient_name_en': 'eggs',
                        'quantity': '3',
                        'unit': '',
                        'confidence': 0.88
                    },
                    {
                        'ingredient_name': 'lait',
                        'ingredient_name_en': 'milk',
                        'quantity': '200',
                        'unit': 'ml',
                        'confidence': 0.91
                    }
                ]
            }
        }
        
        for lang, config in language_configs.items():
            print(f"\n{lang.title()} Recipe Configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            
            if lang in multilingual_results:
                result = multilingual_results[lang]
                print(f"\nSample ingredients in {lang}:")
                for ingredient in result['ingredients']:
                    original = ingredient['ingredient_name']
                    english = ingredient.get('ingredient_name_en', original)
                    quantity = ingredient.get('quantity', '')
                    unit = ingredient.get('unit', '')
                    
                    print(f"  {quantity} {unit} {original}")
                    if original != english:
                        print(f"    (English: {english})")
        
        print("\nMultilingual processing tips:")
        print("- EasyOCR performs well with European languages")
        print("- PaddleOCR good for German and other languages")
        print("- Enable automatic translation for ingredient names")
        print("- Consider regional measurement systems")
    
    def example_5_batch_processing(self):
        """Example 5: Batch processing multiple images"""
        print("\n" + "="*60)
        print("EXAMPLE 5: Batch Processing Multiple Images")
        print("="*60)
        
        # Simulate batch processing
        example_images = [
            'cookbook_page_1.jpg',
            'cookbook_page_2.jpg',
            'handwritten_card.jpg',
            'blog_recipe.jpg',
            'app_screenshot.png'
        ]
        
        print(f"Processing batch of {len(example_images)} images...")
        
        # Mock batch results
        batch_results = []
        total_ingredients = 0
        processing_times = []
        
        for i, image_name in enumerate(example_images, 1):
            print(f"\nProcessing {i}/{len(example_images)}: {image_name}")
            
            # Simulate processing
            mock_result = {
                'filename': image_name,
                'batch_index': i,
                'status': 'success',
                'processing_time': 5.0 + (i * 1.5),  # Simulated times
                'ingredient_count': 3 + (i % 4),  # Simulated counts
                'quality_score': 0.7 + (i * 0.05),
                'confidence_score': 0.6 + (i * 0.08)
            }
            
            batch_results.append(mock_result)
            total_ingredients += mock_result['ingredient_count']
            processing_times.append(mock_result['processing_time'])
            
            print(f"  ✓ Found {mock_result['ingredient_count']} ingredients")
            print(f"  ✓ Processing time: {mock_result['processing_time']:.1f}s")
            print(f"  ✓ Quality: {mock_result['quality_score']:.2f}")
        
        # Batch summary
        print(f"\nBatch Processing Summary:")
        print(f"Total images: {len(example_images)}")
        print(f"Successful: {len([r for r in batch_results if r['status'] == 'success'])}")
        print(f"Total ingredients found: {total_ingredients}")
        print(f"Average processing time: {sum(processing_times)/len(processing_times):.1f}s")
        print(f"Total batch time: {sum(processing_times):.1f}s")
        
        # Save batch results
        batch_summary = {
            'batch_info': {
                'total_images': len(example_images),
                'successful_images': len(batch_results),
                'total_ingredients': total_ingredients,
                'avg_processing_time': sum(processing_times)/len(processing_times),
                'total_time': sum(processing_times)
            },
            'results': batch_results
        }
        
        self._save_example_result(batch_summary, "example_5_batch")
        
        print("\nBatch processing best practices:")
        print("- Process images in parallel for better performance")
        print("- Use appropriate confidence thresholds per image type")
        print("- Implement error handling for failed images")
        print("- Save results incrementally for large batches")
    
    def example_6_error_handling_and_validation(self):
        """Example 6: Error handling and result validation"""
        print("\n" + "="*60)
        print("EXAMPLE 6: Error Handling and Result Validation")
        print("="*60)
        
        # Demonstrate various error scenarios and handling
        error_scenarios = [
            {
                'name': 'Low Quality Image',
                'error_type': 'quality_warning',
                'quality_score': 0.3,
                'confidence_score': 0.2,
                'ingredient_count': 1,
                'message': 'Image quality is poor, results may be unreliable'
            },
            {
                'name': 'No Ingredients Found',
                'error_type': 'no_results',
                'quality_score': 0.8,
                'confidence_score': 0.0,
                'ingredient_count': 0,
                'message': 'No ingredients detected in image'
            },
            {
                'name': 'Processing Timeout',
                'error_type': 'timeout',
                'message': 'Processing exceeded time limit'
            },
            {
                'name': 'Unsupported Format',
                'error_type': 'validation_error',
                'message': 'Image format not supported'
            }
        ]
        
        print("Common error scenarios and handling:")
        
        for scenario in error_scenarios:
            print(f"\nScenario: {scenario['name']}")
            print(f"Error type: {scenario['error_type']}")
            print(f"Message: {scenario['message']}")
            
            if scenario['error_type'] == 'quality_warning':
                print("Recommended actions:")
                print("  - Try image preprocessing")
                print("  - Lower confidence threshold")
                print("  - Use different OCR engine")
                print("  - Manual review of results")
            
            elif scenario['error_type'] == 'no_results':
                print("Recommended actions:")
                print("  - Check if image contains text")
                print("  - Try very low confidence threshold (0.1)")
                print("  - Enable preprocessing")
                print("  - Verify image is not upside down")
            
            elif scenario['error_type'] == 'timeout':
                print("Recommended actions:")
                print("  - Reduce image size")
                print("  - Use faster OCR engine")
                print("  - Increase timeout limit")
                print("  - Process image in smaller sections")
            
            elif scenario['error_type'] == 'validation_error':
                print("Recommended actions:")
                print("  - Convert to supported format (JPEG, PNG)")
                print("  - Check file is not corrupted")
                print("  - Verify file permissions")
        
        # Validation examples
        print("\nResult Validation Examples:")
        
        # Mock ingredient with validation issues
        suspicious_ingredients = [
            {
                'ingredient_name': 'Recipe',  # Likely not an ingredient
                'confidence': 0.15,
                'issue': 'Low confidence and likely header text'
            },
            {
                'ingredient_name': 'abc123xyz',  # Gibberish
                'confidence': 0.25,
                'issue': 'Contains non-food text pattern'
            },
            {
                'ingredient_name': 'salt',
                'quantity': 'invalid',  # Invalid quantity
                'confidence': 0.85,
                'issue': 'Good ingredient but invalid quantity format'
            }
        ]
        
        for ingredient in suspicious_ingredients:
            print(f"\nIngredient: {ingredient['ingredient_name']}")
            print(f"Issue: {ingredient['issue']}")
            print(f"Confidence: {ingredient['confidence']}")
            
            # Validation logic examples
            if ingredient['confidence'] < 0.2:
                print("Action: Filter out due to low confidence")
            elif ingredient['ingredient_name'].lower() in ['recipe', 'ingredients', 'directions']:
                print("Action: Filter out as likely header text")
            elif not any(c.isalpha() for c in ingredient['ingredient_name']):
                print("Action: Filter out as likely OCR error")
            else:
                print("Action: Accept but flag for review")
        
        print("\nValidation best practices:")
        print("- Set minimum confidence thresholds")
        print("- Filter common false positives")
        print("- Validate quantity and unit formats")
        print("- Flag unusual ingredient names for review")
    
    def example_7_performance_optimization(self):
        """Example 7: Performance optimization techniques"""
        print("\n" + "="*60)
        print("EXAMPLE 7: Performance Optimization Techniques")
        print("="*60)
        
        optimization_techniques = [
            {
                'name': 'Image Resizing',
                'description': 'Resize large images to optimal size',
                'before': {'size': '4000x3000px', 'processing_time': 25.3},
                'after': {'size': '2000x1500px', 'processing_time': 8.7},
                'improvement': '65% faster'
            },
            {
                'name': 'OCR Engine Selection',
                'description': 'Choose optimal OCR engine per image type',
                'before': {'engine': 'tesseract', 'processing_time': 12.1},
                'after': {'engine': 'paddleocr', 'processing_time': 7.8},
                'improvement': '35% faster'
            },
            {
                'name': 'Caching',
                'description': 'Cache results for repeated processing',
                'before': {'cache': 'disabled', 'processing_time': 9.2},
                'after': {'cache': 'enabled', 'processing_time': 0.3},
                'improvement': '97% faster (cache hit)'
            },
            {
                'name': 'Preprocessing',
                'description': 'Smart preprocessing based on image analysis',
                'before': {'accuracy': '72%', 'processing_time': 8.1},
                'after': {'accuracy': '89%', 'processing_time': 9.5},
                'improvement': '17% better accuracy'
            }
        ]
        
        print("Performance optimization techniques:")
        
        for technique in optimization_techniques:
            print(f"\n{technique['name']}:")
            print(f"  Description: {technique['description']}")
            print(f"  Before: {technique['before']}")
            print(f"  After: {technique['after']}")
            print(f"  Improvement: {technique['improvement']}")
        
        # Performance monitoring example
        print("\nPerformance Monitoring Example:")
        performance_metrics = {
            'processing_time': 8.5,
            'memory_usage_mb': 245,
            'cpu_usage_percent': 65,
            'cache_hit_rate': 0.35,
            'success_rate': 0.94,
            'avg_confidence': 0.78
        }
        
        for metric, value in performance_metrics.items():
            status = "✓" if self._is_metric_good(metric, value) else "⚠"
            print(f"  {status} {metric}: {value}")
        
        print("\nOptimization recommendations:")
        print("- Resize images to 1500-2500px maximum dimension")
        print("- Enable caching for production use")
        print("- Use async processing for batch operations")
        print("- Monitor memory usage for large batches")
        print("- Choose OCR engine based on image characteristics")
    
    def example_8_api_integration(self):
        """Example 8: API integration examples"""
        print("\n" + "="*60)
        print("EXAMPLE 8: API Integration Examples")
        print("="*60)
        
        # Python requests example
        print("Python requests example:")
        python_example = '''
import requests

# Process single image
with open('recipe.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'format_hint': 'cookbook',
        'confidence_threshold': 0.3,
        'language_hint': 'en'
    }
    
    response = requests.post(
        'http://localhost:8000/process',
        headers={'Authorization': 'Bearer YOUR_API_KEY'},
        files=files,
        data=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Found {len(result['ingredients'])} ingredients")
        for ing in result['ingredients']:
            print(f"- {ing['quantity']} {ing['unit']} {ing['ingredient_name']}")
    else:
        print(f"Error: {response.status_code}")
'''
        print(python_example)
        
        # JavaScript fetch example
        print("\nJavaScript fetch example:")
        js_example = '''
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('format_hint', 'digital');
formData.append('confidence_threshold', '0.4');

fetch('http://localhost:8000/process', {
    method: 'POST',
    headers: {
        'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log(`Found ${data.ingredients.length} ingredients`);
    data.ingredients.forEach(ing => {
        console.log(`- ${ing.quantity} ${ing.unit} ${ing.ingredient_name}`);
    });
})
.catch(error => console.error('Error:', error));
'''
        print(js_example)
        
        # cURL example
        print("\ncURL example:")
        curl_example = '''
curl -X POST "http://localhost:8000/process" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -F "file=@recipe.jpg" \\
  -F "format_hint=handwritten" \\
  -F "confidence_threshold=0.25" \\
  -F "enable_preprocessing=true"
'''
        print(curl_example)
        
        # Batch processing API example
        print("\nBatch processing API example:")
        batch_example = '''
# Start batch job
files = [('files', open('recipe1.jpg', 'rb')),
         ('files', open('recipe2.jpg', 'rb')),
         ('files', open('recipe3.jpg', 'rb'))]

response = requests.post(
    'http://localhost:8000/batch',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    files=files,
    data={'job_name': 'cookbook_batch', 'max_concurrent': 3}
)

batch_id = response.json()['batch_id']

# Check batch status
status_response = requests.get(
    f'http://localhost:8000/batch/{batch_id}',
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)

print(f"Batch status: {status_response.json()['status']}")
'''
        print(batch_example)
        
        print("\nAPI integration best practices:")
        print("- Always include proper authentication headers")
        print("- Handle rate limiting (429 responses)")
        print("- Implement retry logic with exponential backoff")
        print("- Validate file size and format before upload")
        print("- Use batch endpoints for multiple images")
        print("- Monitor API usage and quotas")
    
    def _display_results(self, result: Dict[str, Any]):
        """Display processing results in a formatted way"""
        print(f"\nProcessing Results for: {result['filename']}")
        print("-" * 40)
        print(f"Format: {result.get('format_type', 'N/A')}")
        print(f"Quality Score: {result.get('quality_score', 0):.2f}")
        print(f"Confidence Score: {result.get('confidence_score', 0):.2f}")
        print(f"Processing Time: {result.get('processing_time', 0):.1f} seconds")
        print(f"Language: {result.get('language', 'N/A')}")
        
        if 'ingredients' in result and result['ingredients']:
            print(f"\nIngredients Found ({len(result['ingredients'])}):")
            for i, ingredient in enumerate(result['ingredients'], 1):
                quantity = ingredient.get('quantity', '')
                unit = ingredient.get('unit', '')
                name = ingredient['ingredient_name']
                prep = ingredient.get('preparation', '')
                confidence = ingredient.get('confidence', 0)
                
                line = f"{i:2d}. "
                if quantity:
                    line += f"{quantity} "
                if unit:
                    line += f"{unit} "
                line += name
                if prep:
                    line += f" ({prep})"
                line += f" [conf: {confidence:.2f}]"
                
                print(line)
        else:
            print("\nNo ingredients found")
    
    def _save_example_result(self, result: Dict[str, Any], filename: str):
        """Save example result to JSON file"""
        output_file = self.results_dir / f"{filename}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {output_file}")
    
    def _is_metric_good(self, metric: str, value: float) -> bool:
        """Determine if a performance metric is good"""
        thresholds = {
            'processing_time': 15.0,  # seconds
            'memory_usage_mb': 500.0,  # MB
            'cpu_usage_percent': 80.0,  # percent
            'cache_hit_rate': 0.3,  # ratio
            'success_rate': 0.9,  # ratio
            'avg_confidence': 0.5  # ratio
        }
        
        if metric in ['processing_time', 'memory_usage_mb', 'cpu_usage_percent']:
            return value <= thresholds[metric]
        else:
            return value >= thresholds[metric]
    
    def run_all_examples(self):
        """Run all examples in sequence"""
        print("Running All Recipe Processing Examples")
        print("="*60)
        
        examples = [
            self.example_1_basic_cookbook_processing,
            self.example_2_handwritten_recipe_card,
            self.example_3_digital_recipe_screenshot,
            self.example_4_multilingual_processing,
            self.example_5_batch_processing,
            self.example_6_error_handling_and_validation,
            self.example_7_performance_optimization,
            self.example_8_api_integration
        ]
        
        for example_func in examples:
            try:
                example_func()
                time.sleep(1)  # Brief pause between examples
            except Exception as e:
                print(f"Error in {example_func.__name__}: {e}")
        
        print("\n" + "="*60)
        print("All examples completed!")
        print(f"Results saved to: {self.results_dir}")
        print("="*60)

def create_sample_images():
    """Create sample recipe images for testing"""
    examples_dir = Path(__file__).parent
    sample_images_dir = examples_dir / "sample_images"
    sample_images_dir.mkdir(exist_ok=True)
    
    # Create simple text images for demonstration
    sample_recipes = [
        {
            'filename': 'cookbook_sample.jpg',
            'title': 'Chocolate Chip Cookies',
            'ingredients': [
                '2 cups all-purpose flour',
                '1 cup butter, softened',
                '3/4 cup brown sugar',
                '1/2 cup white sugar',
                '2 large eggs',
                '1 tsp vanilla extract',
                '1 tsp baking soda',
                '1/2 tsp salt',
                '2 cups chocolate chips'
            ]
        },
        {
            'filename': 'handwritten_sample.jpg',
            'title': "Grandma's Apple Pie",
            'ingredients': [
                '6 apples, sliced',
                '1 c sugar',
                '2 tbsp flour',
                '1 tsp cinnamon',
                '1/4 tsp nutmeg',
                'pie crust (2)'
            ]
        }
    ]
    
    for recipe in sample_recipes:
        # Create a simple image with text
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a better font if available
            font_title = ImageFont.truetype("arial.ttf", 24)
            font_text = ImageFont.truetype("arial.ttf", 18)
        except:
            # Fall back to default font
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()
        
        # Draw title
        draw.text((50, 50), recipe['title'], fill='black', font=font_title)
        draw.text((50, 100), 'Ingredients:', fill='black', font=font_title)
        
        # Draw ingredients
        y_pos = 140
        for ingredient in recipe['ingredients']:
            draw.text((70, y_pos), f"• {ingredient}", fill='black', font=font_text)
            y_pos += 30
        
        # Save image
        img_path = sample_images_dir / recipe['filename']
        img.save(img_path, 'JPEG', quality=90)
        print(f"Created sample image: {img_path}")
    
    return sample_images_dir

if __name__ == "__main__":
    # Create sample images for testing
    sample_dir = create_sample_images()
    print(f"Sample images created in: {sample_dir}")
    
    # Run all examples
    examples = RecipeProcessingExamples()
    examples.run_all_examples()
    
    print("\nTo run individual examples:")
    print("python comprehensive_usage_examples.py")
    print("\nOr import and run specific examples:")
    print("from comprehensive_usage_examples import RecipeProcessingExamples")
    print("examples = RecipeProcessingExamples()")
    print("examples.example_1_basic_cookbook_processing()")