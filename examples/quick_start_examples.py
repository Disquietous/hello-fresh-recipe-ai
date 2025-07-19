#!/usr/bin/env python3
"""
Quick Start Examples for Recipe Processing API
Simple, ready-to-run examples for immediate use
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ingredient_pipeline import IngredientExtractionPipeline
except ImportError:
    print("Error: Cannot import IngredientExtractionPipeline")
    print("Make sure you're running from the project root directory")
    print("and that the src/ directory contains the pipeline module")
    sys.exit(1)

def quick_start_example():
    """Simplest possible example - just process an image"""
    print("🚀 Quick Start Example")
    print("=" * 40)
    
    # Initialize the pipeline
    pipeline = IngredientExtractionPipeline()
    
    # For this example, we'll show what the code looks like
    print("# Initialize the pipeline")
    print("pipeline = IngredientExtractionPipeline()")
    print()
    print("# Process an image (replace with your image path)")
    print("result = pipeline.process_image('your_recipe_image.jpg')")
    print()
    print("# Display results")
    print("for ingredient in result['ingredients']:")
    print("    print(f\"{ingredient['quantity']} {ingredient['unit']} {ingredient['ingredient_name']}\")")
    
    # Show what a typical result looks like
    print("\n📋 Typical Result:")
    print("-" * 20)
    mock_ingredients = [
        "2 cups flour",
        "1 cup sugar", 
        "3 large eggs",
        "1/2 cup butter"
    ]
    
    for ingredient in mock_ingredients:
        print(f"• {ingredient}")
    
    print(f"\n✅ Found {len(mock_ingredients)} ingredients in 5.2 seconds")

def cookbook_example():
    """Example for processing cookbook pages"""
    print("\n📚 Cookbook Page Example")
    print("=" * 40)
    
    print("# Best settings for printed cookbook pages")
    print("result = pipeline.process_image(")
    print("    'cookbook_page.jpg',")
    print("    format_hint='cookbook',")
    print("    ocr_engine='tesseract',")
    print("    confidence_threshold=0.3")
    print(")")
    
    print("\n💡 Tips for cookbook pages:")
    print("• Use 'tesseract' OCR engine for clean printed text")
    print("• Set confidence threshold around 0.3")
    print("• Expect high accuracy (85-95%)")
    print("• Works best with high-contrast, well-lit images")

def handwritten_example():
    """Example for processing handwritten recipes"""
    print("\n✍️ Handwritten Recipe Example")
    print("=" * 40)
    
    print("# Best settings for handwritten recipes")
    print("result = pipeline.process_image(")
    print("    'handwritten_recipe.jpg',")
    print("    format_hint='handwritten',")
    print("    ocr_engine='easyocr',")
    print("    confidence_threshold=0.2,")
    print("    enable_preprocessing=True")
    print(")")
    
    print("\n💡 Tips for handwritten recipes:")
    print("• Use 'easyocr' for better handwriting recognition")
    print("• Lower confidence threshold (0.2) due to handwriting variability")
    print("• Enable preprocessing for cleaner text")
    print("• Expect lower confidence scores (40-70%)")
    print("• Manual review recommended for critical recipes")

def digital_screenshot_example():
    """Example for processing digital screenshots"""
    print("\n📱 Digital Screenshot Example")
    print("=" * 40)
    
    print("# Best settings for app screenshots or digital recipes")
    print("result = pipeline.process_image(")
    print("    'recipe_screenshot.png',")
    print("    format_hint='digital',")
    print("    ocr_engine='paddleocr',")
    print("    confidence_threshold=0.4")
    print(")")
    
    print("\n💡 Tips for digital screenshots:")
    print("• Use 'paddleocr' for modern digital fonts")
    print("• Good accuracy expected (80-95%)")
    print("• Watch out for UI elements (buttons, ads)")
    print("• Consider cropping to ingredients section only")

def batch_processing_example():
    """Example for processing multiple images"""
    print("\n📦 Batch Processing Example")
    print("=" * 40)
    
    print("# Process multiple images at once")
    print("image_paths = [")
    print("    'recipe1.jpg',")
    print("    'recipe2.jpg',")
    print("    'recipe3.jpg'")
    print("]")
    print()
    print("results = []")
    print("for image_path in image_paths:")
    print("    try:")
    print("        result = pipeline.process_image(image_path)")
    print("        results.append(result)")
    print("        print(f'✅ {image_path}: {len(result[\"ingredients\"])} ingredients')")
    print("    except Exception as e:")
    print("        print(f'❌ {image_path}: Error - {e}')")
    
    print("\n💡 Tips for batch processing:")
    print("• Process images one by one for error handling")
    print("• Save results as you go for large batches")
    print("• Use appropriate settings for each image type")
    print("• Consider using async processing for better performance")

def error_handling_example():
    """Example with proper error handling"""
    print("\n🛡️ Error Handling Example")
    print("=" * 40)
    
    print("# Robust processing with error handling")
    print("def process_recipe_safely(image_path):")
    print("    try:")
    print("        # Check if file exists")
    print("        if not os.path.exists(image_path):")
    print("            return {'error': 'File not found'}")
    print()
    print("        # Process the image")
    print("        result = pipeline.process_image(image_path)")
    print()
    print("        # Validate results")
    print("        if not result.get('ingredients'):")
    print("            print('⚠️ No ingredients found - try lower confidence threshold')")
    print("            # Retry with lower threshold")
    print("            result = pipeline.process_image(image_path, confidence_threshold=0.1)")
    print()
    print("        return result")
    print()
    print("    except Exception as e:")
    print("        print(f'❌ Processing failed: {e}')")
    print("        return {'error': str(e)}")
    
    print("\n💡 Error handling tips:")
    print("• Always check if files exist before processing")
    print("• Implement retry logic with different settings")
    print("• Validate results and handle empty responses")
    print("• Log errors for debugging")

def api_usage_example():
    """Example using the REST API"""
    print("\n🌐 API Usage Example")
    print("=" * 40)
    
    print("# Using the REST API with Python requests")
    print("import requests")
    print()
    print("# Start the API server first:")
    print("# uvicorn src.api.recipe_processing_api:app --host 0.0.0.0 --port 8000")
    print()
    print("# Process an image via API")
    print("with open('recipe.jpg', 'rb') as f:")
    print("    response = requests.post(")
    print("        'http://localhost:8000/process',")
    print("        files={'file': f},")
    print("        data={")
    print("            'format_hint': 'cookbook',")
    print("            'confidence_threshold': 0.3")
    print("        }")
    print("    )")
    print()
    print("if response.status_code == 200:")
    print("    result = response.json()")
    print("    print(f'Found {len(result[\"ingredients\"])} ingredients')")
    print("else:")
    print("    print(f'API Error: {response.status_code}')")
    
    print("\n💡 API usage tips:")
    print("• Start the API server before making requests")
    print("• Check response status codes")
    print("• Handle rate limiting if processing many images")
    print("• Use batch endpoints for multiple images")

def optimization_tips():
    """Performance optimization tips"""
    print("\n⚡ Performance Optimization Tips")
    print("=" * 40)
    
    print("🖼️ Image Optimization:")
    print("• Resize large images to 1500-2500px max dimension")
    print("• Use good lighting and avoid blurry images")
    print("• Crop to ingredients section if possible")
    print("• Convert to JPEG for smaller file sizes")
    
    print("\n🔧 Processing Optimization:")
    print("• Choose the right OCR engine for your image type:")
    print("  - Tesseract: Clean printed text (cookbooks)")
    print("  - EasyOCR: Handwritten text, multiple languages")
    print("  - PaddleOCR: Digital text, mixed layouts")
    
    print("\n⚙️ Configuration Optimization:")
    print("• Lower confidence threshold for handwritten text")
    print("• Enable preprocessing for poor quality images")
    print("• Use format hints when you know the image type")
    print("• Enable caching for repeated processing")

def troubleshooting_guide():
    """Common issues and solutions"""
    print("\n🔧 Troubleshooting Guide")
    print("=" * 40)
    
    issues = [
        {
            'problem': 'No ingredients found',
            'solutions': [
                'Lower confidence threshold to 0.1',
                'Try different OCR engine',
                'Enable image preprocessing',
                'Check if image contains readable text'
            ]
        },
        {
            'problem': 'Low confidence scores',
            'solutions': [
                'Improve image quality (lighting, focus)',
                'Increase image resolution',
                'Try different OCR engine',
                'Enable preprocessing'
            ]
        },
        {
            'problem': 'Processing is slow',
            'solutions': [
                'Resize image to smaller dimensions',
                'Use faster OCR engine (tesseract)',
                'Enable caching',
                'Process in batches'
            ]
        },
        {
            'problem': 'Wrong ingredients detected',
            'solutions': [
                'Increase confidence threshold',
                'Use format hints',
                'Crop image to ingredients section',
                'Filter results manually'
            ]
        }
    ]
    
    for issue in issues:
        print(f"\n❓ Problem: {issue['problem']}")
        print("   Solutions:")
        for solution in issue['solutions']:
            print(f"   • {solution}")

def main():
    """Run all quick start examples"""
    print("🍳 Recipe Processing API - Quick Start Examples")
    print("=" * 60)
    
    examples = [
        quick_start_example,
        cookbook_example,
        handwritten_example,
        digital_screenshot_example,
        batch_processing_example,
        error_handling_example,
        api_usage_example,
        optimization_tips,
        troubleshooting_guide
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Quick Start Examples Complete!")
    print("\nNext Steps:")
    print("1. Try processing your own recipe images")
    print("2. Experiment with different OCR engines and settings")
    print("3. Check out comprehensive_usage_examples.py for advanced features")
    print("4. Read the full documentation in docs/USAGE_GUIDE.md")
    print("=" * 60)

if __name__ == "__main__":
    main()