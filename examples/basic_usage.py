#!/usr/bin/env python3
"""
HelloFresh Recipe AI - Basic Usage Examples
Demonstrates how to use the ingredient extraction pipeline.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ingredient_pipeline import IngredientExtractionPipeline
import json


def example_1_basic_usage():
    """Example 1: Basic pipeline usage with default configuration."""
    print("🔸 Example 1: Basic Pipeline Usage")
    print("-" * 50)
    
    # Initialize pipeline with default config
    pipeline = IngredientExtractionPipeline()
    
    # Process an image (you'll need to provide an actual image path)
    image_path = "examples/sample_recipe.jpg"  # Replace with actual path
    
    try:
        results = pipeline.process_image(image_path, "results/example1")
        
        print(f"✅ Processed: {image_path}")
        print(f"📊 Found {len(results['ingredients'])} ingredients:")
        
        for ing in results['ingredients']:
            confidence = ing['confidence_scores']['overall']
            print(f"   - {ing['quantity']} {ing['unit']} {ing['ingredient_name']} (confidence: {confidence:.2f})")
        
        print(f"📁 Results saved to: results/example1/")
        
    except FileNotFoundError:
        print(f"⚠️  Sample image not found: {image_path}")
        print("   Please provide a valid recipe image path")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_2_custom_config():
    """Example 2: Using custom configuration."""
    print("\n🔸 Example 2: Custom Configuration")
    print("-" * 50)
    
    # Load custom configuration
    custom_config = {
        'text_detection': {
            'model_path': 'yolov8s.pt',  # Use larger model for better accuracy
            'confidence_threshold': 0.3
        },
        'ocr': {
            'engine': 'paddleocr',  # Use PaddleOCR instead of EasyOCR
            'gpu': True
        },
        'parsing': {
            'min_confidence': 0.6,  # Higher confidence threshold
            'normalize_ingredients': True
        },
        'output': {
            'save_annotated_image': True,
            'save_cropped_regions': True,  # Save individual text regions
            'output_format': 'json'
        }
    }
    
    # Initialize with custom config
    pipeline = IngredientExtractionPipeline(custom_config)
    
    # Process multiple images
    image_paths = [
        "examples/recipe1.jpg",
        "examples/recipe2.jpg"
    ]
    
    for image_path in image_paths:
        try:
            results = pipeline.process_image(image_path, "results/example2")
            print(f"✅ Processed: {Path(image_path).name}")
            print(f"   Ingredients: {len(results['ingredients'])}")
            print(f"   Validation score: {results['validation_results']['validation_score']:.2f}")
            
        except FileNotFoundError:
            print(f"⚠️  Image not found: {image_path}")
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")


def example_3_config_file():
    """Example 3: Using configuration file."""
    print("\n🔸 Example 3: Configuration File")
    print("-" * 50)
    
    config_path = "configs/pipeline_config.json"
    
    try:
        # Load config from file
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        pipeline = IngredientExtractionPipeline(config)
        
        image_path = "examples/complex_recipe.jpg"
        results = pipeline.process_image(image_path, "results/example3")
        
        print(f"✅ Processed with config from: {config_path}")
        print(f"📊 Pipeline summary:")
        print(f"   Total regions: {results['detection_summary']['total_regions_detected']}")
        print(f"   High confidence: {results['detection_summary']['high_confidence_ingredients']}")
        print(f"   Medium confidence: {results['detection_summary']['medium_confidence_ingredients']}")
        print(f"   Low confidence: {results['detection_summary']['low_confidence_ingredients']}")
        
    except FileNotFoundError as e:
        print(f"⚠️  File not found: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_4_batch_processing():
    """Example 4: Batch processing multiple images."""
    print("\n🔸 Example 4: Batch Processing")
    print("-" * 50)
    
    # Find all images in a directory
    images_dir = Path("examples/recipe_images")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    if images_dir.exists():
        image_files = [
            f for f in images_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if image_files:
            pipeline = IngredientExtractionPipeline()
            
            all_results = []
            for image_file in image_files:
                try:
                    results = pipeline.process_image(str(image_file), "results/batch")
                    all_results.append(results)
                    print(f"✅ {image_file.name}: {len(results['ingredients'])} ingredients")
                except Exception as e:
                    print(f"❌ {image_file.name}: {e}")
            
            # Summary
            total_ingredients = sum(len(r['ingredients']) for r in all_results)
            print(f"\n📊 Batch Summary:")
            print(f"   Images processed: {len(all_results)}")
            print(f"   Total ingredients: {total_ingredients}")
            print(f"   Average per image: {total_ingredients/len(all_results):.1f}")
        else:
            print(f"⚠️  No images found in {images_dir}")
    else:
        print(f"⚠️  Directory not found: {images_dir}")


def example_5_parsing_only():
    """Example 5: Using just the ingredient parser."""
    print("\n🔸 Example 5: Ingredient Parsing Only")
    print("-" * 50)
    
    from utils.text_utils import IngredientParser
    
    parser = IngredientParser()
    
    # Sample ingredient lines
    ingredient_texts = [
        "2 cups all-purpose flour",
        "1 lb ground beef",
        "3 tbsp olive oil",
        "1/2 cup chopped onions",
        "2 cloves garlic, minced",
        "Salt and pepper to taste",
        "1 can (14 oz) diced tomatoes"
    ]
    
    print("Parsing ingredient text lines:")
    for text in ingredient_texts:
        result = parser.parse_ingredient_line(text)
        
        print(f"📝 '{text}'")
        print(f"   → Quantity: {result['amount']}")
        print(f"   → Unit: {result['unit']} ({result['unit_category']})")
        print(f"   → Ingredient: {result['ingredient_name']}")
        print(f"   → Quality: {result['parsing_quality']} (confidence: {result['confidence_score']:.2f})")
        print()


def example_6_validation():
    """Example 6: Recipe validation and quality assessment."""
    print("\n🔸 Example 6: Recipe Validation")
    print("-" * 50)
    
    from utils.text_utils import RecipeDataValidator
    
    # Sample ingredient data (as would come from pipeline)
    sample_ingredients = [
        {
            'ingredient_name': 'Flour',
            'amount': '2',
            'unit': 'cups',
            'confidence_score': 0.9,
            'parsing_quality': 'excellent'
        },
        {
            'ingredient_name': 'Salt',
            'amount': None,
            'unit': None,
            'confidence_score': 0.7,
            'parsing_quality': 'good'
        },
        {
            'ingredient_name': '',  # Invalid - empty name
            'amount': '1',
            'unit': 'cup',
            'confidence_score': 0.3,
            'parsing_quality': 'poor'
        }
    ]
    
    validator = RecipeDataValidator()
    validation_results = validator.validate_recipe(sample_ingredients)
    
    print(f"📊 Validation Results:")
    print(f"   Total ingredients: {validation_results['total_ingredients']}")
    print(f"   Valid ingredients: {validation_results['valid_ingredients']}")
    print(f"   Validation score: {validation_results['validation_score']:.2f}")
    print(f"   Overall quality: {validation_results['overall_quality']}")
    
    if validation_results['issues']:
        print(f"\n⚠️  Issues found:")
        for issue in validation_results['issues']:
            print(f"   - {issue}")
    
    if validation_results['warnings']:
        print(f"\n🔔 Warnings:")
        for warning in validation_results['warnings']:
            print(f"   - {warning}")


def main():
    """Run all examples."""
    print("🍽️  HelloFresh Recipe AI - Usage Examples")
    print("=" * 60)
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Run examples
    example_1_basic_usage()
    example_2_custom_config()
    example_3_config_file()
    example_4_batch_processing()
    example_5_parsing_only()
    example_6_validation()
    
    print("\n✨ All examples completed!")
    print("💡 Check the results/ directory for output files")
    print("📚 See README.md for more detailed usage instructions")


if __name__ == "__main__":
    main()