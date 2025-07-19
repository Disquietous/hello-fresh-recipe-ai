#!/usr/bin/env python3
"""
Quick start guide for Recipe OCR Pipeline.
Minimal example to get started with ingredient extraction.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from recipe_ocr_pipeline import RecipeOCRPipeline


def quick_start():
    """Minimal example to extract ingredients from a recipe image."""
    
    print("Recipe OCR Pipeline - Quick Start")
    print("=" * 40)
    
    # 1. Initialize the pipeline (uses default settings)
    print("1. Initializing OCR pipeline...")
    pipeline = RecipeOCRPipeline()
    
    # 2. Process a recipe image
    image_path = "path/to/your/recipe_image.jpg"  # Replace with your image path
    
    print(f"2. Processing image: {Path(image_path).name}")
    
    try:
        # Process the image
        result = pipeline.process_image(image_path, output_dir="quick_start_output")
        
        # 3. Display the results
        print(f"\n3. Results:")
        print(f"   Processing time: {result.processing_time:.2f} seconds")
        print(f"   Ingredients found: {result.ingredients_extracted}")
        
        # Show extracted ingredients
        if result.ingredients:
            print(f"\nExtracted Ingredients:")
            print("-" * 30)
            
            for i, ingredient in enumerate(result.ingredients, 1):
                # Build ingredient string
                parts = []
                if ingredient.get('quantity'):
                    parts.append(ingredient['quantity'])
                if ingredient.get('unit'):
                    parts.append(ingredient['unit'])
                if ingredient.get('ingredient_name'):
                    parts.append(ingredient['ingredient_name'])
                
                ingredient_text = ' '.join(parts)
                confidence = ingredient.get('confidence', 0)
                
                print(f"{i}. {ingredient_text} (confidence: {confidence:.2f})")
        else:
            print("\nNo ingredients extracted.")
            print("Try adjusting the confidence threshold or using a clearer image.")
        
        print(f"\nOutput saved to: quick_start_output/")
        
    except FileNotFoundError:
        print(f"\nImage not found: {image_path}")
        print("Please replace with the path to an actual recipe image.")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("- Make sure the image file exists")
        print("- Install required packages: pip install ultralytics easyocr")
        print("- Try with a different image")


def simple_batch_example():
    """Simple batch processing example."""
    
    print("\n" + "=" * 40)
    print("Batch Processing Example")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = RecipeOCRPipeline()
    
    # Multiple images
    image_paths = [
        "recipe1.jpg",
        "recipe2.jpg", 
        "recipe3.jpg"
    ]
    
    print(f"Processing {len(image_paths)} images...")
    
    try:
        results = pipeline.process_batch(image_paths, "batch_output")
        
        print(f"\nBatch Results:")
        total_ingredients = sum(r.ingredients_extracted for r in results)
        print(f"  Total ingredients extracted: {total_ingredients}")
        print(f"  Average per image: {total_ingredients/len(results):.1f}")
        
    except Exception as e:
        print(f"Batch processing error: {e}")


def command_line_usage():
    """Show command line usage."""
    
    print("\n" + "=" * 40)
    print("Command Line Usage")
    print("=" * 40)
    
    print("You can also use the pipeline from command line:")
    print("")
    print("# Process single image")
    print("python src/recipe_ocr_pipeline.py recipe.jpg --output-dir output/")
    print("")
    print("# Batch process directory")
    print("python src/recipe_ocr_pipeline.py recipe_images/ --batch --output-dir batch_output/")
    print("")
    print("# With custom settings")
    print("python src/recipe_ocr_pipeline.py recipe.jpg \\")
    print("    --confidence 0.2 \\")
    print("    --ocr-engine easyocr \\")
    print("    --aggressive-cleaning \\")
    print("    --save-regions")
    print("")
    print("# Use configuration file")
    print("python src/recipe_ocr_pipeline.py recipe.jpg --config configs/ocr_pipeline_config.json")


if __name__ == "__main__":
    quick_start()
    simple_batch_example()
    command_line_usage()