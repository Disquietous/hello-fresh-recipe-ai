#!/usr/bin/env python3
"""
Basic usage examples for the Recipe OCR Pipeline.
Demonstrates how to use the complete OCR system for ingredient extraction.
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from recipe_ocr_pipeline import RecipeOCRPipeline
from output_formatter import OutputFormatter


def example_single_image():
    """Example: Process a single recipe image."""
    print("=" * 60)
    print("EXAMPLE 1: Single Image Processing")
    print("=" * 60)
    
    # Initialize pipeline with default configuration
    pipeline = RecipeOCRPipeline()
    
    # Example image path (replace with your actual image)
    image_path = "path/to/your/recipe_image.jpg"
    output_dir = "output/single_image"
    
    print(f"Processing image: {image_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Process the image
        result = pipeline.process_image(image_path, output_dir)
        
        # Display results
        print(f"\nProcessing completed in {result.processing_time:.2f} seconds")
        print(f"Text regions detected: {result.text_regions_detected}")
        print(f"Ingredients extracted: {result.ingredients_extracted}")
        
        if result.ingredients_extracted > 0:
            print("\nExtracted Ingredients:")
            for i, ingredient in enumerate(result.ingredients, 1):
                quantity = ingredient.get('quantity', '')
                unit = ingredient.get('unit', '')
                name = ingredient.get('ingredient_name', '')
                confidence = ingredient.get('confidence', 0)
                
                print(f"  {i}. {quantity} {unit} {name} (confidence: {confidence:.2f})")
        
        # Display confidence summary
        if result.confidence_summary:
            print(f"\nConfidence Summary:")
            print(f"  Average detection: {result.confidence_summary.get('avg_detection_confidence', 0):.2f}")
            print(f"  Average OCR: {result.confidence_summary.get('avg_ocr_confidence', 0):.2f}")
            print(f"  Average parsing: {result.confidence_summary.get('avg_parsing_confidence', 0):.2f}")
        
        # Show any warnings or errors
        if result.error_log:
            print(f"\nWarnings/Errors:")
            for error in result.error_log[:5]:  # Show first 5
                print(f"  - {error}")
        
        return result
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def example_custom_configuration():
    """Example: Use custom configuration for better accuracy."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 60)
    
    # Custom configuration for high accuracy
    custom_config = {
        "text_detection": {
            "confidence_threshold": 0.15,  # Lower threshold to catch more text
            "merge_overlapping": True,
            "iou_threshold": 0.3
        },
        "ocr": {
            "primary_engine": "easyocr",
            "enable_fallback": True,
            "preprocessing_variants": True,  # Try multiple preprocessing methods
            "min_confidence": 0.2
        },
        "text_cleaning": {
            "enabled": True,
            "aggressive_mode": True,  # More aggressive text cleaning
            "min_improvement_threshold": 0.05
        },
        "ingredient_parsing": {
            "min_confidence": 0.2,  # Accept lower confidence ingredients
            "validate_ingredients": True
        },
        "output": {
            "save_annotated_image": True,
            "save_region_images": True,  # Save individual text regions
            "include_debug_info": True
        }
    }
    
    # Initialize with custom config
    pipeline = RecipeOCRPipeline(custom_config)
    
    image_path = "path/to/difficult_recipe_image.jpg"
    output_dir = "output/custom_config"
    
    print("Using high-accuracy configuration:")
    print("- Lower detection threshold")
    print("- Multiple OCR preprocessing variants")
    print("- Aggressive text cleaning")
    print("- Saving debug information")
    
    try:
        result = pipeline.process_image(image_path, output_dir)
        
        print(f"\nResults with custom configuration:")
        print(f"  Regions detected: {result.text_regions_detected}")
        print(f"  Ingredients extracted: {result.ingredients_extracted}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        print(f"Error with custom configuration: {e}")
        return None


def example_batch_processing():
    """Example: Process multiple images in batch."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RecipeOCRPipeline()
    
    # List of image paths
    image_paths = [
        "path/to/recipe1.jpg",
        "path/to/recipe2.jpg",
        "path/to/recipe3.jpg"
    ]
    
    output_dir = "output/batch_processing"
    
    print(f"Processing {len(image_paths)} images in batch...")
    
    try:
        # Process batch
        results = pipeline.process_batch(image_paths, output_dir)
        
        # Display batch summary
        total_ingredients = sum(r.ingredients_extracted for r in results)
        successful_extractions = sum(1 for r in results if r.ingredients_extracted > 0)
        total_time = sum(r.processing_time for r in results)
        
        print(f"\nBatch Processing Summary:")
        print(f"  Images processed: {len(results)}")
        print(f"  Successful extractions: {successful_extractions}")
        print(f"  Success rate: {successful_extractions/len(results):.1%}")
        print(f"  Total ingredients extracted: {total_ingredients}")
        print(f"  Average ingredients per image: {total_ingredients/len(results):.1f}")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Average time per image: {total_time/len(results):.2f}s")
        
        return results
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return []


def example_output_formatting():
    """Example: Format results in different output formats."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Output Formatting")
    print("=" * 60)
    
    # Process an image first
    pipeline = RecipeOCRPipeline()
    result = pipeline.process_image("path/to/recipe_image.jpg")
    
    if not result or result.ingredients_extracted == 0:
        print("No ingredients extracted for formatting example")
        return
    
    # Initialize formatter
    formatter = OutputFormatter({
        "include_confidence_scores": True,
        "sort_by_confidence": True,
        "filter_low_confidence": True,
        "min_confidence_threshold": 0.3
    })
    
    print("Formatting results in different formats:")
    
    # Format as JSON
    json_output = formatter.format_results(result, "json")
    print(f"\n1. JSON format ({len(json_output.content)} characters)")
    
    # Format as CSV
    csv_output = formatter.format_results(result, "csv")
    print(f"2. CSV format ({csv_output.metadata['rows']} rows)")
    
    # Format as human-readable text
    text_output = formatter.format_results(result, "txt")
    print(f"3. Text format ({text_output.metadata['lines']} lines)")
    
    # Format as XML
    xml_output = formatter.format_results(result, "xml")
    print(f"4. XML format ({xml_output.metadata['elements']} elements)")
    
    # Save outputs
    output_dir = Path("output/formatted_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    formatter.save_formatted_output(json_output, output_dir / "results.json")
    formatter.save_formatted_output(csv_output, output_dir / "results.csv")
    formatter.save_formatted_output(text_output, output_dir / "results.txt")
    formatter.save_formatted_output(xml_output, output_dir / "results.xml")
    
    print(f"\nAll formats saved to: {output_dir}")
    
    # Show sample of text output
    print(f"\nSample text output:")
    print("-" * 40)
    lines = text_output.content.split('\n')
    for line in lines[:15]:  # Show first 15 lines
        print(line)
    if len(lines) > 15:
        print("...")


def example_advanced_usage():
    """Example: Advanced usage with component access."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Advanced Component Usage")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RecipeOCRPipeline()
    
    # Access individual components for fine control
    text_detector = pipeline.text_detector
    ocr_engine = pipeline.ocr_engine
    ingredient_parser = pipeline.ingredient_parser
    text_cleaner = pipeline.text_cleaner
    
    print("Available OCR engines:", ocr_engine.get_available_engines())
    
    # Example: Process image step by step
    import cv2
    
    image_path = "path/to/recipe_image.jpg"
    image = cv2.imread(image_path)
    
    if image is not None:
        print(f"\nStep-by-step processing of {Path(image_path).name}:")
        
        # Step 1: Detect text regions
        print("1. Detecting text regions...")
        regions = text_detector.detect_text_regions(image)
        print(f"   Found {len(regions)} text regions")
        
        # Step 2: Process each region
        print("2. Processing text regions...")
        extracted_ingredients = []
        
        for i, region in enumerate(regions[:3]):  # Process first 3 regions
            # Extract region image
            region_image = text_detector.extract_text_region_image(image, region)
            
            # Apply OCR
            ocr_result = ocr_engine.extract_text(region_image, engine="easyocr")
            print(f"   Region {i+1}: '{ocr_result.text}' (confidence: {ocr_result.confidence:.2f})")
            
            # Clean text
            if text_cleaner and ocr_result.text:
                cleaned = text_cleaner.clean_text(ocr_result.text)
                if cleaned.confidence_improvement > 0.1:
                    print(f"   Cleaned: '{cleaned.cleaned_text}' (improvement: {cleaned.confidence_improvement:.2f})")
                    text_to_parse = cleaned.cleaned_text
                else:
                    text_to_parse = ocr_result.text
            else:
                text_to_parse = ocr_result.text
            
            # Parse ingredient
            if text_to_parse:
                parsed = ingredient_parser.parse_ingredient_line(text_to_parse)
                if parsed.is_valid():
                    print(f"   Parsed: {parsed.quantity} {parsed.unit} {parsed.ingredient_name}")
                    extracted_ingredients.append(parsed)
        
        print(f"\n3. Successfully extracted {len(extracted_ingredients)} ingredients")


def example_error_handling():
    """Example: Error handling and troubleshooting."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Error Handling")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RecipeOCRPipeline()
    
    # Test cases for common issues
    test_cases = [
        ("nonexistent_image.jpg", "File not found"),
        ("corrupted_image.jpg", "Corrupted image"),
        ("empty_image.jpg", "Empty/blank image"),
        ("low_quality_image.jpg", "Low quality image")
    ]
    
    print("Testing error handling with problematic inputs:")
    
    for image_path, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Image: {image_path}")
        
        try:
            result = pipeline.process_image(image_path, "output/error_tests")
            
            if result.error_log:
                print(f"Errors encountered: {len(result.error_log)}")
                for error in result.error_log[:2]:
                    print(f"  - {error}")
            else:
                print("Processed successfully (unexpected)")
                
        except Exception as e:
            print(f"Exception caught: {e}")
            print("This is expected for invalid inputs")


def main():
    """Run all examples."""
    print("Recipe OCR Pipeline - Usage Examples")
    print("=" * 60)
    
    print("\nNOTE: Replace image paths with actual paths to your recipe images")
    print("Examples will show structure even with missing images\n")
    
    # Run examples (comment out any you don't want to run)
    try:
        example_single_image()
        example_custom_configuration()
        example_batch_processing()
        example_output_formatting()
        example_advanced_usage()
        example_error_handling()
        
    except Exception as e:
        print(f"\nExample execution error: {e}")
        print("This is expected when image paths don't exist")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nTo use with real images:")
    print("1. Replace image paths with actual recipe image files")
    print("2. Install required dependencies: pip install ultralytics easyocr pytesseract")
    print("3. Run individual examples or the complete pipeline")
    print("=" * 60)


if __name__ == "__main__":
    main()