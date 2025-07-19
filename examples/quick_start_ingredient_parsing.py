#!/usr/bin/env python3
"""
Quick Start Guide for Intelligent Ingredient Parsing
Simple examples to get started with the enhanced ingredient parsing system.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from enhanced_ingredient_parser import EnhancedIngredientParser


def main():
    """Quick start examples for ingredient parsing."""
    
    print("🚀 Quick Start: Intelligent Ingredient Parsing")
    print("=" * 50)
    
    # 1. Basic Setup
    print("\n1. 🔧 Basic Setup")
    print("-" * 20)
    
    # Initialize parser (add your API keys for full functionality)
    config = {
        'spoonacular_api_key': None,  # Add your key here
        'usda_api_key': None,         # Add your key here
    }
    
    parser = EnhancedIngredientParser(config)
    print("✅ Parser initialized!")
    
    # 2. Parse a single ingredient
    print("\n2. 📝 Parse Single Ingredient")
    print("-" * 30)
    
    ingredient_text = "2 cups all-purpose flour"
    result = parser.parse_ingredient(ingredient_text)
    
    print(f"Input: '{ingredient_text}'")
    print(f"Standardized: '{result.standardized_format}'")
    print(f"Quantity: {result.quantity}")
    print(f"Unit: {result.unit}")
    print(f"Ingredient: {result.ingredient_name}")
    print(f"Confidence: {result.confidence:.3f}")
    
    # 3. Parse multiple ingredients
    print("\n3. 📋 Parse Multiple Ingredients")
    print("-" * 35)
    
    ingredients = [
        "2 cups flour",
        "1 tsp vanilla extract",
        "3 large eggs, beaten",
        "1/2 cup melted butter",
        "1 tbsp olive oil"
    ]
    
    results = parser.parse_ingredients_batch(ingredients)
    
    for i, result in enumerate(results):
        print(f"{i+1}. '{result.original_text}' → '{result.standardized_format}'")
    
    # 4. Handle typos and abbreviations
    print("\n4. 🔧 Typo Correction & Abbreviation Expansion")
    print("-" * 48)
    
    typo_examples = [
        "2 tbsp oliv oil",      # abbreviation + typo
        "1 tsp vanilia",        # typo
        "3 large egs",          # typo
        "1 lb chiken breast"    # typo
    ]
    
    for example in typo_examples:
        result = parser.parse_ingredient(example)
        print(f"'{example}' → '{result.standardized_format}'")
        
        if result.typo_corrections:
            print(f"  🔧 Corrections: {', '.join(result.typo_corrections)}")
        
        if result.abbreviation_expansions:
            print(f"  📝 Expansions: {', '.join(result.abbreviation_expansions)}")
    
    # 5. Export to JSON
    print("\n5. 📄 Export to Standardized JSON")
    print("-" * 35)
    
    # Parse some example ingredients
    sample_ingredients = [
        "2 cups all-purpose flour",
        "1 tsp baking powder",
        "1/2 cup sugar",
        "1 large egg"
    ]
    
    parsed_results = parser.parse_ingredients_batch(sample_ingredients)
    json_output = parser.export_to_standardized_json(parsed_results)
    
    print("JSON structure:")
    print(f"  - Total ingredients: {json_output['summary']['total_ingredients']}")
    print(f"  - Success rate: {json_output['summary']['parsing_success_rate']:.1%}")
    print(f"  - Average confidence: {json_output['summary']['average_confidence']:.3f}")
    
    # Save example output
    output_file = Path(__file__).parent / "quick_start_results.json"
    with open(output_file, 'w') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"  - Full results saved to: {output_file}")
    
    # 6. Advanced Features Preview
    print("\n6. 🎯 Advanced Features Preview")
    print("-" * 35)
    
    # Parse a complex ingredient
    complex_ingredient = "1 package (8 oz) cream cheese, softened to room temperature"
    result = parser.parse_ingredient(complex_ingredient)
    
    print(f"Complex ingredient: '{complex_ingredient}'")
    print(f"  Standardized: '{result.standardized_format}'")
    print(f"  Quantity: {result.quantity}")
    print(f"  Unit: {result.unit}")
    print(f"  Ingredient: {result.ingredient_name}")
    print(f"  Preparation: {result.preparation}")
    print(f"  Confidence: {result.confidence:.3f}")
    
    if result.normalized_quantity and result.normalized_unit:
        print(f"  Normalized: {result.normalized_quantity} {result.normalized_unit}")
    
    if result.database_match:
        print(f"  Database match: {result.database_match['name']}")
    
    if result.nutritional_info:
        calories = result.nutritional_info.get('calories')
        if calories:
            print(f"  Calories: {calories:.1f}")
    
    # 7. Usage Tips
    print("\n7. 💡 Usage Tips")
    print("-" * 15)
    
    tips = [
        "Add API keys to config for full database integration",
        "Use batch parsing for better performance with multiple ingredients",
        "Check confidence scores to identify parsing issues",
        "Use standardized_format for consistent recipe output",
        "Export to JSON for integration with other systems"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")
    
    # 8. Next Steps
    print("\n8. 🎯 Next Steps")
    print("-" * 15)
    
    print("  • Run the full demo: python examples/ingredient_parsing_demo.py")
    print("  • Run tests: python examples/test_ingredient_parsing.py")
    print("  • Check the comprehensive documentation in the source files")
    print("  • Integrate with your recipe OCR pipeline")
    
    print("\n🎉 Quick start completed!")
    print("You're ready to use intelligent ingredient parsing! 🚀")


if __name__ == "__main__":
    main()