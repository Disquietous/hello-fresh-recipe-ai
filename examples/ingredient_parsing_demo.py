#!/usr/bin/env python3
"""
Comprehensive Ingredient Parsing Demo
Demonstrates all intelligent ingredient parsing capabilities including:
- Text format parsing for various ingredient patterns
- Unit normalization and conversion
- Database integration with USDA and Spoonacular
- Typo correction and abbreviation expansion
- Standardized JSON output
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from enhanced_ingredient_parser import EnhancedIngredientParser


def main():
    """Comprehensive ingredient parsing demonstration."""
    
    print("🍳 HelloFresh Recipe AI - Intelligent Ingredient Parsing Demo")
    print("=" * 60)
    
    # Initialize the enhanced parser
    print("\n1. Initializing Enhanced Ingredient Parser...")
    
    # Example configuration (you can add your API keys here)
    config = {
        'spoonacular_api_key': None,  # Add your Spoonacular API key
        'usda_api_key': None,         # Add your USDA API key
        'cache_enabled': True,
        'cache_ttl': 86400
    }
    
    parser = EnhancedIngredientParser(config)
    print("✅ Parser initialized successfully!")
    
    # Test ingredient examples covering various formats
    print("\n2. Testing Various Ingredient Text Formats...")
    
    test_ingredients = [
        # Standard format
        "2 cups all-purpose flour",
        "1 tsp vanilla extract",
        "3 large eggs, beaten",
        "1/2 cup melted butter",
        
        # Fractional and mixed numbers
        "1 1/2 cups sugar",
        "2/3 cup milk",
        "1/4 teaspoon salt",
        "3/4 pound ground beef",
        
        # Ranges and approximations
        "2-3 cloves garlic, minced",
        "about 1 lb chicken breast",
        "1 to 2 tablespoons olive oil",
        
        # Complex descriptions with preparations
        "1 medium onion, diced",
        "2 tomatoes, peeled and chopped",
        "1 cup fresh spinach, washed and stemmed",
        "8 oz cream cheese, softened to room temperature",
        
        # Brand names and package sizes
        "1 package (8 oz) Philadelphia cream cheese",
        "2 cans (14.5 oz each) diced tomatoes",
        "1 bottle (750ml) white wine",
        
        # Abbreviations and shorthand
        "2 tbsp olive oil",
        "1 tsp baking powder",
        "1 lb ground beef",
        "8 oz mozzarella cheese",
        
        # Typos and common errors
        "2 cups floru",           # flour
        "1 tsp vanilia extract",  # vanilla
        "3 large egs",            # eggs
        "1/2 cup butr",           # butter
        "2 tbsp oliv oil",        # olive oil
        
        # Unusual formats
        "Salt and pepper to taste",
        "A pinch of red pepper flakes",
        "Handful of fresh herbs",
        "Some grated parmesan cheese",
        
        # International ingredients
        "100g plain flour",
        "250ml whole milk",
        "2 cloves garlic",
        "1 kg chicken thighs"
    ]
    
    print(f"📝 Testing {len(test_ingredients)} ingredient examples...")
    
    # Parse all ingredients
    results = parser.parse_ingredients_batch(test_ingredients)
    
    # Display results
    print("\n3. Parsing Results with Analysis...")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i:2d}. Original: {result.original_text}")
        print(f"    Standardized: {result.standardized_format}")
        print(f"    Confidence: {result.confidence:.3f}")
        
        # Show components
        components = []
        if result.quantity:
            components.append(f"Qty: {result.quantity}")
        if result.unit:
            components.append(f"Unit: {result.unit}")
        if result.ingredient_name:
            components.append(f"Ingredient: {result.ingredient_name}")
        if result.preparation:
            components.append(f"Prep: {result.preparation}")
        
        if components:
            print(f"    Components: {' | '.join(components)}")
        
        # Show normalization
        if result.normalized_quantity and result.normalized_unit:
            print(f"    Normalized: {result.normalized_quantity} {result.normalized_unit}")
        
        # Show corrections and expansions
        if result.typo_corrections:
            print(f"    🔧 Typo corrections: {', '.join(result.typo_corrections)}")
        
        if result.abbreviation_expansions:
            print(f"    📝 Abbreviation expansions: {', '.join(result.abbreviation_expansions)}")
        
        # Show database information
        if result.database_match:
            print(f"    🗃️  Database match: {result.database_match['name']} ({result.database_match['source']})")
        
        # Show nutritional info
        if result.nutritional_info:
            calories = result.nutritional_info.get('calories')
            protein = result.nutritional_info.get('protein_g')
            if calories:
                print(f"    🍎 Nutrition: {calories:.1f} calories", end="")
                if protein:
                    print(f", {protein:.1f}g protein", end="")
                print()
        
        # Show alternatives
        if result.alternatives:
            print(f"    💡 Alternatives: {', '.join(result.alternatives[:3])}")
    
    # Show parsing statistics
    print("\n4. Parsing Statistics and Analysis...")
    print("-" * 60)
    
    stats = parser.get_parsing_statistics(results)
    
    print(f"📊 Total ingredients processed: {stats['total_ingredients']}")
    print(f"✅ Successfully parsed: {stats['success_rates']['successful_parsing']:.1%}")
    print(f"🎯 With database matches: {stats['success_rates']['with_database_match']:.1%}")
    print(f"🍎 With nutritional info: {stats['success_rates']['with_nutritional_info']:.1%}")
    print(f"📈 Average confidence: {stats['confidence_distribution']['average_confidence']:.3f}")
    print(f"🔧 Typo corrections made: {stats['corrections_and_expansions']['typo_corrections']}")
    print(f"📝 Abbreviations expanded: {stats['corrections_and_expansions']['abbreviation_expansions']}")
    
    print(f"\n🎯 Confidence Distribution:")
    print(f"   High (≥0.8): {stats['confidence_distribution']['high_confidence']}")
    print(f"   Medium (0.5-0.8): {stats['confidence_distribution']['medium_confidence']}")
    print(f"   Low (<0.5): {stats['confidence_distribution']['low_confidence']}")
    
    if stats['parsing_methods']:
        print(f"\n⚙️  Parsing Methods Used:")
        for method, count in stats['parsing_methods'].items():
            print(f"   {method}: {count}")
    
    if stats['food_categories']:
        print(f"\n🥘 Food Categories Found:")
        for category, count in sorted(stats['food_categories'].items()):
            print(f"   {category}: {count}")
    
    # Export to standardized JSON
    print("\n5. Standardized JSON Output...")
    print("-" * 60)
    
    json_output = parser.export_to_standardized_json(results, include_metadata=True)
    
    # Save to file
    output_file = Path(__file__).parent / "parsed_ingredients.json"
    with open(output_file, 'w') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Full results saved to: {output_file}")
    
    # Show a sample of the JSON structure
    print(f"\n📋 Sample JSON Structure:")
    sample_ingredient = json_output['ingredients'][0]
    print(json.dumps(sample_ingredient, indent=2, ensure_ascii=False))
    
    print(f"\n📊 JSON Summary:")
    summary = json_output['summary']
    print(f"   Total ingredients: {summary['total_ingredients']}")
    print(f"   Success rate: {summary['parsing_success_rate']:.1%}")
    print(f"   Average confidence: {summary['average_confidence']}")
    print(f"   Methods used: {', '.join(summary['parsing_methods_used'])}")
    
    # Demonstrate specific capabilities
    print("\n6. Demonstrating Specific Capabilities...")
    print("-" * 60)
    
    # Test typo correction
    print("\n🔧 Typo Correction Examples:")
    typo_examples = [
        "2 cups floru",
        "1 tsp vanilia",
        "3 large egs",
        "1 lb chiken breast"
    ]
    
    for example in typo_examples:
        result = parser.parse_ingredient(example)
        print(f"   '{example}' → '{result.standardized_format}'")
        if result.typo_corrections:
            print(f"     Corrections: {', '.join(result.typo_corrections)}")
    
    # Test abbreviation expansion
    print("\n📝 Abbreviation Expansion Examples:")
    abbrev_examples = [
        "2 tbsp olive oil",
        "1 tsp baking powder",
        "1 lb ground beef",
        "8 oz cream cheese"
    ]
    
    for example in abbrev_examples:
        result = parser.parse_ingredient(example)
        print(f"   '{example}' → '{result.standardized_format}'")
        if result.abbreviation_expansions:
            print(f"     Expansions: {', '.join(result.abbreviation_expansions)}")
    
    # Test normalization
    print("\n⚖️  Unit Normalization Examples:")
    unit_examples = [
        "2 cups flour",
        "1 lb butter",
        "500ml milk",
        "8 oz cheese"
    ]
    
    for example in unit_examples:
        result = parser.parse_ingredient(example)
        if result.normalized_quantity and result.normalized_unit:
            print(f"   '{example}' → {result.normalized_quantity} {result.normalized_unit}")
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 60)
    print("\nThis demo showcased:")
    print("✅ Parsing various ingredient text formats")
    print("✅ Typo correction and abbreviation expansion")
    print("✅ Unit normalization and conversion")
    print("✅ Database integration for ingredient recognition")
    print("✅ Nutritional information calculation")
    print("✅ Standardized JSON output")
    print("✅ Comprehensive parsing statistics")
    
    if not config.get('spoonacular_api_key') or not config.get('usda_api_key'):
        print("\n💡 Note: To get full database integration and nutritional information,")
        print("   add your API keys to the config dictionary in this script.")
    
    print(f"\n📁 Check the generated file: {output_file}")
    

if __name__ == "__main__":
    main()