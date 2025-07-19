#!/usr/bin/env python3
"""
Enhanced Intelligent Ingredient Parser
Advanced NLP pipeline combining all parsing capabilities with typo correction,
abbreviation handling, and standardized JSON output.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import our existing modules
from ingredient_parser import IngredientParser, ParsedIngredient
from intelligent_ingredient_parser import IntelligentIngredientParser
from ingredient_normalizer import IngredientNormalizer, NormalizationResult
from food_database_integration import FoodDatabaseIntegration, IngredientMatch

try:
    from textdistance import levenshtein
    TEXTDISTANCE_AVAILABLE = True
except ImportError:
    TEXTDISTANCE_AVAILABLE = False

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False


@dataclass
class EnhancedIngredient:
    """Enhanced ingredient with complete parsing, normalization, and database information."""
    # Original input
    original_text: str
    
    # Parsed components
    quantity: Optional[str]
    unit: Optional[str]
    ingredient_name: str
    preparation: Optional[str]
    modifier: Optional[str]
    brand: Optional[str]
    
    # Normalized values
    normalized_quantity: Optional[float]
    normalized_unit: Optional[str]
    normalized_ingredient: str
    standardized_format: str
    
    # Database information
    database_match: Optional[Dict[str, Any]]
    nutritional_info: Optional[Dict[str, Any]]
    food_category: Optional[str]
    
    # Parsing metadata
    parsing_method: str
    confidence: float
    alternatives: List[str]
    typo_corrections: List[str]
    abbreviation_expansions: List[str]
    
    # Additional metadata
    timestamp: str
    processing_notes: List[str]


class EnhancedIngredientParser:
    """Enhanced ingredient parser with comprehensive NLP capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced ingredient parser.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize component parsers
        self.basic_parser = IngredientParser()
        self.intelligent_parser = IntelligentIngredientParser(config)
        self.normalizer = IngredientNormalizer(config)
        self.database_integration = FoodDatabaseIntegration(config)
        
        # Initialize spell checker
        self.spell_checker = None
        if SPELLCHECKER_AVAILABLE:
            self.spell_checker = SpellChecker()
            # Add food-specific words to dictionary
            self._add_food_words_to_dictionary()
        
        # Load abbreviation mappings
        self.abbreviations = self._load_abbreviations()
        
        # Load typo corrections
        self.typo_corrections = self._load_typo_corrections()
        
        self.logger.info("Initialized EnhancedIngredientParser")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for enhanced parser."""
        logger = logging.getLogger('enhanced_ingredient_parser')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _add_food_words_to_dictionary(self):
        """Add food-specific words to spell checker dictionary."""
        if not self.spell_checker:
            return
        
        food_words = [
            # Common ingredients
            'flour', 'sugar', 'butter', 'milk', 'eggs', 'vanilla', 'baking', 'powder',
            'salt', 'pepper', 'onion', 'garlic', 'tomato', 'cheese', 'chicken', 'beef',
            'pork', 'fish', 'olive', 'vegetable', 'coconut', 'almond', 'walnut',
            
            # Units and measurements
            'tablespoon', 'teaspoon', 'ounce', 'pound', 'gram', 'kilogram', 'liter',
            'milliliter', 'cup', 'pint', 'quart', 'gallon', 'slice', 'clove', 'bunch',
            'pinch', 'dash', 'splash',
            
            # Cooking terms
            'chopped', 'diced', 'sliced', 'minced', 'grated', 'shredded', 'beaten',
            'whipped', 'melted', 'softened', 'room', 'temperature', 'fresh', 'frozen',
            'canned', 'dried', 'ground', 'whole', 'halved', 'quartered', 'crushed',
            
            # Food categories
            'organic', 'free-range', 'grass-fed', 'wild-caught', 'extra-virgin',
            'unsalted', 'salted', 'low-fat', 'non-fat', 'whole', 'skim'
        ]
        
        # Add words to spell checker
        for word in food_words:
            self.spell_checker.word_frequency.load_words([word])
    
    def _load_abbreviations(self) -> Dict[str, str]:
        """Load abbreviation mappings."""
        return {
            # Units
            'c': 'cup',
            'C': 'cup',
            'tbsp': 'tablespoon',
            'tbs': 'tablespoon',
            'T': 'tablespoon',
            'tsp': 'teaspoon',
            'ts': 'teaspoon',
            't': 'teaspoon',
            'lb': 'pound',
            'lbs': 'pounds',
            'oz': 'ounce',
            'fl oz': 'fluid ounce',
            'pt': 'pint',
            'qt': 'quart',
            'gal': 'gallon',
            'l': 'liter',
            'ml': 'milliliter',
            'g': 'gram',
            'kg': 'kilogram',
            'in': 'inch',
            'ft': 'foot',
            'cm': 'centimeter',
            'mm': 'millimeter',
            'pcs': 'pieces',
            'pc': 'piece',
            'pkg': 'package',
            
            # Ingredients
            'choc': 'chocolate',
            'chkn': 'chicken',
            'bf': 'beef',
            'tmt': 'tomato',
            'tmts': 'tomatoes',
            'ptt': 'potato',
            'ptts': 'potatoes',
            'onin': 'onion',
            'onins': 'onions',
            'garl': 'garlic',
            'carr': 'carrot',
            'carrs': 'carrots',
            'cell': 'celery',
            'chz': 'cheese',
            'butr': 'butter',
            'mlk': 'milk',
            'eg': 'egg',
            'egs': 'eggs',
            'flr': 'flour',
            'sgr': 'sugar',
            'slt': 'salt',
            'ppr': 'pepper',
            'veg': 'vegetable',
            'vegs': 'vegetables',
            'frt': 'fruit',
            'frts': 'fruits',
            
            # Cooking terms
            'chpd': 'chopped',
            'dcd': 'diced',
            'slcd': 'sliced',
            'mncd': 'minced',
            'grtd': 'grated',
            'shrd': 'shredded',
            'mltd': 'melted',
            'sftd': 'softened',
            'frsh': 'fresh',
            'frzn': 'frozen',
            'cnd': 'canned',
            'drd': 'dried',
            'grnd': 'ground',
            'whl': 'whole',
            
            # Brand abbreviations
            'pk': 'package',
            'btl': 'bottle',
            'cn': 'can',
            'jr': 'jar',
            'bx': 'box',
            'bg': 'bag'
        }
    
    def _load_typo_corrections(self) -> Dict[str, str]:
        """Load common typo corrections."""
        return {
            # Common typos
            'floru': 'flour',
            'sugra': 'sugar',
            'suger': 'sugar',
            'sugur': 'sugar',
            'buttr': 'butter',
            'buter': 'butter',
            'buther': 'butter',
            'onoin': 'onion',
            'onuon': 'onion',
            'galic': 'garlic',
            'garlc': 'garlic',
            'garlick': 'garlic',
            'tomato': 'tomato',
            'tomatoe': 'tomato',
            'potatoe': 'potato',
            'potaot': 'potato',
            'chiken': 'chicken',
            'chikn': 'chicken',
            'chicke': 'chicken',
            'vanila': 'vanilla',
            'vanilia': 'vanilla',
            'vanillia': 'vanilla',
            'chocolat': 'chocolate',
            'chocolte': 'chocolate',
            'choclate': 'chocolate',
            'cinnamon': 'cinnamon',
            'cinammon': 'cinnamon',
            'cinamon': 'cinnamon',
            'oregano': 'oregano',
            'oregeno': 'oregano',
            'organo': 'oregano',
            'paprika': 'paprika',
            'paprica': 'paprika',
            'paprikka': 'paprika',
            'parsley': 'parsley',
            'parsely': 'parsley',
            'parslye': 'parsley',
            'basil': 'basil',
            'bazil': 'basil',
            'basal': 'basil',
            'thyme': 'thyme',
            'time': 'thyme',
            'tyme': 'thyme',
            'rosemary': 'rosemary',
            'rosemarry': 'rosemary',
            'rosemery': 'rosemary',
            
            # Unit typos
            'tablspoon': 'tablespoon',
            'tablespon': 'tablespoon',
            'tablspn': 'tablespoon',
            'teaspn': 'teaspoon',
            'teaspon': 'teaspoon',
            'teaspoon': 'teaspoon',
            'ounse': 'ounce',
            'ounze': 'ounce',
            'ounces': 'ounces',
            'pund': 'pound',
            'pouds': 'pounds',
            'grms': 'grams',
            'gramms': 'grams',
            'litr': 'liter',
            'litrs': 'liters',
            'mililitr': 'milliliter',
            'mililitrs': 'milliliters'
        }
    
    def _correct_typos(self, text: str) -> Tuple[str, List[str]]:
        """
        Correct typos in text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (corrected_text, list_of_corrections)
        """
        corrections = []
        words = text.split()
        corrected_words = []
        
        for word in words:
            word_lower = word.lower()
            
            # Check predefined corrections first
            if word_lower in self.typo_corrections:
                correction = self.typo_corrections[word_lower]
                corrections.append(f"{word} -> {correction}")
                corrected_words.append(correction)
            elif self.spell_checker and word_lower not in self.spell_checker:
                # Use spell checker for unknown words
                candidates = self.spell_checker.candidates(word_lower)
                if candidates:
                    correction = min(candidates, key=lambda x: levenshtein(word_lower, x) if TEXTDISTANCE_AVAILABLE else len(x))
                    if correction != word_lower:
                        corrections.append(f"{word} -> {correction}")
                        corrected_words.append(correction)
                    else:
                        corrected_words.append(word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words), corrections
    
    def _expand_abbreviations(self, text: str) -> Tuple[str, List[str]]:
        """
        Expand abbreviations in text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (expanded_text, list_of_expansions)
        """
        expansions = []
        words = text.split()
        expanded_words = []
        
        for word in words:
            word_lower = word.lower()
            
            if word_lower in self.abbreviations:
                expansion = self.abbreviations[word_lower]
                expansions.append(f"{word} -> {expansion}")
                expanded_words.append(expansion)
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words), expansions
    
    def parse_ingredient(self, text: str) -> EnhancedIngredient:
        """
        Parse ingredient text with comprehensive analysis.
        
        Args:
            text: Raw ingredient text
            
        Returns:
            Enhanced ingredient with complete information
        """
        processing_notes = []
        
        # Step 1: Correct typos
        corrected_text, typo_corrections = self._correct_typos(text)
        if typo_corrections:
            processing_notes.append(f"Corrected typos: {', '.join(typo_corrections)}")
        
        # Step 2: Expand abbreviations
        expanded_text, abbreviation_expansions = self._expand_abbreviations(corrected_text)
        if abbreviation_expansions:
            processing_notes.append(f"Expanded abbreviations: {', '.join(abbreviation_expansions)}")
        
        # Step 3: Parse with intelligent parser
        parsed_ingredient = self.intelligent_parser.parse_ingredient_text(expanded_text)
        
        # Step 4: Normalize the parsed ingredient
        normalization_result = self.normalizer.normalize_ingredient_complete(
            parsed_ingredient.quantity,
            parsed_ingredient.unit,
            parsed_ingredient.ingredient_name
        )
        
        # Step 5: Search database for matches
        database_match = None
        nutritional_info = None
        food_category = None
        
        if parsed_ingredient.ingredient_name:
            ingredient_match = self.database_integration.search_ingredient(
                parsed_ingredient.ingredient_name
            )
            
            if ingredient_match.best_match:
                database_match = {
                    'id': ingredient_match.best_match.id,
                    'name': ingredient_match.best_match.name,
                    'description': ingredient_match.best_match.description,
                    'source': ingredient_match.best_match.source,
                    'confidence': ingredient_match.confidence
                }
                
                food_category = ingredient_match.best_match.category
                
                # Get nutritional information if we have quantity and unit
                if parsed_ingredient.quantity and parsed_ingredient.unit:
                    try:
                        quantity_float = float(parsed_ingredient.quantity)
                        nutritional_data = self.database_integration.get_nutritional_info(
                            ingredient_match.best_match,
                            quantity_float,
                            parsed_ingredient.unit
                        )
                        
                        nutritional_info = {
                            'calories': nutritional_data.calories,
                            'protein_g': nutritional_data.protein_g,
                            'carbs_g': nutritional_data.carbs_g,
                            'fat_g': nutritional_data.fat_g,
                            'fiber_g': nutritional_data.fiber_g,
                            'sugar_g': nutritional_data.sugar_g,
                            'sodium_mg': nutritional_data.sodium_mg,
                            'serving_size_g': nutritional_data.serving_size_g
                        }
                    except (ValueError, TypeError):
                        processing_notes.append("Could not calculate nutritional info due to invalid quantity")
        
        # Step 6: Generate alternatives
        alternatives = []
        if parsed_ingredient.alternatives:
            alternatives.extend(parsed_ingredient.alternatives)
        if normalization_result.alternatives:
            alternatives.extend(normalization_result.alternatives)
        
        # Remove duplicates
        alternatives = list(set(alternatives))
        
        # Step 7: Calculate overall confidence
        confidence_factors = [
            parsed_ingredient.confidence,
            normalization_result.confidence,
            ingredient_match.confidence if 'ingredient_match' in locals() else 0.5
        ]
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Step 8: Create enhanced ingredient
        enhanced_ingredient = EnhancedIngredient(
            original_text=text,
            quantity=parsed_ingredient.quantity,
            unit=parsed_ingredient.unit,
            ingredient_name=parsed_ingredient.ingredient_name,
            preparation=parsed_ingredient.preparation,
            modifier=parsed_ingredient.modifier,
            brand=parsed_ingredient.brand,
            normalized_quantity=normalization_result.normalized_measurement.value if normalization_result.normalized_measurement else None,
            normalized_unit=normalization_result.normalized_measurement.unit if normalization_result.normalized_measurement else None,
            normalized_ingredient=normalization_result.normalized_ingredient,
            standardized_format=normalization_result.standardized_format,
            database_match=database_match,
            nutritional_info=nutritional_info,
            food_category=food_category,
            parsing_method=parsed_ingredient.parsing_method,
            confidence=overall_confidence,
            alternatives=alternatives,
            typo_corrections=typo_corrections,
            abbreviation_expansions=abbreviation_expansions,
            timestamp=datetime.now().isoformat(),
            processing_notes=processing_notes
        )
        
        return enhanced_ingredient
    
    def parse_ingredients_batch(self, ingredient_texts: List[str]) -> List[EnhancedIngredient]:
        """Parse multiple ingredients in batch."""
        results = []
        
        for text in ingredient_texts:
            try:
                result = self.parse_ingredient(text)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to parse ingredient '{text}': {e}")
                # Create minimal result for failed parsing
                results.append(EnhancedIngredient(
                    original_text=text,
                    quantity=None,
                    unit=None,
                    ingredient_name=text,
                    preparation=None,
                    modifier=None,
                    brand=None,
                    normalized_quantity=None,
                    normalized_unit=None,
                    normalized_ingredient=text.lower(),
                    standardized_format=text,
                    database_match=None,
                    nutritional_info=None,
                    food_category=None,
                    parsing_method="error",
                    confidence=0.0,
                    alternatives=[],
                    typo_corrections=[],
                    abbreviation_expansions=[],
                    timestamp=datetime.now().isoformat(),
                    processing_notes=[f"Parsing failed: {str(e)}"]
                ))
        
        return results
    
    def export_to_standardized_json(self, ingredients: List[EnhancedIngredient], 
                                   include_metadata: bool = True) -> Dict[str, Any]:
        """
        Export ingredients to standardized JSON format.
        
        Args:
            ingredients: List of parsed ingredients
            include_metadata: Whether to include parsing metadata
            
        Returns:
            Standardized JSON structure
        """
        # Convert ingredients to dictionaries
        ingredient_dicts = []
        for ingredient in ingredients:
            ingredient_dict = asdict(ingredient)
            
            # Filter out None values for cleaner output
            ingredient_dict = {k: v for k, v in ingredient_dict.items() if v is not None}
            
            # Remove metadata if requested
            if not include_metadata:
                keys_to_remove = [
                    'parsing_method', 'typo_corrections', 'abbreviation_expansions',
                    'timestamp', 'processing_notes'
                ]
                for key in keys_to_remove:
                    ingredient_dict.pop(key, None)
            
            ingredient_dicts.append(ingredient_dict)
        
        # Calculate summary statistics
        total_ingredients = len(ingredients)
        successfully_parsed = len([i for i in ingredients if i.parsing_method != "error"])
        with_database_match = len([i for i in ingredients if i.database_match])
        with_nutritional_info = len([i for i in ingredients if i.nutritional_info])
        average_confidence = sum(i.confidence for i in ingredients) / total_ingredients if total_ingredients > 0 else 0
        
        # Get unique parsing methods
        parsing_methods = list(set(i.parsing_method for i in ingredients))
        
        # Get unique food categories
        food_categories = list(set(i.food_category for i in ingredients if i.food_category))
        
        # Create standardized output
        output = {
            "ingredients": ingredient_dicts,
            "summary": {
                "total_ingredients": total_ingredients,
                "successfully_parsed": successfully_parsed,
                "parsing_success_rate": successfully_parsed / total_ingredients if total_ingredients > 0 else 0,
                "with_database_match": with_database_match,
                "with_nutritional_info": with_nutritional_info,
                "average_confidence": round(average_confidence, 3),
                "parsing_methods_used": parsing_methods,
                "food_categories_found": food_categories
            }
        }
        
        if include_metadata:
            output["metadata"] = {
                "parser_version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "capabilities": {
                    "typo_correction": SPELLCHECKER_AVAILABLE,
                    "abbreviation_expansion": True,
                    "unit_normalization": True,
                    "database_integration": True,
                    "nutritional_calculation": True,
                    "nlp_parsing": True
                }
            }
        
        return output
    
    def get_parsing_statistics(self, ingredients: List[EnhancedIngredient]) -> Dict[str, Any]:
        """Get detailed parsing statistics."""
        if not ingredients:
            return {"error": "No ingredients provided"}
        
        total = len(ingredients)
        
        # Success rates
        successful_parsing = len([i for i in ingredients if i.parsing_method != "error"])
        with_quantity = len([i for i in ingredients if i.quantity])
        with_unit = len([i for i in ingredients if i.unit])
        with_preparation = len([i for i in ingredients if i.preparation])
        with_database_match = len([i for i in ingredients if i.database_match])
        with_nutritional_info = len([i for i in ingredients if i.nutritional_info])
        
        # Corrections and expansions
        typo_corrections = sum(len(i.typo_corrections) for i in ingredients)
        abbreviation_expansions = sum(len(i.abbreviation_expansions) for i in ingredients)
        
        # Confidence distribution
        confidences = [i.confidence for i in ingredients]
        high_confidence = len([c for c in confidences if c >= 0.8])
        medium_confidence = len([c for c in confidences if 0.5 <= c < 0.8])
        low_confidence = len([c for c in confidences if c < 0.5])
        
        # Parsing methods
        parsing_methods = {}
        for ingredient in ingredients:
            method = ingredient.parsing_method
            parsing_methods[method] = parsing_methods.get(method, 0) + 1
        
        # Food categories
        food_categories = {}
        for ingredient in ingredients:
            if ingredient.food_category:
                category = ingredient.food_category
                food_categories[category] = food_categories.get(category, 0) + 1
        
        return {
            "total_ingredients": total,
            "success_rates": {
                "successful_parsing": successful_parsing / total,
                "with_quantity": with_quantity / total,
                "with_unit": with_unit / total,
                "with_preparation": with_preparation / total,
                "with_database_match": with_database_match / total,
                "with_nutritional_info": with_nutritional_info / total
            },
            "corrections_and_expansions": {
                "typo_corrections": typo_corrections,
                "abbreviation_expansions": abbreviation_expansions,
                "ingredients_with_corrections": len([i for i in ingredients if i.typo_corrections]),
                "ingredients_with_expansions": len([i for i in ingredients if i.abbreviation_expansions])
            },
            "confidence_distribution": {
                "high_confidence": high_confidence,
                "medium_confidence": medium_confidence,
                "low_confidence": low_confidence,
                "average_confidence": sum(confidences) / len(confidences)
            },
            "parsing_methods": parsing_methods,
            "food_categories": food_categories
        }


def main():
    """Main enhanced ingredient parsing script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced ingredient parsing with comprehensive analysis')
    parser.add_argument('--text', '-t', help='Single ingredient text to parse')
    parser.add_argument('--file', '-f', help='File containing ingredient texts (one per line)')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--config', '-c', help='Configuration file (JSON)')
    parser.add_argument('--format', choices=['json', 'detailed', 'simple'], default='detailed', 
                       help='Output format')
    parser.add_argument('--include-metadata', action='store_true', 
                       help='Include parsing metadata in output')
    parser.add_argument('--statistics', action='store_true', 
                       help='Show parsing statistics')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize enhanced parser
    enhanced_parser = EnhancedIngredientParser(config)
    
    # Get ingredient texts to parse
    ingredient_texts = []
    
    if args.text:
        ingredient_texts = [args.text]
    elif args.file:
        with open(args.file, 'r') as f:
            ingredient_texts = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("Enter ingredient texts (empty line to finish):")
        while True:
            line = input("> ").strip()
            if not line:
                break
            ingredient_texts.append(line)
    
    if not ingredient_texts:
        print("No ingredient texts provided.")
        return 1
    
    # Parse ingredients
    try:
        print(f"Parsing {len(ingredient_texts)} ingredient(s)...")
        results = enhanced_parser.parse_ingredients_batch(ingredient_texts)
        
        if args.format == 'json':
            # JSON output
            output_data = enhanced_parser.export_to_standardized_json(
                results, 
                include_metadata=args.include_metadata
            )
            print(json.dumps(output_data, indent=2, ensure_ascii=False))
        
        elif args.format == 'simple':
            # Simple output
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.standardized_format}")
        
        else:
            # Detailed output
            print(f"\nEnhanced Ingredient Parsing Results:")
            print(f"====================================")
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.original_text}")
                print(f"   Standardized: {result.standardized_format}")
                print(f"   Confidence: {result.confidence:.3f}")
                
                if result.typo_corrections:
                    print(f"   Typo corrections: {', '.join(result.typo_corrections)}")
                
                if result.abbreviation_expansions:
                    print(f"   Abbreviation expansions: {', '.join(result.abbreviation_expansions)}")
                
                if result.database_match:
                    print(f"   Database match: {result.database_match['name']} ({result.database_match['source']})")
                
                if result.nutritional_info:
                    calories = result.nutritional_info.get('calories')
                    if calories:
                        print(f"   Calories: {calories:.1f}")
                
                if result.alternatives:
                    print(f"   Alternatives: {', '.join(result.alternatives[:3])}")
                
                print(f"   Method: {result.parsing_method}")
        
        # Show statistics if requested
        if args.statistics:
            stats = enhanced_parser.get_parsing_statistics(results)
            print(f"\nParsing Statistics:")
            print(f"==================")
            print(f"Total ingredients: {stats['total_ingredients']}")
            print(f"Success rate: {stats['success_rates']['successful_parsing']:.1%}")
            print(f"Database matches: {stats['success_rates']['with_database_match']:.1%}")
            print(f"Nutritional info: {stats['success_rates']['with_nutritional_info']:.1%}")
            print(f"Average confidence: {stats['confidence_distribution']['average_confidence']:.3f}")
            print(f"Typo corrections: {stats['corrections_and_expansions']['typo_corrections']}")
            print(f"Abbreviation expansions: {stats['corrections_and_expansions']['abbreviation_expansions']}")
        
        # Save results if requested
        if args.output:
            output_data = enhanced_parser.export_to_standardized_json(
                results, 
                include_metadata=True
            )
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Parsing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())