#!/usr/bin/env python3
"""
Ingredient text parser for extracting structured data from OCR text.
Parses quantities, units, and ingredient names from recipe text.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import logging
from fractions import Fraction
import unicodedata


@dataclass
class ParsedIngredient:
    """Structured ingredient data."""
    raw_text: str
    quantity: Optional[str] = None
    unit: Optional[str] = None
    ingredient_name: Optional[str] = None
    preparation: Optional[str] = None
    confidence: float = 0.0
    parsing_notes: List[str] = None
    
    def __post_init__(self):
        if self.parsing_notes is None:
            self.parsing_notes = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def is_valid(self) -> bool:
        """Check if parsing was successful."""
        return bool(self.ingredient_name and self.ingredient_name.strip())


class IngredientParser:
    """Parser for extracting structured ingredient data from text."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ingredient parser.
        
        Args:
            config_path: Path to parser configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration and reference data
        self._load_config(config_path)
        self._load_units_data()
        self._load_ingredients_data()
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _load_config(self, config_path: Optional[str]):
        """Load parser configuration."""
        self.config = {
            "min_confidence": 0.3,
            "max_quantity": 1000,
            "common_preparations": [
                "chopped", "diced", "minced", "sliced", "grated", "shredded",
                "melted", "softened", "room temperature", "fresh", "dried",
                "ground", "whole", "halved", "quartered", "crushed",
                "beaten", "whipped", "sifted", "toasted", "roasted"
            ],
            "ignore_words": [
                "recipe", "ingredients", "instructions", "method", "serves",
                "cooking", "time", "prep", "total", "difficulty"
            ]
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                self.config.update(custom_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
    
    def _load_units_data(self):
        """Load units reference data."""
        # Volume units
        self.volume_units = {
            # Standard units
            "cup": ["cup", "cups", "c", "c."],
            "tablespoon": ["tablespoon", "tablespoons", "tbsp", "tbsp.", "tbs", "T", "T."],
            "teaspoon": ["teaspoon", "teaspoons", "tsp", "tsp.", "t", "t."],
            "fluid_ounce": ["fl oz", "fl. oz", "fluid ounce", "fluid ounces", "fl oz."],
            "pint": ["pint", "pints", "pt", "pt."],
            "quart": ["quart", "quarts", "qt", "qt."],
            "gallon": ["gallon", "gallons", "gal", "gal."],
            "milliliter": ["ml", "mL", "milliliter", "milliliters", "millilitre", "millilitres"],
            "liter": ["l", "L", "liter", "liters", "litre", "litres"]
        }
        
        # Weight units
        self.weight_units = {
            "pound": ["pound", "pounds", "lb", "lbs", "lb.", "lbs."],
            "ounce": ["ounce", "ounces", "oz", "oz."],
            "gram": ["gram", "grams", "g", "g."],
            "kilogram": ["kilogram", "kilograms", "kg", "kg."]
        }
        
        # Count units
        self.count_units = {
            "piece": ["piece", "pieces", "pc", "pcs"],
            "item": ["item", "items"],
            "clove": ["clove", "cloves"],
            "slice": ["slice", "slices"],
            "sheet": ["sheet", "sheets"],
            "leaf": ["leaf", "leaves"],
            "sprig": ["sprig", "sprigs"],
            "head": ["head", "heads"],
            "bunch": ["bunch", "bunches"],
            "package": ["package", "packages", "pkg", "pkg."],
            "can": ["can", "cans"],
            "jar": ["jar", "jars"],
            "bottle": ["bottle", "bottles"],
            "box": ["box", "boxes"]
        }
        
        # Combine all units
        self.all_units = {}
        self.all_units.update(self.volume_units)
        self.all_units.update(self.weight_units)
        self.all_units.update(self.count_units)
        
        # Create unit lookup dictionary
        self.unit_lookup = {}
        for standard_unit, variations in self.all_units.items():
            for variation in variations:
                self.unit_lookup[variation.lower()] = standard_unit
    
    def _load_ingredients_data(self):
        """Load ingredients reference data."""
        # Common ingredients for validation and normalization
        self.common_ingredients = {
            # Flour and grains
            "flour": ["flour", "all-purpose flour", "bread flour", "cake flour", "whole wheat flour"],
            "sugar": ["sugar", "granulated sugar", "brown sugar", "powdered sugar", "confectioners sugar"],
            "salt": ["salt", "table salt", "sea salt", "kosher salt"],
            "butter": ["butter", "unsalted butter", "salted butter"],
            "eggs": ["egg", "eggs", "large eggs", "egg whites", "egg yolks"],
            "milk": ["milk", "whole milk", "skim milk", "2% milk"],
            "water": ["water", "cold water", "warm water", "hot water"],
            "oil": ["oil", "vegetable oil", "olive oil", "coconut oil", "canola oil"],
            "vanilla": ["vanilla", "vanilla extract", "pure vanilla extract"],
            "baking_powder": ["baking powder", "baking soda", "sodium bicarbonate"],
            # Add more categories as needed
        }
        
        # Create ingredient lookup
        self.ingredient_lookup = {}
        for standard_name, variations in self.common_ingredients.items():
            for variation in variations:
                self.ingredient_lookup[variation.lower()] = standard_name
    
    def _compile_patterns(self):
        """Compile regex patterns for parsing."""
        # Quantity patterns
        self.quantity_patterns = [
            # Fractions: 1/2, 3/4, 1 1/2
            r'(\d+\s+\d+/\d+|\d+/\d+)',
            # Decimals: 1.5, 2.25
            r'(\d+\.\d+)',
            # Whole numbers: 1, 2, 10
            r'(\d+)',
            # Unicode fractions: ½, ¼, ¾
            r'([½¼¾⅓⅔⅛⅜⅝⅞])',
            # Ranges: 2-3, 1 to 2
            r'(\d+\s*[-–—]\s*\d+|\d+\s+to\s+\d+)',
            # Approximate: about 2, ~3
            r'(about\s+\d+|~\d+|\d+\s*\+/-\s*\d*)'
        ]
        
        # Combined quantity pattern
        quantity_pattern = '|'.join(self.quantity_patterns)
        
        # Unit pattern - match any known unit
        unit_variations = []
        for variations in self.all_units.values():
            unit_variations.extend(variations)
        unit_pattern = '|'.join(re.escape(unit) for unit in sorted(unit_variations, key=len, reverse=True))
        
        # Main parsing pattern
        self.main_pattern = re.compile(
            rf'^\s*({quantity_pattern})?\s*({unit_pattern})?\s*(.*?)(?:\s*[,;]\s*(.*?))?$',
            re.IGNORECASE
        )
        
        # Preparation pattern (comma-separated descriptions)
        self.preparation_pattern = re.compile(
            r',\s*(' + '|'.join(self.config["common_preparations"]) + r')\b.*',
            re.IGNORECASE
        )
        
        # Unicode fraction mapping
        self.unicode_fractions = {
            '½': '1/2', '¼': '1/4', '¾': '3/4', '⅓': '1/3', '⅔': '2/3',
            '⅛': '1/8', '⅜': '3/8', '⅝': '5/8', '⅞': '7/8'
        }
    
    def parse_ingredient_line(self, text: str) -> ParsedIngredient:
        """
        Parse a single ingredient line into structured data.
        
        Args:
            text: Raw ingredient text
            
        Returns:
            Parsed ingredient data
        """
        if not text or not text.strip():
            return ParsedIngredient(raw_text=text, confidence=0.0)
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Check if this looks like an ingredient line
        if not self._is_ingredient_line(cleaned_text):
            return ParsedIngredient(
                raw_text=text,
                confidence=0.0,
                parsing_notes=["Does not appear to be an ingredient line"]
            )
        
        # Extract components
        quantity, unit, ingredient_name, preparation = self._extract_components(cleaned_text)
        
        # Calculate confidence
        confidence = self._calculate_confidence(quantity, unit, ingredient_name)
        
        # Create result
        result = ParsedIngredient(
            raw_text=text,
            quantity=quantity,
            unit=unit,
            ingredient_name=ingredient_name,
            preparation=preparation,
            confidence=confidence
        )
        
        # Add parsing notes
        self._add_parsing_notes(result, cleaned_text)
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Replace unicode fractions with ASCII equivalents
        for unicode_frac, ascii_frac in self.unicode_fractions.items():
            text = text.replace(unicode_frac, ascii_frac)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove bullet points and list markers
        text = re.sub(r'^[•·▪▫◦‣⁃]\s*', '', text)
        text = re.sub(r'^\d+\.\s*', '', text)  # Remove numbered list markers
        text = re.sub(r'^[-*]\s*', '', text)   # Remove dash/asterisk markers
        
        return text
    
    def _is_ingredient_line(self, text: str) -> bool:
        """Check if text appears to be an ingredient line."""
        text_lower = text.lower()
        
        # Check for ignore words
        for ignore_word in self.config["ignore_words"]:
            if ignore_word in text_lower:
                return False
        
        # Must contain some alphanumeric content
        if not re.search(r'[a-zA-Z]', text):
            return False
        
        # Too short to be meaningful
        if len(text.strip()) < 3:
            return False
        
        # Likely ingredient if contains quantity or unit patterns
        has_quantity = bool(re.search('|'.join(self.quantity_patterns), text, re.IGNORECASE))
        has_unit = any(unit in text_lower for variations in self.all_units.values() for unit in variations)
        
        return has_quantity or has_unit or len(text.strip()) > 5
    
    def _extract_components(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Extract quantity, unit, ingredient name, and preparation."""
        
        # Try main pattern first
        match = self.main_pattern.match(text)
        
        if match:
            quantity_raw = match.group(1)
            unit_raw = match.group(2)
            ingredient_raw = match.group(3)
            preparation_raw = match.group(4)
        else:
            # Fallback parsing
            quantity_raw, remaining = self._extract_quantity(text)
            unit_raw, remaining = self._extract_unit(remaining)
            ingredient_raw = remaining
            preparation_raw = None
        
        # Process each component
        quantity = self._normalize_quantity(quantity_raw) if quantity_raw else None
        unit = self._normalize_unit(unit_raw) if unit_raw else None
        
        # Split ingredient and preparation
        if ingredient_raw:
            ingredient_name, preparation = self._split_ingredient_preparation(ingredient_raw)
            if preparation_raw and not preparation:
                preparation = preparation_raw
        else:
            ingredient_name = None
            preparation = preparation_raw
        
        # Clean and validate
        ingredient_name = self._clean_ingredient_name(ingredient_name) if ingredient_name else None
        preparation = self._clean_preparation(preparation) if preparation else None
        
        return quantity, unit, ingredient_name, preparation
    
    def _extract_quantity(self, text: str) -> Tuple[Optional[str], str]:
        """Extract quantity from text."""
        for pattern in self.quantity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                quantity = match.group(1)
                remaining = text[:match.start()] + text[match.end():]
                return quantity.strip(), remaining.strip()
        return None, text
    
    def _extract_unit(self, text: str) -> Tuple[Optional[str], str]:
        """Extract unit from text."""
        # Try to match units at the beginning of remaining text
        for standard_unit, variations in self.all_units.items():
            for variation in sorted(variations, key=len, reverse=True):
                pattern = rf'\b{re.escape(variation)}\b'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    remaining = text[:match.start()] + text[match.end():]
                    return variation, remaining.strip()
        return None, text
    
    def _normalize_quantity(self, quantity: str) -> str:
        """Normalize quantity string."""
        if not quantity:
            return quantity
        
        quantity = quantity.strip()
        
        # Handle unicode fractions
        for unicode_frac, ascii_frac in self.unicode_fractions.items():
            quantity = quantity.replace(unicode_frac, ascii_frac)
        
        # Handle mixed numbers (e.g., "1 1/2" -> "1.5")
        mixed_match = re.match(r'(\d+)\s+(\d+)/(\d+)', quantity)
        if mixed_match:
            whole = int(mixed_match.group(1))
            numerator = int(mixed_match.group(2))
            denominator = int(mixed_match.group(3))
            decimal = whole + numerator / denominator
            return str(decimal)
        
        # Handle simple fractions
        frac_match = re.match(r'(\d+)/(\d+)', quantity)
        if frac_match:
            try:
                fraction = Fraction(int(frac_match.group(1)), int(frac_match.group(2)))
                return str(float(fraction))
            except ZeroDivisionError:
                return quantity
        
        # Handle ranges (use the midpoint)
        range_match = re.match(r'(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)', quantity)
        if range_match:
            start = float(range_match.group(1))
            end = float(range_match.group(2))
            midpoint = (start + end) / 2
            return str(midpoint)
        
        # Handle approximate quantities
        approx_match = re.match(r'(?:about\s+|~)?(\d+(?:\.\d+)?)', quantity, re.IGNORECASE)
        if approx_match:
            return approx_match.group(1)
        
        return quantity
    
    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit to standard form."""
        if not unit:
            return unit
        
        unit_lower = unit.lower().strip()
        return self.unit_lookup.get(unit_lower, unit)
    
    def _split_ingredient_preparation(self, text: str) -> Tuple[str, Optional[str]]:
        """Split ingredient name and preparation method."""
        # Look for preparation keywords
        prep_match = self.preparation_pattern.search(text)
        
        if prep_match:
            ingredient_name = text[:prep_match.start()].strip()
            preparation = text[prep_match.start():].strip(' ,')
            return ingredient_name, preparation
        
        # Check for parenthetical preparations
        paren_match = re.search(r'(.+?)\s*\(([^)]+)\)', text)
        if paren_match:
            ingredient_name = paren_match.group(1).strip()
            preparation = paren_match.group(2).strip()
            return ingredient_name, preparation
        
        return text.strip(), None
    
    def _clean_ingredient_name(self, name: str) -> str:
        """Clean and normalize ingredient name."""
        if not name:
            return name
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Remove trailing punctuation
        name = re.sub(r'[,;:]+$', '', name)
        
        # Normalize to lowercase for lookup, but preserve original case
        return name
    
    def _clean_preparation(self, preparation: str) -> str:
        """Clean preparation text."""
        if not preparation:
            return preparation
        
        # Remove leading comma/semicolon
        preparation = re.sub(r'^[,;]\s*', '', preparation)
        
        # Remove extra whitespace
        preparation = re.sub(r'\s+', ' ', preparation.strip())
        
        return preparation
    
    def _calculate_confidence(self, quantity: Optional[str], unit: Optional[str], 
                            ingredient_name: Optional[str]) -> float:
        """Calculate parsing confidence score."""
        score = 0.0
        
        # Base score for having an ingredient name
        if ingredient_name and len(ingredient_name.strip()) > 2:
            score += 0.4
        
        # Bonus for having quantity
        if quantity:
            try:
                qty_val = float(quantity)
                if 0 < qty_val <= self.config["max_quantity"]:
                    score += 0.3
                else:
                    score += 0.1  # Unusual quantity
            except ValueError:
                score += 0.1  # Non-numeric quantity
        
        # Bonus for having valid unit
        if unit and unit in self.unit_lookup.values():
            score += 0.2
        elif unit:
            score += 0.1  # Unknown unit
        
        # Bonus for recognized ingredient
        if ingredient_name:
            ingredient_lower = ingredient_name.lower()
            if any(ing in ingredient_lower for ing in self.ingredient_lookup.keys()):
                score += 0.1
        
        return min(score, 1.0)
    
    def _add_parsing_notes(self, result: ParsedIngredient, cleaned_text: str):
        """Add parsing notes to result."""
        if not result.quantity:
            result.parsing_notes.append("No quantity detected")
        
        if not result.unit:
            result.parsing_notes.append("No unit detected")
        
        if not result.ingredient_name:
            result.parsing_notes.append("No ingredient name detected")
        elif len(result.ingredient_name) < 3:
            result.parsing_notes.append("Very short ingredient name")
        
        if result.confidence < self.config["min_confidence"]:
            result.parsing_notes.append("Low confidence parse")
    
    def parse_ingredient_list(self, text_lines: List[str]) -> List[ParsedIngredient]:
        """
        Parse multiple ingredient lines.
        
        Args:
            text_lines: List of ingredient text lines
            
        Returns:
            List of parsed ingredients
        """
        results = []
        
        for line in text_lines:
            if line and line.strip():
                result = self.parse_ingredient_line(line)
                results.append(result)
        
        self.logger.info(f"Parsed {len(results)} ingredient lines")
        return results
    
    def get_parsing_statistics(self, results: List[ParsedIngredient]) -> Dict[str, Any]:
        """
        Get statistics about parsing results.
        
        Args:
            results: List of parsing results
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {"total": 0}
        
        total = len(results)
        valid = sum(1 for r in results if r.is_valid())
        with_quantity = sum(1 for r in results if r.quantity)
        with_unit = sum(1 for r in results if r.unit)
        with_preparation = sum(1 for r in results if r.preparation)
        
        avg_confidence = sum(r.confidence for r in results) / total
        
        return {
            "total": total,
            "valid": valid,
            "success_rate": valid / total,
            "with_quantity": with_quantity,
            "with_unit": with_unit,
            "with_preparation": with_preparation,
            "average_confidence": avg_confidence,
            "high_confidence": sum(1 for r in results if r.confidence > 0.7),
            "medium_confidence": sum(1 for r in results if 0.3 <= r.confidence <= 0.7),
            "low_confidence": sum(1 for r in results if r.confidence < 0.3)
        }


def main():
    """Example usage of ingredient parser."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize parser
    parser = IngredientParser()
    
    # Example ingredient lines
    example_lines = [
        "2 cups all-purpose flour",
        "1 tsp vanilla extract",
        "3 large eggs, beaten",
        "1/2 cup melted butter",
        "2-3 cloves garlic, minced",
        "1 lb ground beef",
        "Salt and pepper to taste"
    ]
    
    print("Ingredient Parser")
    print("================")
    
    print("\nExample parsing results:")
    for line in example_lines:
        result = parser.parse_ingredient_line(line)
        print(f"\nInput: {line}")
        print(f"  Quantity: {result.quantity}")
        print(f"  Unit: {result.unit}")
        print(f"  Ingredient: {result.ingredient_name}")
        print(f"  Preparation: {result.preparation}")
        print(f"  Confidence: {result.confidence:.2f}")
    
    # Parse all lines
    results = parser.parse_ingredient_list(example_lines)
    stats = parser.get_parsing_statistics(results)
    
    print(f"\nParsing Statistics:")
    print(f"  Total lines: {stats['total']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Average confidence: {stats['average_confidence']:.2f}")


if __name__ == "__main__":
    main()