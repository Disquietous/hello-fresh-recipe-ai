#!/usr/bin/env python3
"""
Intelligent Ingredient Parser
Advanced NLP pipeline for parsing OCR text into structured ingredient components
with normalization, database integration, and error handling.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict
from fractions import Fraction
import unicodedata

# Add src to path
sys.path.append(str(Path(__file__).parent))

try:
    import spacy
    from spacy.matcher import Matcher
    from spacy.util import filter_spans
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from text_cleaner import TextCleaner


@dataclass
class IngredientComponent:
    """Individual component of an ingredient."""
    text: str
    component_type: str  # 'quantity', 'unit', 'ingredient', 'preparation', 'modifier'
    confidence: float
    normalized_value: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class ParsedIngredient:
    """Fully parsed ingredient with all components."""
    original_text: str
    quantity: Optional[str]
    unit: Optional[str]
    ingredient_name: str
    preparation: Optional[str]
    modifier: Optional[str]
    brand: Optional[str]
    
    # Normalization
    normalized_quantity: Optional[float]
    normalized_unit: Optional[str]
    normalized_ingredient: str
    
    # Database integration
    usda_id: Optional[str]
    spoonacular_id: Optional[str]
    food_category: Optional[str]
    
    # Parsing metadata
    components: List[IngredientComponent]
    confidence: float
    parsing_method: str
    alternatives: List[str]
    
    # Nutritional info (if available)
    nutritional_info: Optional[Dict[str, Any]] = None


@dataclass
class IngredientDatabase:
    """Ingredient database with common ingredients and their variations."""
    ingredients: Dict[str, Dict[str, Any]]
    units: Dict[str, Dict[str, Any]]
    modifiers: Dict[str, List[str]]
    preparations: Dict[str, List[str]]


class IntelligentIngredientParser:
    """Advanced ingredient parser with NLP capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize intelligent ingredient parser.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize components
        self.text_cleaner = TextCleaner()
        self.nlp = self._initialize_nlp()
        self.matcher = self._initialize_matcher()
        
        # Load ingredient database
        self.ingredient_db = self._load_ingredient_database()
        
        # Initialize external APIs
        self.spoonacular_api_key = self.config.get('spoonacular_api_key')
        self.usda_api_key = self.config.get('usda_api_key')
        
        # Parsing patterns
        self.quantity_patterns = self._create_quantity_patterns()
        self.unit_patterns = self._create_unit_patterns()
        self.ingredient_patterns = self._create_ingredient_patterns()
        self.preparation_patterns = self._create_preparation_patterns()
        
        self.logger.info("Initialized IntelligentIngredientParser")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for ingredient parser."""
        logger = logging.getLogger('intelligent_ingredient_parser')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP pipeline."""
        if not SPACY_AVAILABLE:
            self.logger.warning("spaCy not available. Using regex-based parsing.")
            return None
        
        try:
            # Try to load English model
            nlp = spacy.load("en_core_web_sm")
            self.logger.info("Loaded spaCy English model")
            return nlp
        except OSError:
            self.logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            return None
    
    def _initialize_matcher(self):
        """Initialize spaCy matcher for pattern matching."""
        if not self.nlp:
            return None
        
        matcher = Matcher(self.nlp.vocab)
        
        # Add quantity patterns
        quantity_patterns = [
            [{"LIKE_NUM": True}],
            [{"LIKE_NUM": True}, {"TEXT": "/"}, {"LIKE_NUM": True}],
            [{"LIKE_NUM": True}, {"TEXT": "-"}, {"LIKE_NUM": True}],
            [{"TEXT": {"IN": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]}}],
            [{"TEXT": {"IN": ["half", "quarter", "third", "whole"]}}],
            [{"TEXT": {"IN": ["a", "an"]}}, {"TEXT": {"IN": ["few", "couple", "handful", "pinch", "dash"]}}]
        ]
        matcher.add("QUANTITY", quantity_patterns)
        
        # Add unit patterns
        unit_patterns = [
            [{"TEXT": {"IN": ["cup", "cups", "c", "C"]}}],
            [{"TEXT": {"IN": ["tablespoon", "tablespoons", "tbsp", "tbs", "T"]}}],
            [{"TEXT": {"IN": ["teaspoon", "teaspoons", "tsp", "ts", "t"]}}],
            [{"TEXT": {"IN": ["pound", "pounds", "lb", "lbs"]}}],
            [{"TEXT": {"IN": ["ounce", "ounces", "oz"]}}],
            [{"TEXT": {"IN": ["gram", "grams", "g"]}}],
            [{"TEXT": {"IN": ["kilogram", "kilograms", "kg"]}}],
            [{"TEXT": {"IN": ["liter", "liters", "l", "L"]}}],
            [{"TEXT": {"IN": ["milliliter", "milliliters", "ml", "mL"]}}],
            [{"TEXT": {"IN": ["inch", "inches", "in"]}}],
            [{"TEXT": {"IN": ["piece", "pieces", "pcs"]}}],
            [{"TEXT": {"IN": ["slice", "slices"]}}],
            [{"TEXT": {"IN": ["clove", "cloves"]}}],
            [{"TEXT": {"IN": ["head", "heads"]}}],
            [{"TEXT": {"IN": ["bunch", "bunches"]}}],
            [{"TEXT": {"IN": ["package", "packages", "pkg"]}}],
            [{"TEXT": {"IN": ["can", "cans"]}}],
            [{"TEXT": {"IN": ["jar", "jars"]}}],
            [{"TEXT": {"IN": ["bottle", "bottles"]}}]
        ]
        matcher.add("UNIT", unit_patterns)
        
        # Add preparation patterns
        prep_patterns = [
            [{"TEXT": {"IN": ["chopped", "diced", "sliced", "minced", "grated", "shredded"]}}],
            [{"TEXT": {"IN": ["beaten", "whipped", "melted", "softened", "room", "temperature"]}}],
            [{"TEXT": {"IN": ["fresh", "frozen", "canned", "dried", "ground"]}}],
            [{"TEXT": {"IN": ["peeled", "seeded", "stemmed", "trimmed"]}}],
            [{"TEXT": {"IN": ["cooked", "raw", "uncooked"]}}],
            [{"TEXT": {"IN": ["fine", "coarse", "roughly", "finely"]}}],
            [{"LEMMA": {"IN": ["chop", "dice", "slice", "mince", "grate", "shred"]}}]
        ]
        matcher.add("PREPARATION", prep_patterns)
        
        return matcher
    
    def _load_ingredient_database(self) -> IngredientDatabase:
        """Load ingredient database with common ingredients and variations."""
        # Load from file if available
        db_path = Path(__file__).parent / "data" / "ingredient_database.json"
        
        if db_path.exists():
            try:
                with open(db_path, 'r') as f:
                    data = json.load(f)
                    return IngredientDatabase(**data)
            except Exception as e:
                self.logger.warning(f"Failed to load ingredient database: {e}")
        
        # Create default database
        ingredients = {
            "flour": {
                "variations": ["all-purpose flour", "ap flour", "plain flour", "wheat flour", "white flour"],
                "category": "baking",
                "density": 120,  # g/cup
                "usda_id": "20081"
            },
            "sugar": {
                "variations": ["granulated sugar", "white sugar", "caster sugar", "superfine sugar"],
                "category": "baking",
                "density": 200,  # g/cup
                "usda_id": "19335"
            },
            "salt": {
                "variations": ["table salt", "kosher salt", "sea salt", "fine salt"],
                "category": "seasoning",
                "density": 292,  # g/cup
                "usda_id": "02047"
            },
            "butter": {
                "variations": ["unsalted butter", "salted butter", "sweet butter"],
                "category": "dairy",
                "density": 227,  # g/cup
                "usda_id": "01001"
            },
            "milk": {
                "variations": ["whole milk", "2% milk", "skim milk", "low-fat milk"],
                "category": "dairy",
                "density": 240,  # g/cup
                "usda_id": "01077"
            },
            "egg": {
                "variations": ["large egg", "medium egg", "small egg", "eggs"],
                "category": "protein",
                "density": 50,  # g/piece
                "usda_id": "01123"
            },
            "chicken": {
                "variations": ["chicken breast", "chicken thigh", "chicken leg", "whole chicken"],
                "category": "protein",
                "density": 140,  # g/cup
                "usda_id": "05062"
            },
            "onion": {
                "variations": ["yellow onion", "white onion", "red onion", "sweet onion"],
                "category": "vegetable",
                "density": 160,  # g/cup
                "usda_id": "11282"
            },
            "garlic": {
                "variations": ["garlic clove", "fresh garlic", "garlic bulb"],
                "category": "vegetable",
                "density": 136,  # g/cup
                "usda_id": "11215"
            },
            "tomato": {
                "variations": ["fresh tomato", "roma tomato", "cherry tomato", "plum tomato"],
                "category": "vegetable",
                "density": 180,  # g/cup
                "usda_id": "11529"
            }
        }
        
        units = {
            "cup": {"type": "volume", "ml": 240, "variations": ["cups", "c", "C"]},
            "tablespoon": {"type": "volume", "ml": 15, "variations": ["tablespoons", "tbsp", "tbs", "T"]},
            "teaspoon": {"type": "volume", "ml": 5, "variations": ["teaspoons", "tsp", "ts", "t"]},
            "pound": {"type": "weight", "g": 453.592, "variations": ["pounds", "lb", "lbs"]},
            "ounce": {"type": "weight", "g": 28.3495, "variations": ["ounces", "oz"]},
            "gram": {"type": "weight", "g": 1, "variations": ["grams", "g"]},
            "kilogram": {"type": "weight", "g": 1000, "variations": ["kilograms", "kg"]},
            "liter": {"type": "volume", "ml": 1000, "variations": ["liters", "l", "L"]},
            "milliliter": {"type": "volume", "ml": 1, "variations": ["milliliters", "ml", "mL"]},
            "piece": {"type": "count", "variations": ["pieces", "pcs", "pc"]},
            "slice": {"type": "count", "variations": ["slices"]},
            "clove": {"type": "count", "variations": ["cloves"]},
            "head": {"type": "count", "variations": ["heads"]},
            "bunch": {"type": "count", "variations": ["bunches"]},
            "package": {"type": "count", "variations": ["packages", "pkg"]},
            "can": {"type": "count", "variations": ["cans"]},
            "jar": {"type": "count", "variations": ["jars"]},
            "bottle": {"type": "count", "variations": ["bottles"]}
        }
        
        modifiers = {
            "size": ["large", "medium", "small", "extra-large", "jumbo"],
            "quality": ["fresh", "frozen", "canned", "dried", "organic"],
            "fat_content": ["whole", "2%", "skim", "low-fat", "non-fat"],
            "brand": ["brand", "name", "label"]
        }
        
        preparations = {
            "cutting": ["chopped", "diced", "sliced", "minced", "grated", "shredded"],
            "cooking": ["cooked", "raw", "boiled", "steamed", "roasted", "grilled"],
            "texture": ["beaten", "whipped", "melted", "softened", "crushed"],
            "temperature": ["room temperature", "cold", "hot", "warm"],
            "processing": ["peeled", "seeded", "stemmed", "trimmed", "cleaned"]
        }
        
        return IngredientDatabase(
            ingredients=ingredients,
            units=units,
            modifiers=modifiers,
            preparations=preparations
        )
    
    def _create_quantity_patterns(self) -> List[re.Pattern]:
        """Create regex patterns for quantity matching."""
        patterns = [
            # Fractions: 1/2, 3/4, 1/3
            re.compile(r'\b(\d+)\s*/\s*(\d+)\b'),
            
            # Mixed numbers: 1 1/2, 2 3/4
            re.compile(r'\b(\d+)\s+(\d+)\s*/\s*(\d+)\b'),
            
            # Decimal numbers: 1.5, 2.25, 0.5
            re.compile(r'\b(\d*\.?\d+)\b'),
            
            # Range: 2-3, 1-2
            re.compile(r'\b(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\b'),
            
            # Written numbers: one, two, three
            re.compile(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b', re.IGNORECASE),
            
            # Fractional words: half, quarter, third
            re.compile(r'\b(half|quarter|third|whole)\b', re.IGNORECASE),
            
            # Approximate quantities: a few, couple, handful
            re.compile(r'\b(a\s+few|couple|handful|pinch|dash|splash|some)\b', re.IGNORECASE),
            
            # Article + amount: a, an
            re.compile(r'\b(a|an)\s+(?=\w)', re.IGNORECASE)
        ]
        
        return patterns
    
    def _create_unit_patterns(self) -> List[re.Pattern]:
        """Create regex patterns for unit matching."""
        # Get all unit variations
        all_units = []
        for unit_data in self.ingredient_db.units.values():
            all_units.extend(unit_data.get('variations', []))
        
        # Sort by length (longest first) to avoid partial matches
        all_units.sort(key=len, reverse=True)
        
        # Escape special regex characters
        escaped_units = [re.escape(unit) for unit in all_units]
        
        patterns = [
            # Standard units
            re.compile(r'\b(' + '|'.join(escaped_units) + r')\b', re.IGNORECASE),
            
            # Parenthetical units: (15 oz), (400g)
            re.compile(r'\((\d+(?:\.\d+)?)\s*(' + '|'.join(escaped_units) + r')\)', re.IGNORECASE),
            
            # Size specifications: 14.5 oz can, 2 lb bag
            re.compile(r'\b(\d+(?:\.\d+)?)\s*(' + '|'.join(escaped_units) + r')\s+(can|bag|box|package|jar|bottle)\b', re.IGNORECASE)
        ]
        
        return patterns
    
    def _create_ingredient_patterns(self) -> List[re.Pattern]:
        """Create regex patterns for ingredient matching."""
        patterns = [
            # Parenthetical information: (80/20), (15 oz), (room temperature)
            re.compile(r'\([^)]+\)'),
            
            # Brand names: Brand® name, Brand™ name
            re.compile(r'\b[A-Z][a-z]+(?:®|™)\s+\w+', re.IGNORECASE),
            
            # Hyphenated ingredients: all-purpose, low-fat, extra-virgin
            re.compile(r'\b\w+(?:-\w+)+\b'),
            
            # Possessive forms: baker's chocolate, shepherd's pie
            re.compile(r"\b\w+'s\s+\w+\b", re.IGNORECASE)
        ]
        
        return patterns
    
    def _create_preparation_patterns(self) -> List[re.Pattern]:
        """Create regex patterns for preparation matching."""
        # Get all preparation terms
        all_preps = []
        for prep_list in self.ingredient_db.preparations.values():
            all_preps.extend(prep_list)
        
        # Sort by length (longest first)
        all_preps.sort(key=len, reverse=True)
        
        patterns = [
            # Standard preparations
            re.compile(r'\b(' + '|'.join(all_preps) + r')\b', re.IGNORECASE),
            
            # Comma-separated preparations: chopped, diced
            re.compile(r',\s*(' + '|'.join(all_preps) + r')\b', re.IGNORECASE),
            
            # Preparation with adverbs: finely chopped, roughly diced
            re.compile(r'\b(finely|roughly|coarsely|thinly|thickly)\s+(' + '|'.join(all_preps) + r')\b', re.IGNORECASE)
        ]
        
        return patterns
    
    def parse_ingredient_text(self, text: str) -> ParsedIngredient:
        """
        Parse ingredient text into structured components.
        
        Args:
            text: Raw ingredient text
            
        Returns:
            Parsed ingredient with all components
        """
        # Clean text
        cleaned_result = self.text_cleaner.clean_text(text)
        cleaned_text = cleaned_result.cleaned_text
        
        # Initialize parsing result
        components = []
        confidence = 0.8
        
        # Try different parsing methods
        parsed_ingredient = None
        
        # Method 1: spaCy NLP parsing
        if self.nlp:
            parsed_ingredient = self._parse_with_spacy(cleaned_text, components)
            if parsed_ingredient:
                parsed_ingredient.parsing_method = "spacy"
        
        # Method 2: Regex-based parsing (fallback)
        if not parsed_ingredient:
            parsed_ingredient = self._parse_with_regex(cleaned_text, components)
            if parsed_ingredient:
                parsed_ingredient.parsing_method = "regex"
        
        # Method 3: Simple pattern matching (last resort)
        if not parsed_ingredient:
            parsed_ingredient = self._parse_with_patterns(cleaned_text, components)
            if parsed_ingredient:
                parsed_ingredient.parsing_method = "pattern"
        
        # If all methods fail, create basic ingredient
        if not parsed_ingredient:
            parsed_ingredient = ParsedIngredient(
                original_text=text,
                quantity=None,
                unit=None,
                ingredient_name=cleaned_text,
                preparation=None,
                modifier=None,
                brand=None,
                normalized_quantity=None,
                normalized_unit=None,
                normalized_ingredient=self._normalize_ingredient_name(cleaned_text),
                usda_id=None,
                spoonacular_id=None,
                food_category=None,
                components=components,
                confidence=0.5,
                parsing_method="fallback",
                alternatives=[]
            )
        
        # Post-processing
        parsed_ingredient = self._post_process_ingredient(parsed_ingredient)
        
        return parsed_ingredient
    
    def _parse_with_spacy(self, text: str, components: List[IngredientComponent]) -> Optional[ParsedIngredient]:
        """Parse ingredient using spaCy NLP."""
        if not self.nlp or not self.matcher:
            return None
        
        try:
            doc = self.nlp(text)
            matches = self.matcher(doc)
            
            # Extract entities
            quantity = None
            unit = None
            ingredient_name = None
            preparation = None
            modifier = None
            brand = None
            
            # Process matches
            match_spans = []
            for match_id, start, end in matches:
                span = doc[start:end]
                label = self.nlp.vocab.strings[match_id]
                match_spans.append((span, label))
                
                # Create component
                component = IngredientComponent(
                    text=span.text,
                    component_type=label.lower(),
                    confidence=0.8,
                    start_pos=span.start_char,
                    end_pos=span.end_char
                )
                components.append(component)
            
            # Filter overlapping spans
            spans = [span for span, _ in match_spans]
            filtered_spans = filter_spans(spans)
            
            # Extract information from spans
            for span in filtered_spans:
                label = None
                for s, l in match_spans:
                    if s == span:
                        label = l
                        break
                
                if label == "QUANTITY":
                    quantity = self._normalize_quantity(span.text)
                elif label == "UNIT":
                    unit = self._normalize_unit(span.text)
                elif label == "PREPARATION":
                    preparation = span.text.lower()
            
            # Extract ingredient name (remaining text)
            used_chars = set()
            for span in filtered_spans:
                for i in range(span.start_char, span.end_char):
                    used_chars.add(i)
            
            ingredient_chars = []
            for i, char in enumerate(text):
                if i not in used_chars:
                    ingredient_chars.append(char)
            
            ingredient_name = ''.join(ingredient_chars).strip()
            ingredient_name = re.sub(r'\s+', ' ', ingredient_name)
            
            # Clean up ingredient name
            ingredient_name = self._clean_ingredient_name(ingredient_name)
            
            if not ingredient_name:
                return None
            
            # Create parsed ingredient
            parsed_ingredient = ParsedIngredient(
                original_text=text,
                quantity=quantity,
                unit=unit,
                ingredient_name=ingredient_name,
                preparation=preparation,
                modifier=modifier,
                brand=brand,
                normalized_quantity=self._parse_quantity_value(quantity) if quantity else None,
                normalized_unit=self._get_normalized_unit(unit) if unit else None,
                normalized_ingredient=self._normalize_ingredient_name(ingredient_name),
                usda_id=None,
                spoonacular_id=None,
                food_category=None,
                components=components,
                confidence=0.8,
                parsing_method="spacy",
                alternatives=[]
            )
            
            return parsed_ingredient
            
        except Exception as e:
            self.logger.warning(f"spaCy parsing failed: {e}")
            return None
    
    def _parse_with_regex(self, text: str, components: List[IngredientComponent]) -> Optional[ParsedIngredient]:
        """Parse ingredient using regex patterns."""
        quantity = None
        unit = None
        ingredient_name = text
        preparation = None
        modifier = None
        brand = None
        
        # Extract quantity
        for pattern in self.quantity_patterns:
            match = pattern.search(text)
            if match:
                quantity = match.group(0)
                ingredient_name = ingredient_name.replace(quantity, '').strip()
                
                component = IngredientComponent(
                    text=quantity,
                    component_type="quantity",
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                components.append(component)
                break
        
        # Extract unit
        for pattern in self.unit_patterns:
            match = pattern.search(text)
            if match:
                unit = match.group(0) if match.lastindex is None else match.group(1)
                ingredient_name = ingredient_name.replace(match.group(0), '').strip()
                
                component = IngredientComponent(
                    text=unit,
                    component_type="unit",
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                components.append(component)
                break
        
        # Extract preparation
        for pattern in self.preparation_patterns:
            match = pattern.search(text)
            if match:
                preparation = match.group(0) if match.lastindex is None else match.group(1)
                ingredient_name = ingredient_name.replace(match.group(0), '').strip()
                
                component = IngredientComponent(
                    text=preparation,
                    component_type="preparation",
                    confidence=0.8,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                components.append(component)
                break
        
        # Clean up ingredient name
        ingredient_name = self._clean_ingredient_name(ingredient_name)
        
        if not ingredient_name:
            return None
        
        # Create parsed ingredient
        parsed_ingredient = ParsedIngredient(
            original_text=text,
            quantity=quantity,
            unit=unit,
            ingredient_name=ingredient_name,
            preparation=preparation,
            modifier=modifier,
            brand=brand,
            normalized_quantity=self._parse_quantity_value(quantity) if quantity else None,
            normalized_unit=self._get_normalized_unit(unit) if unit else None,
            normalized_ingredient=self._normalize_ingredient_name(ingredient_name),
            usda_id=None,
            spoonacular_id=None,
            food_category=None,
            components=components,
            confidence=0.7,
            parsing_method="regex",
            alternatives=[]
        )
        
        return parsed_ingredient
    
    def _parse_with_patterns(self, text: str, components: List[IngredientComponent]) -> Optional[ParsedIngredient]:
        """Parse ingredient using simple patterns."""
        # Simple pattern: [quantity] [unit] [ingredient] [, preparation]
        
        # Remove common punctuation
        clean_text = re.sub(r'[(),]', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        words = clean_text.split()
        
        if not words:
            return None
        
        quantity = None
        unit = None
        ingredient_parts = []
        preparation = None
        
        i = 0
        
        # Try to extract quantity (first word)
        if i < len(words) and self._is_quantity(words[i]):
            quantity = words[i]
            i += 1
        
        # Try to extract unit (next word)
        if i < len(words) and self._is_unit(words[i]):
            unit = words[i]
            i += 1
        
        # Remaining words are ingredient name
        while i < len(words):
            # Check if this looks like a preparation word
            if self._is_preparation(words[i]):
                preparation = ' '.join(words[i:])
                break
            else:
                ingredient_parts.append(words[i])
                i += 1
        
        ingredient_name = ' '.join(ingredient_parts).strip()
        
        if not ingredient_name:
            return None
        
        # Create components
        pos = 0
        if quantity:
            components.append(IngredientComponent(
                text=quantity,
                component_type="quantity",
                confidence=0.6,
                start_pos=pos,
                end_pos=pos + len(quantity)
            ))
            pos += len(quantity) + 1
        
        if unit:
            components.append(IngredientComponent(
                text=unit,
                component_type="unit",
                confidence=0.6,
                start_pos=pos,
                end_pos=pos + len(unit)
            ))
            pos += len(unit) + 1
        
        if ingredient_name:
            components.append(IngredientComponent(
                text=ingredient_name,
                component_type="ingredient",
                confidence=0.7,
                start_pos=pos,
                end_pos=pos + len(ingredient_name)
            ))
        
        # Create parsed ingredient
        parsed_ingredient = ParsedIngredient(
            original_text=text,
            quantity=quantity,
            unit=unit,
            ingredient_name=ingredient_name,
            preparation=preparation,
            modifier=None,
            brand=None,
            normalized_quantity=self._parse_quantity_value(quantity) if quantity else None,
            normalized_unit=self._get_normalized_unit(unit) if unit else None,
            normalized_ingredient=self._normalize_ingredient_name(ingredient_name),
            usda_id=None,
            spoonacular_id=None,
            food_category=None,
            components=components,
            confidence=0.6,
            parsing_method="pattern",
            alternatives=[]
        )
        
        return parsed_ingredient
    
    def _is_quantity(self, word: str) -> bool:
        """Check if word represents a quantity."""
        # Check if it's a number
        try:
            float(word)
            return True
        except ValueError:
            pass
        
        # Check if it's a fraction
        if '/' in word:
            parts = word.split('/')
            if len(parts) == 2:
                try:
                    float(parts[0])
                    float(parts[1])
                    return True
                except ValueError:
                    pass
        
        # Check if it's a written number
        written_numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                          'half', 'quarter', 'third', 'whole', 'a', 'an']
        
        return word.lower() in written_numbers
    
    def _is_unit(self, word: str) -> bool:
        """Check if word represents a unit."""
        word_lower = word.lower()
        
        for unit_data in self.ingredient_db.units.values():
            if word_lower in [v.lower() for v in unit_data.get('variations', [])]:
                return True
        
        return False
    
    def _is_preparation(self, word: str) -> bool:
        """Check if word represents a preparation method."""
        word_lower = word.lower()
        
        for prep_list in self.ingredient_db.preparations.values():
            if word_lower in [p.lower() for p in prep_list]:
                return True
        
        return False
    
    def _clean_ingredient_name(self, name: str) -> str:
        """Clean ingredient name by removing unwanted characters."""
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Remove leading/trailing punctuation
        name = re.sub(r'^[^\w]+|[^\w]+$', '', name)
        
        # Remove parenthetical information that's not part of the name
        name = re.sub(r'\([^)]*\)', '', name)
        
        # Remove extra whitespace again
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def _normalize_quantity(self, quantity_str: str) -> str:
        """Normalize quantity string."""
        if not quantity_str:
            return quantity_str
        
        # Convert written numbers to digits
        written_to_digit = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'half': '0.5', 'quarter': '0.25', 'third': '0.33', 'whole': '1',
            'a': '1', 'an': '1'
        }
        
        quantity_lower = quantity_str.lower().strip()
        if quantity_lower in written_to_digit:
            return written_to_digit[quantity_lower]
        
        return quantity_str
    
    def _normalize_unit(self, unit_str: str) -> str:
        """Normalize unit string."""
        if not unit_str:
            return unit_str
        
        unit_lower = unit_str.lower().strip()
        
        # Find the canonical unit
        for canonical_unit, unit_data in self.ingredient_db.units.items():
            variations = [v.lower() for v in unit_data.get('variations', [])]
            if unit_lower in variations:
                return canonical_unit
        
        return unit_str
    
    def _parse_quantity_value(self, quantity_str: str) -> Optional[float]:
        """Parse quantity string to float value."""
        if not quantity_str:
            return None
        
        try:
            # Handle fractions
            if '/' in quantity_str:
                # Check for mixed numbers (e.g., "1 1/2")
                parts = quantity_str.split()
                if len(parts) == 2 and '/' in parts[1]:
                    whole = float(parts[0])
                    frac = Fraction(parts[1])
                    return whole + float(frac)
                else:
                    # Simple fraction
                    return float(Fraction(quantity_str))
            
            # Handle ranges (e.g., "2-3")
            if '-' in quantity_str:
                parts = quantity_str.split('-')
                if len(parts) == 2:
                    try:
                        min_val = float(parts[0])
                        max_val = float(parts[1])
                        return (min_val + max_val) / 2  # Return average
                    except ValueError:
                        pass
            
            # Handle regular numbers
            return float(quantity_str)
            
        except (ValueError, ZeroDivisionError):
            return None
    
    def _get_normalized_unit(self, unit_str: str) -> Optional[str]:
        """Get normalized unit."""
        if not unit_str:
            return None
        
        normalized = self._normalize_unit(unit_str)
        return normalized if normalized != unit_str else unit_str
    
    def _normalize_ingredient_name(self, name: str) -> str:
        """Normalize ingredient name for database matching."""
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove common words that don't affect matching
        stop_words = ['fresh', 'frozen', 'canned', 'dried', 'ground', 'whole', 'chopped', 'sliced']
        words = normalized.split()
        filtered_words = [word for word in words if word not in stop_words]
        
        if filtered_words:
            normalized = ' '.join(filtered_words)
        
        return normalized
    
    def _post_process_ingredient(self, ingredient: ParsedIngredient) -> ParsedIngredient:
        """Post-process parsed ingredient with database lookups and validation."""
        # Database lookup
        ingredient = self._lookup_ingredient_in_database(ingredient)
        
        # External API lookup
        if self.spoonacular_api_key:
            ingredient = self._lookup_spoonacular(ingredient)
        
        if self.usda_api_key:
            ingredient = self._lookup_usda(ingredient)
        
        # Generate alternatives
        ingredient.alternatives = self._generate_alternatives(ingredient)
        
        # Final confidence adjustment
        ingredient.confidence = self._calculate_final_confidence(ingredient)
        
        return ingredient
    
    def _lookup_ingredient_in_database(self, ingredient: ParsedIngredient) -> ParsedIngredient:
        """Look up ingredient in local database."""
        normalized_name = ingredient.normalized_ingredient
        
        # Direct match
        if normalized_name in self.ingredient_db.ingredients:
            ingredient_data = self.ingredient_db.ingredients[normalized_name]
            ingredient.food_category = ingredient_data.get('category')
            ingredient.usda_id = ingredient_data.get('usda_id')
            return ingredient
        
        # Fuzzy match
        if FUZZYWUZZY_AVAILABLE:
            best_match = process.extractOne(
                normalized_name,
                self.ingredient_db.ingredients.keys(),
                score_cutoff=80
            )
            
            if best_match:
                matched_name, score = best_match
                ingredient_data = self.ingredient_db.ingredients[matched_name]
                ingredient.food_category = ingredient_data.get('category')
                ingredient.usda_id = ingredient_data.get('usda_id')
                ingredient.confidence *= (score / 100)
        
        return ingredient
    
    def _lookup_spoonacular(self, ingredient: ParsedIngredient) -> ParsedIngredient:
        """Look up ingredient in Spoonacular API."""
        if not REQUESTS_AVAILABLE or not self.spoonacular_api_key:
            return ingredient
        
        try:
            url = "https://api.spoonacular.com/food/ingredients/search"
            params = {
                'apiKey': self.spoonacular_api_key,
                'query': ingredient.normalized_ingredient,
                'number': 1
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    result = data['results'][0]
                    ingredient.spoonacular_id = str(result.get('id'))
                    
                    # Get nutritional info
                    nutrition_url = f"https://api.spoonacular.com/food/ingredients/{ingredient.spoonacular_id}/information"
                    nutrition_params = {
                        'apiKey': self.spoonacular_api_key,
                        'amount': 1,
                        'unit': 'serving'
                    }
                    
                    nutrition_response = requests.get(nutrition_url, params=nutrition_params, timeout=5)
                    if nutrition_response.status_code == 200:
                        nutrition_data = nutrition_response.json()
                        ingredient.nutritional_info = nutrition_data.get('nutrition')
        
        except Exception as e:
            self.logger.warning(f"Spoonacular API error: {e}")
        
        return ingredient
    
    def _lookup_usda(self, ingredient: ParsedIngredient) -> ParsedIngredient:
        """Look up ingredient in USDA API."""
        if not REQUESTS_AVAILABLE or not self.usda_api_key:
            return ingredient
        
        try:
            url = "https://api.nal.usda.gov/fdc/v1/foods/search"
            params = {
                'api_key': self.usda_api_key,
                'query': ingredient.normalized_ingredient,
                'pageSize': 1
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('foods'):
                    food = data['foods'][0]
                    ingredient.usda_id = str(food.get('fdcId'))
                    
                    # Extract category if available
                    if 'foodCategory' in food:
                        ingredient.food_category = food['foodCategory']
        
        except Exception as e:
            self.logger.warning(f"USDA API error: {e}")
        
        return ingredient
    
    def _generate_alternatives(self, ingredient: ParsedIngredient) -> List[str]:
        """Generate alternative names for the ingredient."""
        alternatives = []
        
        # Get variations from database
        normalized_name = ingredient.normalized_ingredient
        if normalized_name in self.ingredient_db.ingredients:
            ingredient_data = self.ingredient_db.ingredients[normalized_name]
            alternatives.extend(ingredient_data.get('variations', []))
        
        # Generate common variations
        name = ingredient.ingredient_name.lower()
        
        # Singular/plural variations
        if name.endswith('s'):
            alternatives.append(name[:-1])
        else:
            alternatives.append(name + 's')
        
        # Remove duplicates and original
        alternatives = list(set(alternatives))
        if ingredient.ingredient_name in alternatives:
            alternatives.remove(ingredient.ingredient_name)
        
        return alternatives[:5]  # Limit to 5 alternatives
    
    def _calculate_final_confidence(self, ingredient: ParsedIngredient) -> float:
        """Calculate final confidence score."""
        confidence = ingredient.confidence
        
        # Boost confidence if we found database matches
        if ingredient.usda_id or ingredient.spoonacular_id:
            confidence *= 1.1
        
        if ingredient.food_category:
            confidence *= 1.05
        
        # Reduce confidence if parsing was uncertain
        if ingredient.parsing_method == "fallback":
            confidence *= 0.7
        elif ingredient.parsing_method == "pattern":
            confidence *= 0.8
        
        return min(confidence, 1.0)
    
    def parse_ingredients_batch(self, ingredient_texts: List[str]) -> List[ParsedIngredient]:
        """Parse multiple ingredient texts in batch."""
        results = []
        
        for text in ingredient_texts:
            try:
                parsed = self.parse_ingredient_text(text)
                results.append(parsed)
            except Exception as e:
                self.logger.error(f"Failed to parse ingredient '{text}': {e}")
                # Create fallback ingredient
                results.append(ParsedIngredient(
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
                    usda_id=None,
                    spoonacular_id=None,
                    food_category=None,
                    components=[],
                    confidence=0.3,
                    parsing_method="error",
                    alternatives=[]
                ))
        
        return results
    
    def export_to_standardized_json(self, ingredients: List[ParsedIngredient]) -> Dict[str, Any]:
        """Export parsed ingredients to standardized JSON format."""
        return {
            "ingredients": [asdict(ingredient) for ingredient in ingredients],
            "parsing_summary": {
                "total_ingredients": len(ingredients),
                "successfully_parsed": len([i for i in ingredients if i.parsing_method != "error"]),
                "average_confidence": sum(i.confidence for i in ingredients) / len(ingredients) if ingredients else 0,
                "methods_used": list(set(i.parsing_method for i in ingredients)),
                "database_matches": len([i for i in ingredients if i.usda_id or i.spoonacular_id]),
                "categories_found": list(set(i.food_category for i in ingredients if i.food_category))
            },
            "metadata": {
                "parser_version": "1.0",
                "timestamp": "2024-01-01T00:00:00Z",
                "features_used": {
                    "spacy_nlp": self.nlp is not None,
                    "fuzzy_matching": FUZZYWUZZY_AVAILABLE,
                    "spoonacular_api": bool(self.spoonacular_api_key),
                    "usda_api": bool(self.usda_api_key)
                }
            }
        }


def main():
    """Main ingredient parsing script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse ingredient text')
    parser.add_argument('--text', '-t', help='Single ingredient text to parse')
    parser.add_argument('--file', '-f', help='File containing ingredient texts (one per line)')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--config', '-c', help='Configuration file')
    parser.add_argument('--spoonacular-key', help='Spoonacular API key')
    parser.add_argument('--usda-key', help='USDA API key')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line arguments
    if args.spoonacular_key:
        config['spoonacular_api_key'] = args.spoonacular_key
    if args.usda_key:
        config['usda_api_key'] = args.usda_key
    
    # Initialize parser
    parser = IntelligentIngredientParser(config)
    
    # Parse ingredients
    try:
        if args.text:
            # Single ingredient
            result = parser.parse_ingredient_text(args.text)
            results = [result]
        elif args.file:
            # Multiple ingredients from file
            with open(args.file, 'r') as f:
                ingredient_texts = [line.strip() for line in f if line.strip()]
            results = parser.parse_ingredients_batch(ingredient_texts)
        else:
            # Interactive mode
            print("Enter ingredient texts (empty line to finish):")
            ingredient_texts = []
            while True:
                line = input("> ").strip()
                if not line:
                    break
                ingredient_texts.append(line)
            
            if ingredient_texts:
                results = parser.parse_ingredients_batch(ingredient_texts)
            else:
                print("No ingredients provided.")
                return 1
        
        # Export results
        export_data = parser.export_to_standardized_json(results)
        
        # Print results
        print(f"\nParsing Results:")
        print(f"================")
        print(f"Total ingredients: {export_data['parsing_summary']['total_ingredients']}")
        print(f"Successfully parsed: {export_data['parsing_summary']['successfully_parsed']}")
        print(f"Average confidence: {export_data['parsing_summary']['average_confidence']:.3f}")
        print(f"Database matches: {export_data['parsing_summary']['database_matches']}")
        
        print(f"\nParsed Ingredients:")
        for i, ingredient in enumerate(results, 1):
            print(f"{i}. {ingredient.original_text}")
            print(f"   Quantity: {ingredient.quantity}")
            print(f"   Unit: {ingredient.unit}")
            print(f"   Ingredient: {ingredient.ingredient_name}")
            print(f"   Preparation: {ingredient.preparation}")
            print(f"   Confidence: {ingredient.confidence:.3f}")
            print(f"   Method: {ingredient.parsing_method}")
            if ingredient.normalized_quantity and ingredient.normalized_unit:
                print(f"   Normalized: {ingredient.normalized_quantity} {ingredient.normalized_unit}")
            print()
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Parsing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())