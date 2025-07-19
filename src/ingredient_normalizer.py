#!/usr/bin/env python3
"""
Ingredient Normalization Module
Converts ingredient quantities and units to standardized formats
with comprehensive unit conversion and measurement handling.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP

# Add src to path
sys.path.append(str(Path(__file__).parent))

try:
    from pint import UnitRegistry
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False


@dataclass
class NormalizedMeasurement:
    """Normalized measurement with standard units."""
    value: float
    unit: str
    unit_type: str  # 'volume', 'weight', 'count', 'length'
    original_value: Optional[str]
    original_unit: Optional[str]
    conversion_factor: float
    confidence: float


@dataclass
class NormalizationResult:
    """Result of ingredient normalization."""
    normalized_measurement: Optional[NormalizedMeasurement]
    normalized_ingredient: str
    standardized_format: str
    alternatives: List[str]
    errors: List[str]
    confidence: float


class IngredientNormalizer:
    """Comprehensive ingredient normalization with unit conversion."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ingredient normalizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize unit registry
        if PINT_AVAILABLE:
            self.ureg = UnitRegistry()
            self.ureg.define('dash = 0.625 * milliliter')
            self.ureg.define('pinch = 0.31 * milliliter')
            self.ureg.define('smidgen = 0.15 * milliliter')
            self.ureg.define('drop = 0.05 * milliliter')
        else:
            self.ureg = None
        
        # Load conversion tables
        self.unit_conversions = self._load_unit_conversions()
        self.ingredient_densities = self._load_ingredient_densities()
        self.unit_abbreviations = self._load_unit_abbreviations()
        self.measurement_equivalents = self._load_measurement_equivalents()
        
        # Standard units for each type
        self.standard_units = {
            'volume': 'milliliter',
            'weight': 'gram',
            'count': 'piece',
            'length': 'centimeter'
        }
        
        self.logger.info("Initialized IngredientNormalizer")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for normalizer."""
        logger = logging.getLogger('ingredient_normalizer')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _load_unit_conversions(self) -> Dict[str, Dict[str, Any]]:
        """Load unit conversion table."""
        return {
            # Volume conversions (to milliliters)
            'cup': {'ml': 240, 'type': 'volume', 'system': 'us'},
            'cups': {'ml': 240, 'type': 'volume', 'system': 'us'},
            'c': {'ml': 240, 'type': 'volume', 'system': 'us'},
            'tablespoon': {'ml': 15, 'type': 'volume', 'system': 'us'},
            'tablespoons': {'ml': 15, 'type': 'volume', 'system': 'us'},
            'tbsp': {'ml': 15, 'type': 'volume', 'system': 'us'},
            'tbs': {'ml': 15, 'type': 'volume', 'system': 'us'},
            'T': {'ml': 15, 'type': 'volume', 'system': 'us'},
            'teaspoon': {'ml': 5, 'type': 'volume', 'system': 'us'},
            'teaspoons': {'ml': 5, 'type': 'volume', 'system': 'us'},
            'tsp': {'ml': 5, 'type': 'volume', 'system': 'us'},
            'ts': {'ml': 5, 'type': 'volume', 'system': 'us'},
            't': {'ml': 5, 'type': 'volume', 'system': 'us'},
            'fluid_ounce': {'ml': 29.5735, 'type': 'volume', 'system': 'us'},
            'fl_oz': {'ml': 29.5735, 'type': 'volume', 'system': 'us'},
            'pint': {'ml': 473.176, 'type': 'volume', 'system': 'us'},
            'pt': {'ml': 473.176, 'type': 'volume', 'system': 'us'},
            'quart': {'ml': 946.353, 'type': 'volume', 'system': 'us'},
            'qt': {'ml': 946.353, 'type': 'volume', 'system': 'us'},
            'gallon': {'ml': 3785.41, 'type': 'volume', 'system': 'us'},
            'gal': {'ml': 3785.41, 'type': 'volume', 'system': 'us'},
            'liter': {'ml': 1000, 'type': 'volume', 'system': 'metric'},
            'l': {'ml': 1000, 'type': 'volume', 'system': 'metric'},
            'milliliter': {'ml': 1, 'type': 'volume', 'system': 'metric'},
            'ml': {'ml': 1, 'type': 'volume', 'system': 'metric'},
            'dash': {'ml': 0.625, 'type': 'volume', 'system': 'us'},
            'pinch': {'ml': 0.31, 'type': 'volume', 'system': 'us'},
            'smidgen': {'ml': 0.15, 'type': 'volume', 'system': 'us'},
            'drop': {'ml': 0.05, 'type': 'volume', 'system': 'us'},
            
            # Weight conversions (to grams)
            'pound': {'g': 453.592, 'type': 'weight', 'system': 'us'},
            'pounds': {'g': 453.592, 'type': 'weight', 'system': 'us'},
            'lb': {'g': 453.592, 'type': 'weight', 'system': 'us'},
            'lbs': {'g': 453.592, 'type': 'weight', 'system': 'us'},
            'ounce': {'g': 28.3495, 'type': 'weight', 'system': 'us'},
            'ounces': {'g': 28.3495, 'type': 'weight', 'system': 'us'},
            'oz': {'g': 28.3495, 'type': 'weight', 'system': 'us'},
            'gram': {'g': 1, 'type': 'weight', 'system': 'metric'},
            'grams': {'g': 1, 'type': 'weight', 'system': 'metric'},
            'g': {'g': 1, 'type': 'weight', 'system': 'metric'},
            'kilogram': {'g': 1000, 'type': 'weight', 'system': 'metric'},
            'kilograms': {'g': 1000, 'type': 'weight', 'system': 'metric'},
            'kg': {'g': 1000, 'type': 'weight', 'system': 'metric'},
            
            # Count conversions
            'piece': {'count': 1, 'type': 'count', 'system': 'universal'},
            'pieces': {'count': 1, 'type': 'count', 'system': 'universal'},
            'pcs': {'count': 1, 'type': 'count', 'system': 'universal'},
            'pc': {'count': 1, 'type': 'count', 'system': 'universal'},
            'item': {'count': 1, 'type': 'count', 'system': 'universal'},
            'items': {'count': 1, 'type': 'count', 'system': 'universal'},
            'each': {'count': 1, 'type': 'count', 'system': 'universal'},
            'slice': {'count': 1, 'type': 'count', 'system': 'universal'},
            'slices': {'count': 1, 'type': 'count', 'system': 'universal'},
            'clove': {'count': 1, 'type': 'count', 'system': 'universal'},
            'cloves': {'count': 1, 'type': 'count', 'system': 'universal'},
            'head': {'count': 1, 'type': 'count', 'system': 'universal'},
            'heads': {'count': 1, 'type': 'count', 'system': 'universal'},
            'bunch': {'count': 1, 'type': 'count', 'system': 'universal'},
            'bunches': {'count': 1, 'type': 'count', 'system': 'universal'},
            'stalk': {'count': 1, 'type': 'count', 'system': 'universal'},
            'stalks': {'count': 1, 'type': 'count', 'system': 'universal'},
            'sprig': {'count': 1, 'type': 'count', 'system': 'universal'},
            'sprigs': {'count': 1, 'type': 'count', 'system': 'universal'},
            'leaf': {'count': 1, 'type': 'count', 'system': 'universal'},
            'leaves': {'count': 1, 'type': 'count', 'system': 'universal'},
            
            # Length conversions (to centimeters)
            'inch': {'cm': 2.54, 'type': 'length', 'system': 'us'},
            'inches': {'cm': 2.54, 'type': 'length', 'system': 'us'},
            'in': {'cm': 2.54, 'type': 'length', 'system': 'us'},
            'foot': {'cm': 30.48, 'type': 'length', 'system': 'us'},
            'feet': {'cm': 30.48, 'type': 'length', 'system': 'us'},
            'ft': {'cm': 30.48, 'type': 'length', 'system': 'us'},
            'centimeter': {'cm': 1, 'type': 'length', 'system': 'metric'},
            'centimeters': {'cm': 1, 'type': 'length', 'system': 'metric'},
            'cm': {'cm': 1, 'type': 'length', 'system': 'metric'},
            'millimeter': {'cm': 0.1, 'type': 'length', 'system': 'metric'},
            'millimeters': {'cm': 0.1, 'type': 'length', 'system': 'metric'},
            'mm': {'cm': 0.1, 'type': 'length', 'system': 'metric'},
            
            # Container-based measurements
            'can': {'type': 'container', 'system': 'universal'},
            'cans': {'type': 'container', 'system': 'universal'},
            'jar': {'type': 'container', 'system': 'universal'},
            'jars': {'type': 'container', 'system': 'universal'},
            'bottle': {'type': 'container', 'system': 'universal'},
            'bottles': {'type': 'container', 'system': 'universal'},
            'package': {'type': 'container', 'system': 'universal'},
            'packages': {'type': 'container', 'system': 'universal'},
            'pkg': {'type': 'container', 'system': 'universal'},
            'box': {'type': 'container', 'system': 'universal'},
            'boxes': {'type': 'container', 'system': 'universal'},
            'bag': {'type': 'container', 'system': 'universal'},
            'bags': {'type': 'container', 'system': 'universal'}
        }
    
    def _load_ingredient_densities(self) -> Dict[str, float]:
        """Load ingredient density table (g/ml)."""
        return {
            'flour': 0.5,  # all-purpose flour
            'sugar': 0.85,  # granulated sugar
            'brown_sugar': 0.96,
            'powdered_sugar': 0.48,
            'butter': 0.911,
            'milk': 1.03,
            'water': 1.0,
            'oil': 0.92,
            'honey': 1.42,
            'molasses': 1.4,
            'corn_syrup': 1.32,
            'salt': 1.22,
            'baking_powder': 0.9,
            'baking_soda': 2.2,
            'vanilla': 0.88,
            'cocoa_powder': 0.41,
            'rice': 0.75,
            'oats': 0.32,
            'breadcrumbs': 0.24,
            'nuts': 0.58,
            'chocolate_chips': 0.63,
            'raisins': 0.67,
            'coconut': 0.35,
            'cream': 0.994,
            'sour_cream': 0.963,
            'yogurt': 1.04,
            'cheese': 1.08,
            'beans': 0.97,
            'lentils': 0.85,
            'quinoa': 0.73,
            'pasta': 0.67,
            'cornmeal': 0.67,
            'semolina': 0.6,
            'mayonnaise': 0.91,
            'ketchup': 1.14,
            'mustard': 1.05,
            'vinegar': 1.05,
            'wine': 0.99,
            'broth': 1.0,
            'stock': 1.0
        }
    
    def _load_unit_abbreviations(self) -> Dict[str, str]:
        """Load unit abbreviation mappings."""
        return {
            'c': 'cup',
            'C': 'cup',
            'tbsp': 'tablespoon',
            'tbs': 'tablespoon',
            'T': 'tablespoon',
            'tsp': 'teaspoon',
            'ts': 'teaspoon',
            't': 'teaspoon',
            'lb': 'pound',
            'lbs': 'pound',
            'oz': 'ounce',
            'fl_oz': 'fluid_ounce',
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
            'pcs': 'piece',
            'pc': 'piece',
            'pkg': 'package'
        }
    
    def _load_measurement_equivalents(self) -> Dict[str, Dict[str, float]]:
        """Load measurement equivalents and common conversions."""
        return {
            'volume_to_volume': {
                'cup_to_tbsp': 16,
                'cup_to_tsp': 48,
                'tbsp_to_tsp': 3,
                'pint_to_cup': 2,
                'quart_to_cup': 4,
                'gallon_to_cup': 16,
                'liter_to_ml': 1000,
                'fl_oz_to_ml': 29.5735
            },
            'weight_to_weight': {
                'lb_to_oz': 16,
                'kg_to_g': 1000,
                'oz_to_g': 28.3495,
                'lb_to_g': 453.592
            },
            'length_to_length': {
                'ft_to_in': 12,
                'in_to_cm': 2.54,
                'cm_to_mm': 10,
                'ft_to_cm': 30.48
            }
        }
    
    def normalize_quantity(self, quantity_str: str) -> Optional[float]:
        """
        Normalize quantity string to float value.
        
        Args:
            quantity_str: Quantity string (e.g., "1 1/2", "0.5", "half")
            
        Returns:
            Normalized float value or None if invalid
        """
        if not quantity_str:
            return None
        
        # Clean the string
        quantity_str = quantity_str.strip().lower()
        
        # Handle special cases
        special_quantities = {
            'a': 1,
            'an': 1,
            'one': 1,
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9,
            'ten': 10,
            'dozen': 12,
            'half': 0.5,
            'quarter': 0.25,
            'third': 0.333,
            'whole': 1,
            'couple': 2,
            'few': 3,
            'several': 4,
            'handful': 0.25,  # Approximate
            'pinch': 0.0625,  # 1/16 tsp
            'dash': 0.125,    # 1/8 tsp
            'splash': 0.25,   # 1/4 tsp
            'some': 1
        }
        
        if quantity_str in special_quantities:
            return special_quantities[quantity_str]
        
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
            
            # Handle ranges (e.g., "2-3", "1 to 2")
            if '-' in quantity_str:
                parts = re.split(r'[-–—]', quantity_str)
                if len(parts) == 2:
                    try:
                        min_val = float(parts[0].strip())
                        max_val = float(parts[1].strip())
                        return (min_val + max_val) / 2  # Return average
                    except ValueError:
                        pass
            
            if ' to ' in quantity_str:
                parts = quantity_str.split(' to ')
                if len(parts) == 2:
                    try:
                        min_val = float(parts[0].strip())
                        max_val = float(parts[1].strip())
                        return (min_val + max_val) / 2
                    except ValueError:
                        pass
            
            # Handle approximate quantities
            if quantity_str.startswith('about') or quantity_str.startswith('approximately'):
                # Remove the word and parse the rest
                quantity_str = re.sub(r'^(about|approximately)\s+', '', quantity_str)
                return self.normalize_quantity(quantity_str)
            
            # Handle regular numbers
            return float(quantity_str)
            
        except (ValueError, ZeroDivisionError):
            return None
    
    def normalize_unit(self, unit_str: str) -> Optional[str]:
        """
        Normalize unit string to standard form.
        
        Args:
            unit_str: Unit string (e.g., "tsp", "tbsp", "c")
            
        Returns:
            Normalized unit string or None if invalid
        """
        if not unit_str:
            return None
        
        unit_str = unit_str.strip().lower()
        
        # Handle plural forms
        if unit_str.endswith('s') and unit_str[:-1] in self.unit_conversions:
            unit_str = unit_str[:-1]
        
        # Check abbreviations
        if unit_str in self.unit_abbreviations:
            return self.unit_abbreviations[unit_str]
        
        # Check direct matches
        if unit_str in self.unit_conversions:
            return unit_str
        
        # Handle common variations
        variations = {
            'cups': 'cup',
            'tablespoons': 'tablespoon',
            'teaspoons': 'teaspoon',
            'pounds': 'pound',
            'ounces': 'ounce',
            'grams': 'gram',
            'kilograms': 'kilogram',
            'liters': 'liter',
            'milliliters': 'milliliter',
            'inches': 'inch',
            'feet': 'foot',
            'centimeters': 'centimeter',
            'millimeters': 'millimeter',
            'pieces': 'piece',
            'slices': 'slice',
            'cloves': 'clove',
            'heads': 'head',
            'bunches': 'bunch',
            'cans': 'can',
            'jars': 'jar',
            'bottles': 'bottle',
            'packages': 'package',
            'boxes': 'box',
            'bags': 'bag'
        }
        
        return variations.get(unit_str, unit_str)
    
    def convert_to_standard_unit(self, quantity: float, unit: str, ingredient: str = None) -> NormalizedMeasurement:
        """
        Convert quantity and unit to standard measurement.
        
        Args:
            quantity: Numeric quantity
            unit: Unit string
            ingredient: Optional ingredient name for density-based conversions
            
        Returns:
            Normalized measurement
        """
        normalized_unit = self.normalize_unit(unit)
        
        if not normalized_unit or normalized_unit not in self.unit_conversions:
            return NormalizedMeasurement(
                value=quantity,
                unit=unit,
                unit_type='unknown',
                original_value=str(quantity),
                original_unit=unit,
                conversion_factor=1.0,
                confidence=0.5
            )
        
        conversion_data = self.unit_conversions[normalized_unit]
        unit_type = conversion_data['type']
        
        if unit_type == 'volume':
            # Convert to milliliters
            ml_value = quantity * conversion_data['ml']
            return NormalizedMeasurement(
                value=ml_value,
                unit='milliliter',
                unit_type='volume',
                original_value=str(quantity),
                original_unit=unit,
                conversion_factor=conversion_data['ml'],
                confidence=0.9
            )
        
        elif unit_type == 'weight':
            # Convert to grams
            g_value = quantity * conversion_data['g']
            return NormalizedMeasurement(
                value=g_value,
                unit='gram',
                unit_type='weight',
                original_value=str(quantity),
                original_unit=unit,
                conversion_factor=conversion_data['g'],
                confidence=0.9
            )
        
        elif unit_type == 'count':
            # Keep as count
            return NormalizedMeasurement(
                value=quantity,
                unit='piece',
                unit_type='count',
                original_value=str(quantity),
                original_unit=unit,
                conversion_factor=1.0,
                confidence=0.9
            )
        
        elif unit_type == 'length':
            # Convert to centimeters
            cm_value = quantity * conversion_data['cm']
            return NormalizedMeasurement(
                value=cm_value,
                unit='centimeter',
                unit_type='length',
                original_value=str(quantity),
                original_unit=unit,
                conversion_factor=conversion_data['cm'],
                confidence=0.9
            )
        
        elif unit_type == 'container':
            # Try to estimate based on common container sizes
            estimated_ml = self._estimate_container_volume(normalized_unit, ingredient)
            if estimated_ml:
                return NormalizedMeasurement(
                    value=estimated_ml * quantity,
                    unit='milliliter',
                    unit_type='volume',
                    original_value=str(quantity),
                    original_unit=unit,
                    conversion_factor=estimated_ml,
                    confidence=0.6
                )
            else:
                return NormalizedMeasurement(
                    value=quantity,
                    unit='piece',
                    unit_type='count',
                    original_value=str(quantity),
                    original_unit=unit,
                    conversion_factor=1.0,
                    confidence=0.7
                )
        
        else:
            # Unknown unit type
            return NormalizedMeasurement(
                value=quantity,
                unit=unit,
                unit_type='unknown',
                original_value=str(quantity),
                original_unit=unit,
                conversion_factor=1.0,
                confidence=0.5
            )
    
    def _estimate_container_volume(self, container_type: str, ingredient: str = None) -> Optional[float]:
        """Estimate container volume in milliliters."""
        # Standard container sizes (approximate)
        container_sizes = {
            'can': {
                'default': 400,  # 14 oz can
                'tomato': 400,   # 14 oz
                'beans': 425,    # 15 oz
                'soup': 300,     # 10.5 oz
                'tuna': 150,     # 5 oz
                'soda': 355      # 12 oz
            },
            'jar': {
                'default': 500,  # 16 oz jar
                'jam': 340,      # 12 oz
                'peanut_butter': 500,  # 16 oz
                'pickles': 680,  # 24 oz
                'sauce': 680     # 24 oz
            },
            'bottle': {
                'default': 500,  # 16 oz bottle
                'wine': 750,     # 25 oz
                'beer': 355,     # 12 oz
                'soda': 500,     # 16 oz
                'oil': 500,      # 16 oz
                'vinegar': 500   # 16 oz
            },
            'package': {
                'default': 250,  # 8 oz
                'cheese': 225,   # 8 oz
                'pasta': 450,    # 1 lb
                'flour': 1000,   # 2 lb
                'sugar': 900     # 2 lb
            },
            'box': {
                'default': 300,  # 10 oz
                'cereal': 400,   # 14 oz
                'pasta': 450,    # 1 lb
                'rice': 900,     # 2 lb
                'crackers': 200  # 7 oz
            },
            'bag': {
                'default': 450,  # 1 lb
                'flour': 2250,   # 5 lb
                'sugar': 2250,   # 5 lb
                'rice': 2250,    # 5 lb
                'chips': 200,    # 7 oz
                'nuts': 150      # 5 oz
            }
        }
        
        if container_type in container_sizes:
            container_data = container_sizes[container_type]
            
            # Try to match ingredient
            if ingredient:
                ingredient_lower = ingredient.lower()
                for ing_key, volume in container_data.items():
                    if ing_key in ingredient_lower:
                        return volume
            
            # Return default
            return container_data['default']
        
        return None
    
    def convert_volume_to_weight(self, volume_ml: float, ingredient: str) -> Optional[float]:
        """
        Convert volume to weight using ingredient density.
        
        Args:
            volume_ml: Volume in milliliters
            ingredient: Ingredient name
            
        Returns:
            Weight in grams or None if density unknown
        """
        # Normalize ingredient name
        ingredient_lower = ingredient.lower()
        
        # Find density
        density = None
        for ing_key, ing_density in self.ingredient_densities.items():
            if ing_key in ingredient_lower:
                density = ing_density
                break
        
        if density:
            return volume_ml * density
        
        return None
    
    def convert_weight_to_volume(self, weight_g: float, ingredient: str) -> Optional[float]:
        """
        Convert weight to volume using ingredient density.
        
        Args:
            weight_g: Weight in grams
            ingredient: Ingredient name
            
        Returns:
            Volume in milliliters or None if density unknown
        """
        # Normalize ingredient name
        ingredient_lower = ingredient.lower()
        
        # Find density
        density = None
        for ing_key, ing_density in self.ingredient_densities.items():
            if ing_key in ingredient_lower:
                density = ing_density
                break
        
        if density:
            return weight_g / density
        
        return None
    
    def normalize_ingredient_name(self, ingredient: str) -> str:
        """
        Normalize ingredient name to standard form.
        
        Args:
            ingredient: Raw ingredient name
            
        Returns:
            Normalized ingredient name
        """
        if not ingredient:
            return ""
        
        # Convert to lowercase
        normalized = ingredient.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove parenthetical information
        normalized = re.sub(r'\([^)]*\)', '', normalized)
        
        # Remove common descriptors that don't affect the ingredient
        descriptors_to_remove = [
            'fresh', 'frozen', 'canned', 'dried', 'ground', 'whole', 'chopped',
            'sliced', 'diced', 'minced', 'grated', 'shredded', 'beaten',
            'melted', 'softened', 'room temperature', 'cold', 'hot', 'warm',
            'organic', 'free-range', 'grass-fed', 'wild-caught', 'raw',
            'cooked', 'uncooked', 'peeled', 'seeded', 'stemmed', 'trimmed',
            'washed', 'cleaned', 'rinsed', 'drained'
        ]
        
        for descriptor in descriptors_to_remove:
            normalized = re.sub(r'\b' + descriptor + r'\b', '', normalized)
        
        # Remove extra whitespace again
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Handle common ingredient variations
        ingredient_mappings = {
            'all-purpose flour': 'flour',
            'ap flour': 'flour',
            'plain flour': 'flour',
            'bread flour': 'flour',
            'cake flour': 'flour',
            'self-rising flour': 'flour',
            'whole wheat flour': 'flour',
            'white sugar': 'sugar',
            'granulated sugar': 'sugar',
            'caster sugar': 'sugar',
            'superfine sugar': 'sugar',
            'light brown sugar': 'brown sugar',
            'dark brown sugar': 'brown sugar',
            'packed brown sugar': 'brown sugar',
            'confectioners sugar': 'powdered sugar',
            'icing sugar': 'powdered sugar',
            'unsalted butter': 'butter',
            'salted butter': 'butter',
            'sweet butter': 'butter',
            'whole milk': 'milk',
            '2% milk': 'milk',
            'skim milk': 'milk',
            'low-fat milk': 'milk',
            'heavy cream': 'cream',
            'heavy whipping cream': 'cream',
            'whipping cream': 'cream',
            'large eggs': 'eggs',
            'medium eggs': 'eggs',
            'small eggs': 'eggs',
            'table salt': 'salt',
            'kosher salt': 'salt',
            'sea salt': 'salt',
            'fine salt': 'salt',
            'coarse salt': 'salt',
            'extra virgin olive oil': 'olive oil',
            'virgin olive oil': 'olive oil',
            'vegetable oil': 'oil',
            'canola oil': 'oil',
            'sunflower oil': 'oil',
            'yellow onion': 'onion',
            'white onion': 'onion',
            'red onion': 'onion',
            'sweet onion': 'onion'
        }
        
        # Check for mappings
        for variation, canonical in ingredient_mappings.items():
            if variation in normalized:
                normalized = normalized.replace(variation, canonical)
                break
        
        # Final cleanup
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def normalize_ingredient_complete(self, quantity_str: str, unit_str: str, ingredient: str) -> NormalizationResult:
        """
        Perform complete ingredient normalization.
        
        Args:
            quantity_str: Quantity string
            unit_str: Unit string
            ingredient: Ingredient name
            
        Returns:
            Complete normalization result
        """
        errors = []
        alternatives = []
        
        # Normalize quantity
        normalized_quantity = self.normalize_quantity(quantity_str)
        if not normalized_quantity and quantity_str:
            errors.append(f"Could not parse quantity: {quantity_str}")
        
        # Normalize unit
        normalized_unit = self.normalize_unit(unit_str)
        if not normalized_unit and unit_str:
            errors.append(f"Could not normalize unit: {unit_str}")
        
        # Normalize ingredient name
        normalized_ingredient = self.normalize_ingredient_name(ingredient)
        
        # Convert to standard measurement
        normalized_measurement = None
        if normalized_quantity and normalized_unit:
            normalized_measurement = self.convert_to_standard_unit(
                normalized_quantity, normalized_unit, normalized_ingredient
            )
        elif normalized_quantity and not normalized_unit:
            # Quantity without unit - assume pieces
            normalized_measurement = NormalizedMeasurement(
                value=normalized_quantity,
                unit='piece',
                unit_type='count',
                original_value=quantity_str,
                original_unit=unit_str,
                conversion_factor=1.0,
                confidence=0.8
            )
        
        # Generate standardized format
        if normalized_measurement:
            if normalized_measurement.unit_type == 'count':
                if normalized_measurement.value == 1:
                    standardized_format = f"{normalized_measurement.value:.0f} {normalized_ingredient}"
                else:
                    standardized_format = f"{normalized_measurement.value:.0f} {normalized_ingredient}"
            else:
                # Format with appropriate precision
                if normalized_measurement.value >= 1000:
                    value_str = f"{normalized_measurement.value:.0f}"
                elif normalized_measurement.value >= 100:
                    value_str = f"{normalized_measurement.value:.1f}"
                else:
                    value_str = f"{normalized_measurement.value:.2f}"
                
                standardized_format = f"{value_str} {normalized_measurement.unit} {normalized_ingredient}"
        else:
            standardized_format = f"{quantity_str} {unit_str} {normalized_ingredient}".strip()
        
        # Generate alternatives
        if normalized_measurement:
            alternatives = self._generate_measurement_alternatives(normalized_measurement, normalized_ingredient)
        
        # Calculate confidence
        confidence = self._calculate_normalization_confidence(
            normalized_quantity, normalized_unit, normalized_ingredient, errors
        )
        
        return NormalizationResult(
            normalized_measurement=normalized_measurement,
            normalized_ingredient=normalized_ingredient,
            standardized_format=standardized_format,
            alternatives=alternatives,
            errors=errors,
            confidence=confidence
        )
    
    def _generate_measurement_alternatives(self, measurement: NormalizedMeasurement, ingredient: str) -> List[str]:
        """Generate alternative measurement formats."""
        alternatives = []
        
        if measurement.unit_type == 'volume':
            # Convert to common volume units
            ml_value = measurement.value
            
            # Convert to cups
            if ml_value >= 240:
                cups = ml_value / 240
                if cups >= 1:
                    alternatives.append(f"{cups:.2f} cups {ingredient}")
            
            # Convert to tablespoons
            if 15 <= ml_value <= 240:
                tbsp = ml_value / 15
                alternatives.append(f"{tbsp:.1f} tablespoons {ingredient}")
            
            # Convert to teaspoons
            if 5 <= ml_value <= 60:
                tsp = ml_value / 5
                alternatives.append(f"{tsp:.1f} teaspoons {ingredient}")
            
            # Convert to liters
            if ml_value >= 1000:
                liters = ml_value / 1000
                alternatives.append(f"{liters:.2f} liters {ingredient}")
        
        elif measurement.unit_type == 'weight':
            # Convert to common weight units
            g_value = measurement.value
            
            # Convert to ounces
            if g_value >= 28.35:
                oz = g_value / 28.35
                alternatives.append(f"{oz:.2f} ounces {ingredient}")
            
            # Convert to pounds
            if g_value >= 453.59:
                lbs = g_value / 453.59
                alternatives.append(f"{lbs:.2f} pounds {ingredient}")
            
            # Convert to kilograms
            if g_value >= 1000:
                kg = g_value / 1000
                alternatives.append(f"{kg:.2f} kilograms {ingredient}")
        
        return alternatives[:3]  # Limit to 3 alternatives
    
    def _calculate_normalization_confidence(self, quantity: Optional[float], unit: Optional[str], 
                                          ingredient: str, errors: List[str]) -> float:
        """Calculate confidence score for normalization."""
        confidence = 1.0
        
        # Reduce confidence for errors
        confidence -= len(errors) * 0.2
        
        # Reduce confidence for missing components
        if not quantity:
            confidence -= 0.2
        if not unit:
            confidence -= 0.1
        if not ingredient:
            confidence -= 0.3
        
        # Boost confidence for known ingredients
        if ingredient.lower() in self.ingredient_densities:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def batch_normalize(self, ingredients: List[Dict[str, str]]) -> List[NormalizationResult]:
        """
        Normalize multiple ingredients in batch.
        
        Args:
            ingredients: List of ingredient dictionaries with 'quantity', 'unit', 'ingredient' keys
            
        Returns:
            List of normalization results
        """
        results = []
        
        for ingredient_data in ingredients:
            try:
                result = self.normalize_ingredient_complete(
                    ingredient_data.get('quantity', ''),
                    ingredient_data.get('unit', ''),
                    ingredient_data.get('ingredient', '')
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to normalize ingredient {ingredient_data}: {e}")
                # Create error result
                results.append(NormalizationResult(
                    normalized_measurement=None,
                    normalized_ingredient=ingredient_data.get('ingredient', ''),
                    standardized_format=f"{ingredient_data.get('quantity', '')} {ingredient_data.get('unit', '')} {ingredient_data.get('ingredient', '')}".strip(),
                    alternatives=[],
                    errors=[f"Normalization failed: {str(e)}"],
                    confidence=0.0
                ))
        
        return results


def main():
    """Main normalization script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Normalize ingredient measurements')
    parser.add_argument('--quantity', '-q', help='Quantity to normalize')
    parser.add_argument('--unit', '-u', help='Unit to normalize')
    parser.add_argument('--ingredient', '-i', help='Ingredient name')
    parser.add_argument('--file', '-f', help='File with ingredient data (JSON)')
    parser.add_argument('--output', '-o', help='Output file')
    
    args = parser.parse_args()
    
    # Initialize normalizer
    normalizer = IngredientNormalizer()
    
    try:
        if args.quantity or args.unit or args.ingredient:
            # Single ingredient normalization
            result = normalizer.normalize_ingredient_complete(
                args.quantity or '',
                args.unit or '',
                args.ingredient or ''
            )
            results = [result]
        
        elif args.file:
            # Batch normalization from file
            with open(args.file, 'r') as f:
                ingredients = json.load(f)
            results = normalizer.batch_normalize(ingredients)
        
        else:
            # Interactive mode
            print("Enter ingredient details (empty to finish):")
            ingredients = []
            
            while True:
                quantity = input("Quantity: ").strip()
                if not quantity:
                    break
                
                unit = input("Unit: ").strip()
                ingredient = input("Ingredient: ").strip()
                
                ingredients.append({
                    'quantity': quantity,
                    'unit': unit,
                    'ingredient': ingredient
                })
            
            if ingredients:
                results = normalizer.batch_normalize(ingredients)
            else:
                print("No ingredients provided.")
                return 1
        
        # Print results
        print(f"\nNormalization Results:")
        print(f"====================")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.standardized_format}")
            
            if result.normalized_measurement:
                print(f"   Normalized: {result.normalized_measurement.value:.2f} {result.normalized_measurement.unit}")
                print(f"   Unit Type: {result.normalized_measurement.unit_type}")
                print(f"   Conversion Factor: {result.normalized_measurement.conversion_factor}")
            
            print(f"   Confidence: {result.confidence:.3f}")
            
            if result.alternatives:
                print(f"   Alternatives:")
                for alt in result.alternatives:
                    print(f"     - {alt}")
            
            if result.errors:
                print(f"   Errors:")
                for error in result.errors:
                    print(f"     - {error}")
        
        # Save results if requested
        if args.output:
            output_data = {
                'results': [asdict(result) for result in results],
                'summary': {
                    'total_ingredients': len(results),
                    'successful_normalizations': len([r for r in results if r.normalized_measurement]),
                    'average_confidence': sum(r.confidence for r in results) / len(results) if results else 0
                }
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            print(f"\nResults saved to: {args.output}")
    
    except Exception as e:
        print(f"Normalization failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())