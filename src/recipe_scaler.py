#!/usr/bin/env python3
"""
Recipe Scaling and Conversion System
Advanced recipe scaling with unit conversions, dietary modifications,
and intelligent scaling of cooking times and temperatures.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from ingredient_normalizer import IngredientNormalizer, NormalizationResult
from enhanced_ingredient_parser import EnhancedIngredient


@dataclass
class ScalingOptions:
    """Recipe scaling options."""
    target_servings: Optional[int] = None
    scale_factor: Optional[float] = None
    dietary_modifications: List[str] = None
    unit_system: str = "metric"  # 'metric', 'imperial', 'us'
    precision: int = 2
    round_to_nice_numbers: bool = True
    scale_cooking_times: bool = True
    scale_temperatures: bool = False
    
    def __post_init__(self):
        if self.dietary_modifications is None:
            self.dietary_modifications = []


@dataclass
class ScaledIngredient:
    """Scaled ingredient with conversions."""
    original_ingredient: EnhancedIngredient
    scaled_quantity: Optional[float]
    scaled_unit: Optional[str]
    display_quantity: str
    display_unit: str
    scaling_factor: float
    conversion_applied: bool
    scaling_notes: List[str]
    
    # Alternative representations
    fraction_display: Optional[str] = None
    decimal_display: Optional[str] = None
    alternative_units: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.scaling_notes is None:
            self.scaling_notes = []
        if self.alternative_units is None:
            self.alternative_units = []


@dataclass
class ScaledRecipe:
    """Complete scaled recipe."""
    original_servings: Optional[int]
    target_servings: Optional[int]
    scaling_factor: float
    scaled_ingredients: List[ScaledIngredient]
    scaled_instructions: List[str]
    scaled_cooking_times: Dict[str, Any]
    scaled_temperatures: Dict[str, Any]
    scaling_options: ScalingOptions
    scaling_notes: List[str]
    nutritional_scaling: Optional[Dict[str, Any]]
    
    def __post_init__(self):
        if self.scaling_notes is None:
            self.scaling_notes = []


class RecipeScaler:
    """Advanced recipe scaling and conversion system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize recipe scaler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize normalizer
        self.normalizer = IngredientNormalizer(self.config.get('normalizer', {}))
        
        # Scaling configurations
        self.nice_numbers = [0.25, 0.33, 0.5, 0.67, 0.75, 1, 1.25, 1.33, 1.5, 1.67, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 10]
        self.fraction_threshold = 0.1  # Use fractions for values less than this
        
        # Unit conversion preferences
        self.unit_preferences = {
            'metric': {
                'volume': ['ml', 'l'],
                'weight': ['g', 'kg'],
                'temperature': 'celsius'
            },
            'imperial': {
                'volume': ['fl oz', 'pint', 'quart'],
                'weight': ['oz', 'lb'],
                'temperature': 'fahrenheit'
            },
            'us': {
                'volume': ['tsp', 'tbsp', 'cup'],
                'weight': ['oz', 'lb'],
                'temperature': 'fahrenheit'
            }
        }
        
        # Cooking time scaling factors
        self.cooking_time_scaling = {
            'baking': lambda factor: factor ** 0.67,  # Non-linear scaling for baking
            'roasting': lambda factor: factor ** 0.67,
            'simmering': lambda factor: factor ** 0.5,
            'frying': lambda factor: factor ** 0.8,
            'steaming': lambda factor: factor ** 0.5,
            'boiling': lambda factor: 1.0,  # Time doesn't scale with quantity
            'grilling': lambda factor: factor ** 0.8,
            'default': lambda factor: factor ** 0.67
        }
        
        # Temperature scaling (usually not needed, but for some applications)
        self.temperature_scaling = {
            'oven': lambda factor: 1.0,  # Oven temp usually doesn't change
            'stovetop': lambda factor: 1.0,
            'deep_fry': lambda factor: 1.0,
            'default': lambda factor: 1.0
        }
        
        # Dietary modification substitutions
        self.dietary_substitutions = {
            'vegan': {
                'butter': 'vegan butter',
                'milk': 'plant milk',
                'eggs': 'flax eggs',
                'cheese': 'vegan cheese',
                'honey': 'maple syrup',
                'cream': 'coconut cream'
            },
            'gluten_free': {
                'flour': 'gluten-free flour',
                'wheat flour': 'gluten-free flour',
                'bread': 'gluten-free bread',
                'pasta': 'gluten-free pasta',
                'soy sauce': 'tamari'
            },
            'low_sodium': {
                'salt': 'herb seasoning',
                'soy sauce': 'low-sodium soy sauce',
                'broth': 'low-sodium broth'
            },
            'keto': {
                'flour': 'almond flour',
                'sugar': 'erythritol',
                'potato': 'cauliflower',
                'rice': 'cauliflower rice'
            },
            'paleo': {
                'flour': 'almond flour',
                'sugar': 'honey',
                'butter': 'coconut oil',
                'cheese': 'nutritional yeast'
            }
        }
        
        self.logger.info("Initialized RecipeScaler")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for scaler."""
        logger = logging.getLogger('recipe_scaler')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def scale_recipe(self, ingredients: List[EnhancedIngredient], 
                    original_servings: Optional[int],
                    instructions: List[str],
                    options: ScalingOptions) -> ScaledRecipe:
        """
        Scale recipe with advanced options.
        
        Args:
            ingredients: List of original ingredients
            original_servings: Original number of servings
            instructions: Original cooking instructions
            options: Scaling options
            
        Returns:
            Scaled recipe with all modifications
        """
        # Calculate scaling factor
        if options.scale_factor:
            scaling_factor = options.scale_factor
            target_servings = int(original_servings * scaling_factor) if original_servings else None
        elif options.target_servings and original_servings:
            scaling_factor = options.target_servings / original_servings
            target_servings = options.target_servings
        else:
            scaling_factor = 1.0
            target_servings = original_servings
        
        scaling_notes = []
        
        # Scale ingredients
        scaled_ingredients = []
        for ingredient in ingredients:
            scaled_ingredient = self._scale_ingredient(ingredient, scaling_factor, options)
            scaled_ingredients.append(scaled_ingredient)
        
        # Apply dietary modifications
        if options.dietary_modifications:
            scaled_ingredients = self._apply_dietary_modifications(
                scaled_ingredients, options.dietary_modifications
            )
            scaling_notes.extend([f"Applied {mod} modifications" for mod in options.dietary_modifications])
        
        # Scale instructions
        scaled_instructions = self._scale_instructions(instructions, scaling_factor, options)
        
        # Scale cooking times
        scaled_cooking_times = {}
        if options.scale_cooking_times:
            scaled_cooking_times = self._scale_cooking_times(instructions, scaling_factor)
            if scaled_cooking_times:
                scaling_notes.append("Cooking times adjusted for scaling")
        
        # Scale temperatures
        scaled_temperatures = {}
        if options.scale_temperatures:
            scaled_temperatures = self._scale_temperatures(instructions, scaling_factor)
            if scaled_temperatures:
                scaling_notes.append("Temperatures adjusted for scaling")
        
        # Calculate nutritional scaling
        nutritional_scaling = self._calculate_nutritional_scaling(
            scaled_ingredients, scaling_factor
        )
        
        return ScaledRecipe(
            original_servings=original_servings,
            target_servings=target_servings,
            scaling_factor=scaling_factor,
            scaled_ingredients=scaled_ingredients,
            scaled_instructions=scaled_instructions,
            scaled_cooking_times=scaled_cooking_times,
            scaled_temperatures=scaled_temperatures,
            scaling_options=options,
            scaling_notes=scaling_notes,
            nutritional_scaling=nutritional_scaling
        )
    
    def _scale_ingredient(self, ingredient: EnhancedIngredient, 
                         scaling_factor: float, options: ScalingOptions) -> ScaledIngredient:
        """
        Scale individual ingredient.
        
        Args:
            ingredient: Original ingredient
            scaling_factor: Scaling factor
            options: Scaling options
            
        Returns:
            Scaled ingredient
        """
        scaling_notes = []
        
        # Handle ingredients without quantities
        if not ingredient.normalized_quantity:
            return ScaledIngredient(
                original_ingredient=ingredient,
                scaled_quantity=None,
                scaled_unit=ingredient.unit,
                display_quantity="",
                display_unit=ingredient.unit or "",
                scaling_factor=scaling_factor,
                conversion_applied=False,
                scaling_notes=["No quantity to scale"]
            )
        
        # Scale the quantity
        scaled_quantity = ingredient.normalized_quantity * scaling_factor
        scaled_unit = ingredient.normalized_unit
        
        # Convert to preferred unit system
        converted_quantity = scaled_quantity
        converted_unit = scaled_unit
        conversion_applied = False
        
        if options.unit_system != "metric":
            converted_quantity, converted_unit = self._convert_to_unit_system(
                scaled_quantity, scaled_unit, options.unit_system
            )
            if converted_unit != scaled_unit:
                conversion_applied = True
                scaling_notes.append(f"Converted to {options.unit_system} units")
        
        # Round to nice numbers if requested
        if options.round_to_nice_numbers:
            converted_quantity = self._round_to_nice_number(converted_quantity)
            scaling_notes.append("Rounded to nice number")
        
        # Format for display
        display_quantity, display_unit = self._format_quantity_for_display(
            converted_quantity, converted_unit, options
        )
        
        # Generate alternative representations
        fraction_display = self._to_fraction_display(converted_quantity)
        decimal_display = f"{converted_quantity:.{options.precision}f}"
        alternative_units = self._generate_alternative_units(
            converted_quantity, converted_unit, options.unit_system
        )
        
        return ScaledIngredient(
            original_ingredient=ingredient,
            scaled_quantity=converted_quantity,
            scaled_unit=converted_unit,
            display_quantity=display_quantity,
            display_unit=display_unit,
            scaling_factor=scaling_factor,
            conversion_applied=conversion_applied,
            scaling_notes=scaling_notes,
            fraction_display=fraction_display,
            decimal_display=decimal_display,
            alternative_units=alternative_units
        )
    
    def _convert_to_unit_system(self, quantity: float, unit: str, 
                               target_system: str) -> Tuple[float, str]:
        """Convert quantity to target unit system."""
        if target_system == "metric":
            return quantity, unit
        
        # Define conversion mappings
        conversions = {
            "imperial": {
                "milliliter": (0.0351951, "fl oz"),
                "liter": (1.76, "pint"),
                "gram": (0.035274, "oz"),
                "kilogram": (2.20462, "lb")
            },
            "us": {
                "milliliter": (0.202884, "tsp"),
                "liter": (4.22675, "cup"),
                "gram": (0.035274, "oz"),
                "kilogram": (2.20462, "lb")
            }
        }
        
        if target_system in conversions and unit in conversions[target_system]:
            factor, new_unit = conversions[target_system][unit]
            return quantity * factor, new_unit
        
        return quantity, unit
    
    def _round_to_nice_number(self, value: float) -> float:
        """Round value to nearest nice number."""
        if value <= 0:
            return value
        
        # Find the closest nice number
        closest_nice = min(self.nice_numbers, key=lambda x: abs(x - value))
        
        # Only use nice number if it's reasonably close
        if abs(closest_nice - value) / value < 0.15:  # Within 15%
            return closest_nice
        
        # Otherwise, round to reasonable precision
        if value < 1:
            return round(value, 2)
        elif value < 10:
            return round(value, 1)
        else:
            return round(value)
    
    def _format_quantity_for_display(self, quantity: float, unit: str, 
                                   options: ScalingOptions) -> Tuple[str, str]:
        """Format quantity for display."""
        if quantity < self.fraction_threshold:
            # Use fractions for small quantities
            fraction_str = self._to_fraction_display(quantity)
            if fraction_str:
                return fraction_str, unit
        
        # Use decimal with specified precision
        if quantity == int(quantity):
            return str(int(quantity)), unit
        else:
            return f"{quantity:.{options.precision}f}", unit
    
    def _to_fraction_display(self, value: float) -> Optional[str]:
        """Convert decimal to fraction display."""
        if value <= 0:
            return None
        
        try:
            # Convert to fraction
            fraction = Fraction(value).limit_denominator(16)
            
            # Only use fractions for common denominators
            if fraction.denominator in [2, 3, 4, 6, 8, 12, 16]:
                if fraction.numerator > fraction.denominator:
                    # Mixed number
                    whole = fraction.numerator // fraction.denominator
                    remainder = fraction.numerator % fraction.denominator
                    if remainder == 0:
                        return str(whole)
                    else:
                        return f"{whole} {remainder}/{fraction.denominator}"
                else:
                    # Simple fraction
                    return f"{fraction.numerator}/{fraction.denominator}"
        except:
            pass
        
        return None
    
    def _generate_alternative_units(self, quantity: float, unit: str, 
                                  unit_system: str) -> List[Dict[str, str]]:
        """Generate alternative unit representations."""
        alternatives = []
        
        # Add common alternatives based on unit type
        if unit in ["ml", "milliliter"]:
            if quantity >= 1000:
                alternatives.append({
                    "quantity": f"{quantity / 1000:.2f}",
                    "unit": "l",
                    "display": f"{quantity / 1000:.2f} l"
                })
            if unit_system == "us":
                if quantity >= 240:
                    alternatives.append({
                        "quantity": f"{quantity / 240:.2f}",
                        "unit": "cup",
                        "display": f"{quantity / 240:.2f} cups"
                    })
                elif quantity >= 15:
                    alternatives.append({
                        "quantity": f"{quantity / 15:.1f}",
                        "unit": "tbsp",
                        "display": f"{quantity / 15:.1f} tbsp"
                    })
                elif quantity >= 5:
                    alternatives.append({
                        "quantity": f"{quantity / 5:.1f}",
                        "unit": "tsp",
                        "display": f"{quantity / 5:.1f} tsp"
                    })
        
        elif unit in ["g", "gram"]:
            if quantity >= 1000:
                alternatives.append({
                    "quantity": f"{quantity / 1000:.2f}",
                    "unit": "kg",
                    "display": f"{quantity / 1000:.2f} kg"
                })
            if unit_system in ["us", "imperial"]:
                if quantity >= 453.592:
                    alternatives.append({
                        "quantity": f"{quantity / 453.592:.2f}",
                        "unit": "lb",
                        "display": f"{quantity / 453.592:.2f} lb"
                    })
                elif quantity >= 28.35:
                    alternatives.append({
                        "quantity": f"{quantity / 28.35:.1f}",
                        "unit": "oz",
                        "display": f"{quantity / 28.35:.1f} oz"
                    })
        
        return alternatives
    
    def _apply_dietary_modifications(self, ingredients: List[ScaledIngredient], 
                                   modifications: List[str]) -> List[ScaledIngredient]:
        """Apply dietary modifications to ingredients."""
        modified_ingredients = []
        
        for ingredient in ingredients:
            modified_ingredient = ingredient
            
            # Apply each modification
            for modification in modifications:
                if modification in self.dietary_substitutions:
                    substitutions = self.dietary_substitutions[modification]
                    
                    ingredient_name = ingredient.original_ingredient.ingredient_name.lower()
                    
                    for original, substitute in substitutions.items():
                        if original in ingredient_name:
                            # Create modified ingredient
                            modified_ingredient = ScaledIngredient(
                                original_ingredient=ingredient.original_ingredient,
                                scaled_quantity=ingredient.scaled_quantity,
                                scaled_unit=ingredient.scaled_unit,
                                display_quantity=ingredient.display_quantity,
                                display_unit=ingredient.display_unit,
                                scaling_factor=ingredient.scaling_factor,
                                conversion_applied=ingredient.conversion_applied,
                                scaling_notes=ingredient.scaling_notes + [f"Substituted with {substitute} for {modification}"],
                                fraction_display=ingredient.fraction_display,
                                decimal_display=ingredient.decimal_display,
                                alternative_units=ingredient.alternative_units
                            )
                            
                            # Update the ingredient name in the original ingredient
                            modified_ingredient.original_ingredient.ingredient_name = substitute
                            break
            
            modified_ingredients.append(modified_ingredient)
        
        return modified_ingredients
    
    def _scale_instructions(self, instructions: List[str], scaling_factor: float, 
                          options: ScalingOptions) -> List[str]:
        """Scale cooking instructions."""
        scaled_instructions = []
        
        for instruction in instructions:
            scaled_instruction = instruction
            
            # Scale cooking times if enabled
            if options.scale_cooking_times:
                scaled_instruction = self._scale_times_in_text(
                    scaled_instruction, scaling_factor
                )
            
            scaled_instructions.append(scaled_instruction)
        
        return scaled_instructions
    
    def _scale_times_in_text(self, text: str, scaling_factor: float) -> str:
        """Scale cooking times mentioned in text."""
        import re
        
        # Pattern for time expressions
        time_patterns = [
            r'(\d+)\s*(?:to\s+\d+\s*)?(?:minutes?|mins?)',
            r'(\d+)\s*(?:to\s+\d+\s*)?(?:hours?|hrs?)',
            r'(\d+)\s*(?:to\s+\d+\s*)?(?:seconds?|secs?)'
        ]
        
        scaled_text = text
        
        for pattern in time_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                original_time = int(match.group(1))
                
                # Apply scaling (use square root for more realistic scaling)
                scaled_time = int(original_time * (scaling_factor ** 0.67))
                
                # Replace in text
                scaled_text = scaled_text.replace(
                    match.group(0),
                    match.group(0).replace(str(original_time), str(scaled_time))
                )
        
        return scaled_text
    
    def _scale_cooking_times(self, instructions: List[str], scaling_factor: float) -> Dict[str, Any]:
        """Extract and scale cooking times from instructions."""
        import re
        
        cooking_times = {}
        
        # Common cooking methods and their time patterns
        cooking_methods = {
            'baking': r'bake.*?(\d+)\s*(?:minutes?|mins?)',
            'roasting': r'roast.*?(\d+)\s*(?:minutes?|mins?)',
            'simmering': r'simmer.*?(\d+)\s*(?:minutes?|mins?)',
            'boiling': r'boil.*?(\d+)\s*(?:minutes?|mins?)',
            'frying': r'fry.*?(\d+)\s*(?:minutes?|mins?)',
            'grilling': r'grill.*?(\d+)\s*(?:minutes?|mins?)'
        }
        
        for instruction in instructions:
            for method, pattern in cooking_methods.items():
                matches = re.finditer(pattern, instruction, re.IGNORECASE)
                for match in matches:
                    original_time = int(match.group(1))
                    
                    # Apply method-specific scaling
                    scaling_func = self.cooking_time_scaling.get(method, self.cooking_time_scaling['default'])
                    scaled_time = int(original_time * scaling_func(scaling_factor))
                    
                    cooking_times[method] = {
                        'original_time': original_time,
                        'scaled_time': scaled_time,
                        'scaling_factor': scaling_func(scaling_factor)
                    }
        
        return cooking_times
    
    def _scale_temperatures(self, instructions: List[str], scaling_factor: float) -> Dict[str, Any]:
        """Extract and scale temperatures from instructions."""
        import re
        
        temperatures = {}
        
        # Temperature patterns
        temp_patterns = [
            r'(\d+)\s*°?[Ff]',  # Fahrenheit
            r'(\d+)\s*°?[Cc]',  # Celsius
            r'(\d+)\s*degrees?'  # Generic degrees
        ]
        
        for instruction in instructions:
            for pattern in temp_patterns:
                matches = re.finditer(pattern, instruction, re.IGNORECASE)
                for match in matches:
                    temp_value = int(match.group(1))
                    
                    # Usually temperatures don't scale with quantity
                    # But include logic for special cases
                    scaled_temp = temp_value  # No scaling for most cases
                    
                    temperatures[f"temp_{temp_value}"] = {
                        'original_temp': temp_value,
                        'scaled_temp': scaled_temp,
                        'scaling_applied': False
                    }
        
        return temperatures
    
    def _calculate_nutritional_scaling(self, ingredients: List[ScaledIngredient], 
                                     scaling_factor: float) -> Optional[Dict[str, Any]]:
        """Calculate how nutrition scales with ingredient changes."""
        if not ingredients:
            return None
        
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0
        
        ingredients_with_nutrition = 0
        
        for ingredient in ingredients:
            if ingredient.original_ingredient.nutritional_info:
                nutrition = ingredient.original_ingredient.nutritional_info
                
                if nutrition.get('calories'):
                    total_calories += nutrition['calories'] * scaling_factor
                    ingredients_with_nutrition += 1
                
                if nutrition.get('protein_g'):
                    total_protein += nutrition['protein_g'] * scaling_factor
                
                if nutrition.get('carbs_g'):
                    total_carbs += nutrition['carbs_g'] * scaling_factor
                
                if nutrition.get('fat_g'):
                    total_fat += nutrition['fat_g'] * scaling_factor
        
        if ingredients_with_nutrition == 0:
            return None
        
        return {
            'total_calories': total_calories,
            'total_protein_g': total_protein,
            'total_carbs_g': total_carbs,
            'total_fat_g': total_fat,
            'scaling_factor': scaling_factor,
            'ingredients_with_nutrition': ingredients_with_nutrition,
            'nutrition_coverage': ingredients_with_nutrition / len(ingredients)
        }
    
    def convert_recipe_units(self, recipe: ScaledRecipe, target_system: str) -> ScaledRecipe:
        """Convert recipe to different unit system."""
        converted_ingredients = []
        
        for ingredient in recipe.scaled_ingredients:
            if ingredient.scaled_quantity and ingredient.scaled_unit:
                converted_quantity, converted_unit = self._convert_to_unit_system(
                    ingredient.scaled_quantity, ingredient.scaled_unit, target_system
                )
                
                # Create new scaling options with target system
                options = ScalingOptions(
                    unit_system=target_system,
                    precision=recipe.scaling_options.precision,
                    round_to_nice_numbers=recipe.scaling_options.round_to_nice_numbers
                )
                
                # Format for display
                display_quantity, display_unit = self._format_quantity_for_display(
                    converted_quantity, converted_unit, options
                )
                
                converted_ingredient = ScaledIngredient(
                    original_ingredient=ingredient.original_ingredient,
                    scaled_quantity=converted_quantity,
                    scaled_unit=converted_unit,
                    display_quantity=display_quantity,
                    display_unit=display_unit,
                    scaling_factor=ingredient.scaling_factor,
                    conversion_applied=True,
                    scaling_notes=ingredient.scaling_notes + [f"Converted to {target_system}"],
                    fraction_display=self._to_fraction_display(converted_quantity),
                    decimal_display=f"{converted_quantity:.{options.precision}f}",
                    alternative_units=self._generate_alternative_units(
                        converted_quantity, converted_unit, target_system
                    )
                )
                
                converted_ingredients.append(converted_ingredient)
            else:
                converted_ingredients.append(ingredient)
        
        # Update scaling options
        new_options = ScalingOptions(
            target_servings=recipe.scaling_options.target_servings,
            scale_factor=recipe.scaling_options.scale_factor,
            dietary_modifications=recipe.scaling_options.dietary_modifications,
            unit_system=target_system,
            precision=recipe.scaling_options.precision,
            round_to_nice_numbers=recipe.scaling_options.round_to_nice_numbers,
            scale_cooking_times=recipe.scaling_options.scale_cooking_times,
            scale_temperatures=recipe.scaling_options.scale_temperatures
        )
        
        return ScaledRecipe(
            original_servings=recipe.original_servings,
            target_servings=recipe.target_servings,
            scaling_factor=recipe.scaling_factor,
            scaled_ingredients=converted_ingredients,
            scaled_instructions=recipe.scaled_instructions,
            scaled_cooking_times=recipe.scaled_cooking_times,
            scaled_temperatures=recipe.scaled_temperatures,
            scaling_options=new_options,
            scaling_notes=recipe.scaling_notes + [f"Converted to {target_system} units"],
            nutritional_scaling=recipe.nutritional_scaling
        )
    
    def export_scaled_recipe(self, recipe: ScaledRecipe, format: str = "json") -> str:
        """Export scaled recipe in specified format."""
        if format == "json":
            return json.dumps(asdict(recipe), indent=2, default=str)
        elif format == "text":
            return self._format_recipe_as_text(recipe)
        elif format == "markdown":
            return self._format_recipe_as_markdown(recipe)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_recipe_as_text(self, recipe: ScaledRecipe) -> str:
        """Format recipe as plain text."""
        lines = []
        
        # Header
        if recipe.target_servings:
            lines.append(f"Recipe for {recipe.target_servings} servings")
        if recipe.scaling_factor != 1.0:
            lines.append(f"Scaled by factor of {recipe.scaling_factor:.2f}")
        lines.append("")
        
        # Ingredients
        lines.append("Ingredients:")
        lines.append("-" * 20)
        for ingredient in recipe.scaled_ingredients:
            if ingredient.display_quantity:
                lines.append(f"• {ingredient.display_quantity} {ingredient.display_unit} {ingredient.original_ingredient.ingredient_name}")
            else:
                lines.append(f"• {ingredient.original_ingredient.ingredient_name}")
        lines.append("")
        
        # Instructions
        if recipe.scaled_instructions:
            lines.append("Instructions:")
            lines.append("-" * 20)
            for i, instruction in enumerate(recipe.scaled_instructions, 1):
                lines.append(f"{i}. {instruction}")
            lines.append("")
        
        # Notes
        if recipe.scaling_notes:
            lines.append("Scaling Notes:")
            lines.append("-" * 20)
            for note in recipe.scaling_notes:
                lines.append(f"• {note}")
        
        return "\n".join(lines)
    
    def _format_recipe_as_markdown(self, recipe: ScaledRecipe) -> str:
        """Format recipe as Markdown."""
        lines = []
        
        # Header
        if recipe.target_servings:
            lines.append(f"# Recipe for {recipe.target_servings} servings")
        if recipe.scaling_factor != 1.0:
            lines.append(f"*Scaled by factor of {recipe.scaling_factor:.2f}*")
        lines.append("")
        
        # Ingredients
        lines.append("## Ingredients")
        lines.append("")
        for ingredient in recipe.scaled_ingredients:
            if ingredient.display_quantity:
                lines.append(f"- {ingredient.display_quantity} {ingredient.display_unit} {ingredient.original_ingredient.ingredient_name}")
            else:
                lines.append(f"- {ingredient.original_ingredient.ingredient_name}")
        lines.append("")
        
        # Instructions
        if recipe.scaled_instructions:
            lines.append("## Instructions")
            lines.append("")
            for i, instruction in enumerate(recipe.scaled_instructions, 1):
                lines.append(f"{i}. {instruction}")
            lines.append("")
        
        # Notes
        if recipe.scaling_notes:
            lines.append("## Scaling Notes")
            lines.append("")
            for note in recipe.scaling_notes:
                lines.append(f"- {note}")
        
        return "\n".join(lines)


def main():
    """Main recipe scaling script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recipe scaling and conversion')
    parser.add_argument('--recipe', '-r', required=True, help='Recipe JSON file')
    parser.add_argument('--servings', '-s', type=int, help='Target servings')
    parser.add_argument('--scale', type=float, help='Scale factor')
    parser.add_argument('--units', choices=['metric', 'imperial', 'us'], default='metric', help='Unit system')
    parser.add_argument('--dietary', nargs='*', choices=['vegan', 'gluten_free', 'low_sodium', 'keto', 'paleo'], help='Dietary modifications')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--format', choices=['json', 'text', 'markdown'], default='json', help='Output format')
    
    args = parser.parse_args()
    
    # Load recipe
    with open(args.recipe, 'r') as f:
        recipe_data = json.load(f)
    
    # Create scaling options
    options = ScalingOptions(
        target_servings=args.servings,
        scale_factor=args.scale,
        unit_system=args.units,
        dietary_modifications=args.dietary or []
    )
    
    # Initialize scaler
    scaler = RecipeScaler()
    
    # Scale recipe (this is a simplified example)
    # In practice, you'd need to parse the recipe data properly
    print("Recipe scaling functionality ready!")
    print(f"Would scale recipe to {args.servings} servings using {args.units} units")
    
    if args.dietary:
        print(f"Would apply dietary modifications: {', '.join(args.dietary)}")
    
    return 0


if __name__ == "__main__":
    exit(main())