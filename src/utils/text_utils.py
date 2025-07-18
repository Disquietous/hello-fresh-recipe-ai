"""
Text processing utilities for HelloFresh Recipe AI project.
Functions for text preprocessing, ingredient parsing, and validation.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from fuzzywuzzy import fuzz
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class IngredientParser:
    """Parser for extracting structured data from ingredient text."""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
        # Comprehensive unit mappings
        self.units = {
            'volume': {
                'ml': ['ml', 'milliliter', 'milliliters', 'mL'],
                'l': ['l', 'liter', 'liters', 'litre', 'litres', 'L'],
                'cup': ['cup', 'cups', 'c'],
                'tbsp': ['tbsp', 'tablespoon', 'tablespoons', 'T', 'tbs'],
                'tsp': ['tsp', 'teaspoon', 'teaspoons', 't'],
                'fl_oz': ['fl oz', 'fluid ounce', 'fluid ounces', 'fl. oz.'],
                'pint': ['pint', 'pints', 'pt'],
                'quart': ['quart', 'quarts', 'qt'],
                'gallon': ['gallon', 'gallons', 'gal']
            },
            'weight': {
                'g': ['g', 'gram', 'grams', 'gr'],
                'kg': ['kg', 'kilogram', 'kilograms', 'kilo', 'kilos'],
                'oz': ['oz', 'ounce', 'ounces', 'oz.'],
                'lb': ['lb', 'lbs', 'pound', 'pounds', '#']
            },
            'count': {
                'piece': ['piece', 'pieces', 'pc', 'pcs'],
                'item': ['item', 'items'],
                'clove': ['clove', 'cloves'],
                'slice': ['slice', 'slices'],
                'whole': ['whole'],
                'half': ['half', '1/2'],
                'quarter': ['quarter', '1/4']
            }
        }
        
        # Create reverse mapping for unit recognition
        self.unit_map = {}
        for category, units in self.units.items():
            for standard, variants in units.items():
                for variant in variants:
                    self.unit_map[variant.lower()] = (category, standard)
        
        # Common ingredients database
        self.ingredients = [
            # Proteins
            'chicken', 'beef', 'pork', 'turkey', 'fish', 'salmon', 'tuna', 'shrimp', 'eggs', 'egg',
            # Vegetables
            'onion', 'onions', 'garlic', 'tomato', 'tomatoes', 'carrot', 'carrots', 'potato', 'potatoes',
            'bell pepper', 'peppers', 'broccoli', 'spinach', 'lettuce', 'cucumber', 'mushrooms', 'mushroom',
            'celery', 'corn', 'peas', 'beans', 'green beans', 'zucchini', 'squash', 'eggplant',
            # Fruits
            'apple', 'apples', 'banana', 'bananas', 'orange', 'oranges', 'lemon', 'lemons', 'lime', 'limes',
            'strawberries', 'blueberries', 'grapes', 'avocado', 'avocados',
            # Grains & Starches
            'rice', 'pasta', 'bread', 'flour', 'oats', 'quinoa', 'barley', 'noodles',
            # Dairy
            'milk', 'cream', 'cheese', 'butter', 'yogurt', 'sour cream', 'heavy cream',
            # Pantry Items
            'oil', 'olive oil', 'vegetable oil', 'salt', 'pepper', 'sugar', 'honey', 'vinegar',
            'soy sauce', 'hot sauce', 'ketchup', 'mayonnaise', 'mustard',
            # Herbs & Spices
            'basil', 'oregano', 'thyme', 'rosemary', 'parsley', 'cilantro', 'paprika', 'cumin',
            'cinnamon', 'ginger', 'garlic powder', 'onion powder'
        ]
    
    def normalize_unit(self, unit_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Normalize unit text to standard form.
        
        Args:
            unit_text (str): Raw unit text
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (category, standardized_unit)
        """
        unit_lower = unit_text.lower().strip()
        return self.unit_map.get(unit_lower, (None, None))
    
    def extract_amount(self, text: str) -> Tuple[Optional[str], str]:
        """
        Extract amount from text.
        
        Args:
            text (str): Input text
            
        Returns:
            Tuple[Optional[str], str]: (amount, remaining_text)
        """
        # Patterns for different amount formats
        patterns = [
            r'(\d+(?:\.\d+)?(?:/\d+)?)',  # 1, 1.5, 1/2
            r'(\d+\s*-\s*\d+)',           # 1-2
            r'(a few|several|some)',       # Qualitative amounts
            r'(half|quarter|third)',       # Word fractions
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = match.group(1)
                remaining = text[:match.start()] + text[match.end():]
                return amount.strip(), remaining.strip()
        
        return None, text
    
    def extract_unit(self, text: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        Extract unit from text.
        
        Args:
            text (str): Input text
            
        Returns:
            Tuple[Optional[str], Optional[str], str]: (category, unit, remaining_text)
        """
        words = text.split()
        
        for i, word in enumerate(words):
            category, standard_unit = self.normalize_unit(word)
            if category:
                remaining_words = words[:i] + words[i+1:]
                remaining_text = ' '.join(remaining_words)
                return category, standard_unit, remaining_text
        
        return None, None, text
    
    def identify_ingredient(self, text: str) -> Tuple[Optional[str], float]:
        """
        Identify ingredient name from text using fuzzy matching.
        
        Args:
            text (str): Input text
            
        Returns:
            Tuple[Optional[str], float]: (ingredient_name, confidence_score)
        """
        text_lower = text.lower().strip()
        
        # Remove common cooking terms
        cooking_terms = ['fresh', 'dried', 'chopped', 'sliced', 'diced', 'minced', 'ground',
                        'large', 'medium', 'small', 'organic', 'raw', 'cooked']
        
        words = word_tokenize(text_lower)
        filtered_words = [w for w in words if w not in self.stop_words and w not in cooking_terms]
        cleaned_text = ' '.join(filtered_words)
        
        best_match = None
        best_score = 0
        
        for ingredient in self.ingredients:
            # Check for exact substring match
            if ingredient in cleaned_text:
                score = len(ingredient) / len(cleaned_text)
                if score > best_score:
                    best_match = ingredient
                    best_score = score
            
            # Check fuzzy match
            fuzzy_score = fuzz.partial_ratio(ingredient, cleaned_text) / 100
            if fuzzy_score > 0.8 and fuzzy_score > best_score:
                best_match = ingredient
                best_score = fuzzy_score
        
        return best_match, best_score
    
    def parse_ingredient_line(self, text: str) -> Dict:
        """
        Parse a complete ingredient line with enhanced parsing.
        
        Args:
            text (str): Ingredient line text
            
        Returns:
            Dict: Parsed ingredient information
        """
        original_text = text.strip()
        working_text = original_text.lower()
        
        # Clean text first
        working_text = self._clean_text(working_text)
        
        # Extract amount
        amount, working_text = self.extract_amount(working_text)
        
        # Extract unit
        unit_category, unit, working_text = self.extract_unit(working_text)
        
        # Identify ingredient
        ingredient_name, confidence = self.identify_ingredient(working_text)
        
        # Enhanced validation
        is_valid = self._validate_parsing_result(amount, unit, ingredient_name, confidence)
        
        return {
            'original_text': original_text,
            'amount': self._normalize_amount(amount) if amount else None,
            'unit': unit,
            'unit_category': unit_category,
            'ingredient_name': self._normalize_ingredient_name(ingredient_name) if ingredient_name else None,
            'confidence_score': confidence,
            'parsed_successfully': is_valid,
            'parsing_quality': self._assess_parsing_quality(amount, unit, ingredient_name, confidence)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better parsing."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Handle common OCR errors
        text = text.replace('|', 'l')  # Common OCR error
        text = text.replace('0', 'o')  # In ingredient names
        text = re.sub(r'[^\w\s\./\-]', ' ', text)  # Remove special chars except useful ones
        
        return text.strip()
    
    def _normalize_amount(self, amount: str) -> str:
        """Normalize amount format."""
        if not amount:
            return None
        
        # Convert common fractions
        fraction_map = {
            '1/2': '0.5',
            '1/3': '0.33',
            '2/3': '0.67',
            '1/4': '0.25',
            '3/4': '0.75',
            '1/8': '0.125',
            '3/8': '0.375',
            '5/8': '0.625',
            '7/8': '0.875'
        }
        
        for fraction, decimal in fraction_map.items():
            if fraction in amount:
                amount = amount.replace(fraction, decimal)
        
        return amount.strip()
    
    def _normalize_ingredient_name(self, ingredient: str) -> str:
        """Normalize ingredient name."""
        if not ingredient:
            return None
        
        # Capitalize first letter of each word
        ingredient = ' '.join(word.capitalize() for word in ingredient.split())
        
        # Handle special cases
        special_cases = {
            'Salt': 'Salt',
            'Black Pepper': 'Black Pepper',
            'Olive Oil': 'Olive Oil',
            'Soy Sauce': 'Soy Sauce'
        }
        
        for special, correct in special_cases.items():
            if special.lower() in ingredient.lower():
                ingredient = correct
        
        return ingredient
    
    def _validate_parsing_result(self, amount: str, unit: str, ingredient: str, confidence: float) -> bool:
        """Validate if parsing result makes sense."""
        # Must have at least ingredient name
        if not ingredient or confidence < 0.3:
            return False
        
        # If we have amount, it should be reasonable
        if amount:
            try:
                amount_val = float(amount.replace('/', '.'))
                if amount_val <= 0 or amount_val > 1000:  # Reasonable bounds
                    return False
            except ValueError:
                # Non-numeric amounts like "a few" are ok
                pass
        
        # If we have unit without amount, that's suspicious
        if unit and not amount:
            return False
        
        return True
    
    def _assess_parsing_quality(self, amount: str, unit: str, ingredient: str, confidence: float) -> str:
        """Assess overall quality of parsing."""
        if not ingredient:
            return 'poor'
        
        score = confidence
        
        # Bonus for having amount and unit
        if amount and unit:
            score += 0.2
        elif amount:
            score += 0.1
        
        # Penalty for very short ingredient names
        if ingredient and len(ingredient.split()) == 1 and len(ingredient) < 4:
            score -= 0.2
        
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        else:
            return 'poor'


class TextPreprocessor:
    """Text preprocessing utilities for OCR enhancement."""
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance image contrast for better OCR."""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l)
            
            # Merge channels
            enhanced_lab = cv2.merge([enhanced_l, a, b])
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_bgr
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    @staticmethod
    def remove_noise(image: np.ndarray) -> np.ndarray:
        """Remove noise from image."""
        if len(image.shape) == 3:
            # Bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(image, 9, 75, 75)
        else:
            # For grayscale, use Gaussian blur
            filtered = cv2.GaussianBlur(image, (3, 3), 0)
        return filtered
    
    @staticmethod
    def binarize_image(image: np.ndarray) -> np.ndarray:
        """Convert image to binary for better OCR."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Use adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary
    
    @staticmethod
    def dilate_text(image: np.ndarray, kernel_size: Tuple[int, int] = (2, 2)) -> np.ndarray:
        """Dilate text to improve connectivity."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        dilated = cv2.dilate(image, kernel, iterations=1)
        return dilated
    
    @staticmethod
    def correct_skew(image: np.ndarray) -> np.ndarray:
        """Correct text skew in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find lines using HoughLines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = theta * 180 / np.pi - 90
                angles.append(angle)
            
            # Calculate median angle
            median_angle = np.median(angles)
            
            # Rotate image
            if abs(median_angle) > 0.5:  # Only rotate if significant skew
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), 
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
                return rotated
        
        return image


class RecipeDataValidator:
    """Validator for recipe data quality and completeness."""
    
    def __init__(self):
        self.required_fields = ['ingredient_name', 'amount']
        self.optional_fields = ['unit', 'unit_category']
    
    def validate_ingredient(self, ingredient_data: Dict) -> Dict:
        """
        Validate individual ingredient data.
        
        Args:
            ingredient_data (Dict): Ingredient information
            
        Returns:
            Dict: Validation results
        """
        issues = []
        warnings = []
        
        # Check required fields
        for field in self.required_fields:
            if not ingredient_data.get(field):
                issues.append(f"Missing required field: {field}")
        
        # Validate amount format
        amount = ingredient_data.get('amount')
        if amount:
            if not re.match(r'^[\d\.\s/\-]+$', str(amount).replace('a few', '').replace('some', '')):
                warnings.append(f"Unusual amount format: {amount}")
        
        # Check confidence score
        confidence = ingredient_data.get('confidence_score', 0)
        if confidence < 0.7:
            warnings.append(f"Low confidence ingredient identification: {confidence:.2f}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'confidence_score': confidence
        }
    
    def validate_recipe(self, recipe_data: List[Dict]) -> Dict:
        """
        Validate complete recipe data.
        
        Args:
            recipe_data (List[Dict]): List of ingredient data
            
        Returns:
            Dict: Recipe validation results
        """
        total_ingredients = len(recipe_data)
        valid_ingredients = 0
        all_issues = []
        all_warnings = []
        
        for i, ingredient in enumerate(recipe_data):
            validation = self.validate_ingredient(ingredient)
            
            if validation['is_valid']:
                valid_ingredients += 1
            
            # Add ingredient index to issues/warnings
            for issue in validation['issues']:
                all_issues.append(f"Ingredient {i+1}: {issue}")
            
            for warning in validation['warnings']:
                all_warnings.append(f"Ingredient {i+1}: {warning}")
        
        validation_score = valid_ingredients / total_ingredients if total_ingredients > 0 else 0
        
        return {
            'total_ingredients': total_ingredients,
            'valid_ingredients': valid_ingredients,
            'validation_score': validation_score,
            'overall_quality': 'high' if validation_score >= 0.8 else 'medium' if validation_score >= 0.6 else 'low',
            'issues': all_issues,
            'warnings': all_warnings
        }


def save_training_data(ingredients_data: List[Dict], output_path: str):
    """
    Save ingredient data in format suitable for training.
    
    Args:
        ingredients_data (List[Dict]): Processed ingredient data
        output_path (str): Path to save training data
    """
    training_data = {
        'version': '1.0',
        'data_type': 'ingredient_detection',
        'total_samples': len(ingredients_data),
        'ingredients': ingredients_data,
        'statistics': {
            'units_found': len(set(ing.get('unit') for ing in ingredients_data if ing.get('unit'))),
            'unique_ingredients': len(set(ing.get('ingredient_name') for ing in ingredients_data if ing.get('ingredient_name'))),
            'avg_confidence': sum(ing.get('confidence_score', 0) for ing in ingredients_data) / len(ingredients_data) if ingredients_data else 0
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)


def load_ingredient_database(file_path: str) -> List[str]:
    """
    Load custom ingredient database from file.
    
    Args:
        file_path (str): Path to ingredient database file
        
    Returns:
        List[str]: List of ingredient names
    """
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
                return data.get('ingredients', [])
            else:
                return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Ingredient database file not found: {file_path}")
        return []