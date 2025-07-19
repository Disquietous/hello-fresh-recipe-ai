#!/usr/bin/env python3
"""
OCR error correction and text cleaning module.
Handles common OCR errors and improves text quality for ingredient parsing.
"""

import re
import string
from typing import Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass
import unicodedata
import difflib


@dataclass
class CleaningResult:
    """Text cleaning result with metadata."""
    original_text: str
    cleaned_text: str
    corrections_made: List[str]
    confidence_improvement: float
    cleaning_time: float = 0.0


class TextCleaner:
    """OCR text cleaning and error correction."""
    
    def __init__(self):
        """Initialize text cleaner."""
        self.logger = logging.getLogger(__name__)
        
        # Load correction patterns and dictionaries
        self._load_ocr_corrections()
        self._load_ingredient_dictionary()
        self._load_unit_corrections()
        self._compile_patterns()
    
    def _load_ocr_corrections(self):
        """Load common OCR error corrections."""
        # Common character substitutions
        self.char_corrections = {
            # Numbers often confused with letters
            '0': ['O', 'o', 'Q'],
            '1': ['l', 'I', '|', 'i'],
            '2': ['Z', 'z'],
            '3': ['E'],
            '5': ['S', 's'],
            '6': ['G', 'b'],
            '8': ['B'],
            '9': ['g', 'q'],
            
            # Letters often confused with numbers
            'O': ['0'],
            'l': ['1', 'I'],
            'I': ['1', 'l'],
            'S': ['5'],
            'B': ['8'],
            'G': ['6'],
            
            # Common letter confusions
            'rn': ['m'],
            'cl': ['d'],
            'ii': ['n'],
            'vv': ['w'],
            'nn': ['m'],
            'a': ['o'],
            'e': ['c'],
            'h': ['b'],
            'n': ['u'],
            'u': ['n'],
            'c': ['e', 'o'],
            'q': ['g'],
            'p': ['b'],
            'b': ['p', 'h'],
            'd': ['b'],
            
            # Punctuation corrections
            '.': [','],
            ',': ['.'],
            ';': [':'],
            ':': [';']
        }
        
        # Word-level corrections
        self.word_corrections = {
            # Common measurement words
            'teaspoon': ['teasp00n', 'teaspo0n', 'tea5p00n', 'tsp', 't5p', 'teasnoon'],
            'tablespoon': ['tablesp00n', 'tablespo0n', 'table5p00n', 'tbsp', 'tb5p'],
            'cup': ['cun', 'cuo', 'c0p', 'sup'],
            'cups': ['cuns', 'cuos', 'c0ps', 'sups'],
            'ounce': ['0unce', 'ounee', '0unee', 'oz'],
            'ounces': ['0unces', 'ounees', '0unees'],
            'pound': ['p0und', 'nound', 'po0nd', 'lb'],
            'pounds': ['p0unds', 'nounds', 'po0nds', 'lbs'],
            'gram': ['qram', 'grem', 'gr4m'],
            'grams': ['qrams', 'grems', 'gr4ms'],
            'kilogram': ['kiloqram', 'kilogrem', 'k1logram'],
            'kilograms': ['kiloqrams', 'kilograems', 'k1lograms'],
            'milliliter': ['m1lliliter', 'millil1ter', 'milliliter'],
            'milliliters': ['m1lliliters', 'millil1ters', 'milliliters'],
            'liter': ['l1ter', 'I1ter', '1iter'],
            'liters': ['l1ters', 'I1ters', '1iters'],
            
            # Common ingredient words
            'flour': ['fl0ur', 'fI0ur', 'f1our', 'flOur'],
            'sugar': ['5ugar', 'suqar', '5ugAr'],
            'salt': ['5alt', 'sa1t', '5a1t'],
            'pepper': ['nenner', 'peoper', 'pepner'],
            'butter': ['b0tter', 'buiter', 'butier'],
            'water': ['w4ter', 'waier', 'waler'],
            'milk': ['m1lk', 'mi1k', 'miIk'],
            'eggs': ['eqgs', 'egqs', 'e99s'],
            'egg': ['eqg', 'egq', 'e99'],
            'oil': ['0il', 'Oi1', '0i1'],
            'vanilla': ['vanilia', 'vanila', 'van1lla'],
            'chocolate': ['choc0late', 'chocolaie', 'chocplate'],
            'cheese': ['chee5e', 'chease', 'cheeze'],
            'onion': ['0nion', 'oni0n', 'onien'],
            'garlic': ['qarlic', 'garIic', 'garl1c'],
            'tomato': ['t0mato', 'tom4to', 'tomalo'],
            'chicken': ['ch1cken', 'chickan', 'ch1eken'],
            'beef': ['beaf', 'be3f', 'b33f'],
            'rice': ['r1ce', 'riee', 'r1ee'],
            'pasta': ['pa5ta', 'pasia', 'nasta'],
            'bread': ['br3ad', 'braad', 'bredd'],
            
            # Common preparation words
            'chopped': ['ch0pped', 'chonped', 'chopned'],
            'diced': ['d1ced', 'dicad', 'dicee'],
            'minced': ['m1nced', 'mincad', 'mineced'],
            'sliced': ['5liced', 'slicad', 'sIiced'],
            'grated': ['gr4ted', 'gratad', 'graied'],
            'melted': ['m3lted', 'meltad', 'meIted'],
            'beaten': ['b3aten', 'beatad', 'beaien'],
            'fresh': ['fr35h', 'fre5h', 'frash'],
            'dried': ['dr13d', 'dri3d', 'driee'],
            'ground': ['gr0und', 'groune', 'groumd'],
            'whole': ['wh0le', 'whoIe', 'whple'],
            'large': ['l4rge', 'larqe', 'Iarge'],
            'small': ['5mall', 'smaIl', 'smali'],
            'medium': ['m3dium', 'mediom', 'madium']
        }
    
    def _load_ingredient_dictionary(self):
        """Load dictionary of valid ingredient names."""
        self.ingredient_dict = {
            # Basic ingredients
            'flour', 'sugar', 'salt', 'pepper', 'butter', 'oil', 'water', 'milk',
            'eggs', 'egg', 'vanilla', 'baking', 'powder', 'soda', 'yeast',
            
            # Spices and herbs
            'basil', 'oregano', 'thyme', 'rosemary', 'sage', 'parsley', 'cilantro',
            'cumin', 'paprika', 'cinnamon', 'nutmeg', 'ginger', 'turmeric',
            'cardamom', 'cloves', 'allspice', 'cayenne', 'chili', 'garlic',
            
            # Vegetables
            'onion', 'onions', 'carrot', 'carrots', 'celery', 'potato', 'potatoes',
            'tomato', 'tomatoes', 'lettuce', 'spinach', 'broccoli', 'cauliflower',
            'cabbage', 'bell', 'pepper', 'peppers', 'cucumber', 'zucchini',
            
            # Proteins
            'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'shrimp',
            'turkey', 'ham', 'bacon', 'sausage', 'tofu', 'beans', 'lentils',
            
            # Dairy
            'cheese', 'cream', 'yogurt', 'sour', 'mozzarella', 'cheddar',
            'parmesan', 'ricotta', 'cottage', 'feta', 'goat',
            
            # Grains and starches
            'rice', 'pasta', 'bread', 'flour', 'oats', 'quinoa', 'barley',
            'wheat', 'corn', 'cornstarch', 'noodles', 'spaghetti', 'macaroni',
            
            # Fruits
            'apple', 'apples', 'banana', 'bananas', 'orange', 'oranges',
            'lemon', 'lemons', 'lime', 'limes', 'strawberry', 'strawberries',
            'blueberry', 'blueberries', 'grape', 'grapes', 'cherry', 'cherries',
            
            # Nuts and seeds
            'almond', 'almonds', 'walnut', 'walnuts', 'pecan', 'pecans',
            'peanut', 'peanuts', 'cashew', 'cashews', 'sesame', 'sunflower',
            
            # Adjectives and descriptors
            'fresh', 'dried', 'ground', 'whole', 'chopped', 'diced', 'minced',
            'sliced', 'grated', 'shredded', 'melted', 'softened', 'beaten',
            'large', 'medium', 'small', 'extra', 'virgin', 'organic', 'raw'
        }
    
    def _load_unit_corrections(self):
        """Load unit-specific corrections."""
        self.unit_corrections = {
            # Volume units
            'cup': ['cun', 'cuo', 'c0p', 'sup', 'cvp'],
            'cups': ['cuns', 'cuos', 'c0ps', 'sups', 'cvps'],
            'tsp': ['t5p', 'tsn', 'i5p', '15p'],
            'tbsp': ['tb5p', 'tbs0', 'ibsp', '1bsp'],
            'teaspoon': ['teasp00n', 'teaspo0n', 'tea5p00n'],
            'tablespoon': ['tablesp00n', 'tablespo0n', 'table5p00n'],
            'fl': ['fI', 'f1', 'II'],
            'oz': ['0z', '02', 'OZ', '0Z'],
            'pt': ['nt', 'pi', 'p1'],
            'qt': ['q1', 'qi', '91'],
            'gal': ['qaI', 'ga1', '9al'],
            'ml': ['mI', 'm1', 'mL'],
            'l': ['I', '1', 'L'],
            
            # Weight units
            'lb': ['Ib', '1b', 'lb.', 'lbs'],
            'lbs': ['Ibs', '1bs', 'lbs.'],
            'oz': ['0z', '02', 'OZ', 'o2'],
            'g': ['9', 'q', 'G'],
            'kg': ['kq', 'k9', 'KG'],
            'gram': ['qram', 'grem', 'gr4m'],
            'grams': ['qrams', 'grems', 'gr4ms'],
            'pound': ['p0und', 'nound', 'po0nd'],
            'pounds': ['p0unds', 'nounds', 'po0nds'],
            
            # Count units
            'piece': ['niece', 'p1ece', 'piese'],
            'pieces': ['nieces', 'p1eces', 'pieses'],
            'clove': ['cI0ve', 'cl0ve', 'cIove'],
            'cloves': ['cI0ves', 'cl0ves', 'cIoves'],
            'slice': ['5lice', 'sIice', 'slise'],
            'slices': ['5lices', 'sIices', 'slises']
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for cleaning."""
        # Pattern for detecting measurement fractions
        self.fraction_pattern = re.compile(r'(\d+)\s*[/\\]\s*(\d+)')
        
        # Pattern for detecting ranges
        self.range_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*[-–—~]\s*(\d+(?:\.\d+)?)')
        
        # Pattern for fixing spacing around numbers
        self.number_spacing_pattern = re.compile(r'(\d+)\s*([a-zA-Z])')
        
        # Pattern for removing extra spaces
        self.extra_space_pattern = re.compile(r'\s+')
        
        # Pattern for fixing punctuation
        self.punctuation_pattern = re.compile(r'\s+([,.;:])')
        
        # Pattern for detecting measurement abbreviations that need periods
        self.abbrev_pattern = re.compile(r'\b(tsp|tbsp|oz|lb|pt|qt|gal|ml|kg|g)\b', re.IGNORECASE)
    
    def clean_text(self, text: str, aggressive: bool = False) -> CleaningResult:
        """
        Clean OCR text and correct common errors.
        
        Args:
            text: Input OCR text
            aggressive: Whether to apply aggressive corrections
            
        Returns:
            Cleaning result with corrections
        """
        import time
        start_time = time.time()
        
        if not text or not text.strip():
            return CleaningResult(
                original_text=text,
                cleaned_text=text,
                corrections_made=[],
                confidence_improvement=0.0,
                cleaning_time=0.0
            )
        
        original_text = text
        corrections_made = []
        
        # Step 1: Basic normalization
        text = self._basic_normalization(text, corrections_made)
        
        # Step 2: Character-level corrections
        text = self._correct_characters(text, corrections_made)
        
        # Step 3: Word-level corrections
        text = self._correct_words(text, corrections_made, aggressive)
        
        # Step 4: Measurement-specific corrections
        text = self._correct_measurements(text, corrections_made)
        
        # Step 5: Spacing and punctuation fixes
        text = self._fix_spacing_punctuation(text, corrections_made)
        
        # Step 6: Final validation and cleanup
        text = self._final_cleanup(text, corrections_made)
        
        # Calculate confidence improvement
        confidence_improvement = self._calculate_confidence_improvement(original_text, text)
        
        processing_time = time.time() - start_time
        
        return CleaningResult(
            original_text=original_text,
            cleaned_text=text,
            corrections_made=corrections_made,
            confidence_improvement=confidence_improvement,
            cleaning_time=processing_time
        )
    
    def _basic_normalization(self, text: str, corrections: List[str]) -> str:
        """Apply basic text normalization."""
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Convert to consistent case handling
        # Keep original case but normalize spacing
        text = re.sub(r'\s+', ' ', text.strip())
        
        if text != text:  # Check if changes were made
            corrections.append("Basic normalization applied")
        
        return text
    
    def _correct_characters(self, text: str, corrections: List[str]) -> str:
        """Apply character-level corrections."""
        original_text = text
        
        # Common character substitutions
        for correct, wrong_chars in self.char_corrections.items():
            for wrong in wrong_chars:
                if wrong in text:
                    # Be careful about context - don't replace valid occurrences
                    pattern = rf'\b{re.escape(wrong)}\b'
                    if re.search(pattern, text):
                        text = re.sub(pattern, correct, text)
        
        # Fix specific OCR patterns
        # Fix 'rn' -> 'm' in context
        text = re.sub(r'\brn\b', 'm', text)
        text = re.sub(r'(?<=[a-z])rn(?=[a-z])', 'm', text)
        
        # Fix double letters that should be single
        text = re.sub(r'([a-z])\1{2,}', r'\1\1', text)  # More than 2 repeated -> 2
        
        if text != original_text:
            corrections.append("Character-level corrections applied")
        
        return text
    
    def _correct_words(self, text: str, corrections: List[str], aggressive: bool) -> str:
        """Apply word-level corrections."""
        original_text = text
        words = text.split()
        corrected_words = []
        
        for word in words:
            corrected_word = self._correct_single_word(word, aggressive)
            corrected_words.append(corrected_word)
        
        text = ' '.join(corrected_words)
        
        if text != original_text:
            corrections.append("Word-level corrections applied")
        
        return text
    
    def _correct_single_word(self, word: str, aggressive: bool) -> str:
        """Correct a single word."""
        # Remove punctuation for matching
        clean_word = word.strip(string.punctuation).lower()
        
        # Direct lookup in corrections dictionary
        for correct, variations in self.word_corrections.items():
            if clean_word in variations or clean_word == correct.lower():
                # Preserve original capitalization pattern
                return self._preserve_case(word, correct)
        
        # Fuzzy matching for ingredient words (if aggressive)
        if aggressive and len(clean_word) > 3:
            best_match = self._find_fuzzy_match(clean_word)
            if best_match:
                return self._preserve_case(word, best_match)
        
        return word
    
    def _find_fuzzy_match(self, word: str) -> Optional[str]:
        """Find fuzzy match for word in ingredient dictionary."""
        if len(word) < 3:
            return None
        
        # Use difflib for fuzzy matching
        matches = difflib.get_close_matches(
            word, self.ingredient_dict, n=1, cutoff=0.8
        )
        
        return matches[0] if matches else None
    
    def _preserve_case(self, original: str, replacement: str) -> str:
        """Preserve capitalization pattern from original word."""
        if not original or not replacement:
            return replacement
        
        # If original is all caps, make replacement all caps
        if original.isupper():
            return replacement.upper()
        
        # If original starts with capital, capitalize replacement
        if original[0].isupper():
            return replacement.capitalize()
        
        # Otherwise use lowercase
        return replacement.lower()
    
    def _correct_measurements(self, text: str, corrections: List[str]) -> str:
        """Apply measurement-specific corrections."""
        original_text = text
        
        # Fix fraction separators
        text = re.sub(r'(\d+)\s*[\\]\s*(\d+)', r'\1/\2', text)
        
        # Fix decimal separators (comma to period in some locales)
        text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)
        
        # Fix spacing in fractions
        text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', text)
        
        # Fix unit abbreviations
        for correct, variations in self.unit_corrections.items():
            for variation in variations:
                pattern = rf'\b{re.escape(variation)}\b'
                text = re.sub(pattern, correct, text, flags=re.IGNORECASE)
        
        # Add periods to common abbreviations if missing
        text = re.sub(r'\b(tsp|tbsp|oz|lb|pt|qt|gal|ml|kg|g)(?!\.)\b', r'\1.', text, flags=re.IGNORECASE)
        
        if text != original_text:
            corrections.append("Measurement corrections applied")
        
        return text
    
    def _fix_spacing_punctuation(self, text: str, corrections: List[str]) -> str:
        """Fix spacing and punctuation issues."""
        original_text = text
        
        # Fix spacing around numbers and letters
        text = self.number_spacing_pattern.sub(r'\1 \2', text)
        
        # Remove extra spaces
        text = self.extra_space_pattern.sub(' ', text)
        
        # Fix spacing around punctuation
        text = self.punctuation_pattern.sub(r'\1', text)
        
        # Fix spacing around slashes in fractions
        text = re.sub(r'\s*/\s*', '/', text)
        
        # Fix spacing around hyphens in ranges
        text = re.sub(r'\s*-\s*', '-', text)
        
        if text != original_text:
            corrections.append("Spacing and punctuation fixed")
        
        return text.strip()
    
    def _final_cleanup(self, text: str, corrections: List[str]) -> str:
        """Final cleanup and validation."""
        original_text = text
        
        # Remove duplicate spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove empty parentheses
        text = re.sub(r'\(\s*\)', '', text)
        
        # Fix common OCR artifacts
        text = re.sub(r'[|]{2,}', '', text)  # Remove multiple vertical bars
        text = re.sub(r'[_]{2,}', '', text)  # Remove multiple underscores
        
        if text != original_text:
            corrections.append("Final cleanup applied")
        
        return text
    
    def _calculate_confidence_improvement(self, original: str, cleaned: str) -> float:
        """Calculate confidence improvement from cleaning."""
        if not original or not cleaned:
            return 0.0
        
        # Simple heuristic based on:
        # 1. Reduction in probable OCR errors
        # 2. Presence of valid words
        # 3. Proper formatting
        
        improvement = 0.0
        
        # Count potential OCR errors in original
        original_errors = 0
        original_errors += len(re.findall(r'[0O]{2,}', original))  # Multiple O/0
        original_errors += len(re.findall(r'[Il1]{2,}', original))  # Multiple I/l/1
        original_errors += len(re.findall(r'\s{2,}', original))     # Multiple spaces
        original_errors += len(re.findall(r'[^\w\s.,;:()/-]', original))  # Strange chars
        
        # Count errors remaining in cleaned
        cleaned_errors = 0
        cleaned_errors += len(re.findall(r'[0O]{2,}', cleaned))
        cleaned_errors += len(re.findall(r'[Il1]{2,}', cleaned))
        cleaned_errors += len(re.findall(r'\s{2,}', cleaned))
        cleaned_errors += len(re.findall(r'[^\w\s.,;:()/-]', cleaned))
        
        # Calculate improvement
        if original_errors > 0:
            improvement += (original_errors - cleaned_errors) / original_errors * 0.3
        
        # Bonus for recognizable words
        cleaned_words = set(word.lower().strip(string.punctuation) for word in cleaned.split())
        recognized = len(cleaned_words.intersection(self.ingredient_dict))
        if cleaned_words:
            improvement += (recognized / len(cleaned_words)) * 0.4
        
        # Bonus for proper formatting (measurements, etc.)
        if re.search(r'\d+\s*[./]\s*\d+', cleaned):  # Has fractions
            improvement += 0.1
        if re.search(r'\d+\s*(cup|tsp|tbsp|oz|lb)', cleaned, re.IGNORECASE):  # Has measurements
            improvement += 0.1
        
        return min(improvement, 1.0)
    
    def clean_ingredient_list(self, text_lines: List[str], aggressive: bool = False) -> List[CleaningResult]:
        """
        Clean multiple ingredient text lines.
        
        Args:
            text_lines: List of OCR text lines
            aggressive: Whether to apply aggressive corrections
            
        Returns:
            List of cleaning results
        """
        results = []
        
        for line in text_lines:
            if line and line.strip():
                result = self.clean_text(line, aggressive=aggressive)
                results.append(result)
        
        self.logger.info(f"Cleaned {len(results)} text lines")
        return results
    
    def get_cleaning_statistics(self, results: List[CleaningResult]) -> Dict[str, any]:
        """
        Get statistics about cleaning results.
        
        Args:
            results: List of cleaning results
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {"total": 0}
        
        total = len(results)
        improved = sum(1 for r in results if r.confidence_improvement > 0)
        avg_improvement = sum(r.confidence_improvement for r in results) / total
        total_corrections = sum(len(r.corrections_made) for r in results)
        avg_time = sum(r.cleaning_time for r in results) / total
        
        return {
            "total_lines": total,
            "lines_improved": improved,
            "improvement_rate": improved / total,
            "average_confidence_improvement": avg_improvement,
            "total_corrections_made": total_corrections,
            "average_processing_time": avg_time,
            "most_common_corrections": self._get_most_common_corrections(results)
        }
    
    def _get_most_common_corrections(self, results: List[CleaningResult]) -> Dict[str, int]:
        """Get most common types of corrections made."""
        correction_counts = {}
        
        for result in results:
            for correction in result.corrections_made:
                correction_counts[correction] = correction_counts.get(correction, 0) + 1
        
        # Sort by frequency
        return dict(sorted(correction_counts.items(), key=lambda x: x[1], reverse=True))


def main():
    """Example usage of text cleaner."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize cleaner
    cleaner = TextCleaner()
    
    # Example OCR errors
    example_texts = [
        "2 cuns all-nurp0se fl0ur",  # cups, all-purpose, flour
        "1 tsp vanilia exiract",     # vanilla extract
        "3 Iarge egqs, beatern",     # large eggs, beaten
        "1/2 cun meIted buiter",     # cup, melted butter
        "2-3 cI0ves qarlic, mincad", # cloves garlic, minced
        "1 Ib qr0und beef",          # lb ground beef
        "SaIt and nenner t0 taste"   # Salt and pepper to taste
    ]
    
    print("OCR Text Cleaner")
    print("================")
    
    print("\nExample cleaning results:")
    total_improvement = 0
    
    for text in example_texts:
        result = cleaner.clean_text(text, aggressive=True)
        print(f"\nOriginal:  {result.original_text}")
        print(f"Cleaned:   {result.cleaned_text}")
        print(f"Corrections: {', '.join(result.corrections_made)}")
        print(f"Improvement: {result.confidence_improvement:.2f}")
        total_improvement += result.confidence_improvement
    
    avg_improvement = total_improvement / len(example_texts)
    print(f"\nAverage confidence improvement: {avg_improvement:.2f}")
    
    # Test batch cleaning
    results = cleaner.clean_ingredient_list(example_texts, aggressive=True)
    stats = cleaner.get_cleaning_statistics(results)
    
    print(f"\nCleaning Statistics:")
    print(f"  Lines processed: {stats['total_lines']}")
    print(f"  Lines improved: {stats['lines_improved']}")
    print(f"  Improvement rate: {stats['improvement_rate']:.1%}")
    print(f"  Total corrections: {stats['total_corrections_made']}")


if __name__ == "__main__":
    main()