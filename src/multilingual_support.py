#!/usr/bin/env python3
"""
Multi-language Recipe Support Module
Provides comprehensive support for processing recipes in multiple languages
including language detection, multilingual OCR, and ingredient parsing.
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

# Add src to path
sys.path.append(str(Path(__file__).parent))

try:
    import langdetect
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from ocr_engine import OCREngine
from text_cleaner import TextCleaner
from ingredient_parser import IngredientParser


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    language: str
    confidence: float
    alternative_languages: List[Tuple[str, float]]
    detected_from: str  # 'text' or 'metadata'


@dataclass
class MultilingualText:
    """Text with language information."""
    text: str
    language: str
    confidence: float
    translation: Optional[str] = None
    original_text: Optional[str] = None


@dataclass
class MultilingualIngredient:
    """Ingredient with multilingual support."""
    ingredient_name: str
    quantity: str
    unit: str
    preparation: Optional[str]
    language: str
    confidence: float
    translations: Dict[str, str]  # language -> translated name
    normalized_name: str  # normalized/standardized name


class MultilingualRecipeProcessor:
    """Processor for multilingual recipe text detection and parsing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize multilingual recipe processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Supported languages
        self.supported_languages = self.config.get('supported_languages', [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'ru', 'zh', 'ja', 'ko'
        ])
        
        # Language-specific configurations
        self.language_configs = self._initialize_language_configs()
        
        # Initialize components
        self.ocr_engine = OCREngine()
        self.text_cleaner = TextCleaner()
        self.ingredient_parser = IngredientParser()
        
        # Language-specific parsers
        self.language_parsers = self._initialize_language_parsers()
        
        # Initialize multilingual OCR if available
        if EASYOCR_AVAILABLE:
            self.multilingual_ocr = easyocr.Reader(self.supported_languages)
        else:
            self.multilingual_ocr = None
            
        self.logger.info(f"Initialized MultilingualRecipeProcessor for languages: {self.supported_languages}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for multilingual processor."""
        logger = logging.getLogger('multilingual_recipe_processor')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_language_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize language-specific configurations."""
        configs = {
            'en': {
                'name': 'English',
                'ocr_lang': 'en',
                'decimal_separator': '.',
                'fraction_words': ['half', 'quarter', 'third'],
                'unit_synonyms': {
                    'cup': ['cups', 'c', 'C'],
                    'tablespoon': ['tablespoons', 'tbsp', 'T'],
                    'teaspoon': ['teaspoons', 'tsp', 't'],
                    'pound': ['pounds', 'lb', 'lbs'],
                    'ounce': ['ounces', 'oz']
                }
            },
            'es': {
                'name': 'Spanish',
                'ocr_lang': 'es',
                'decimal_separator': ',',
                'fraction_words': ['medio', 'media', 'cuarto', 'tercio'],
                'unit_synonyms': {
                    'taza': ['tazas'],
                    'cucharada': ['cucharadas', 'cda'],
                    'cucharadita': ['cucharaditas', 'cdta'],
                    'gramo': ['gramos', 'g', 'gr'],
                    'kilogramo': ['kilogramos', 'kg']
                }
            },
            'fr': {
                'name': 'French',
                'ocr_lang': 'fr',
                'decimal_separator': ',',
                'fraction_words': ['demi', 'quart', 'tiers'],
                'unit_synonyms': {
                    'tasse': ['tasses'],
                    'cuillère': ['cuillères', 'c.'],
                    'cuillérée': ['cuillérées'],
                    'gramme': ['grammes', 'g'],
                    'kilogramme': ['kilogrammes', 'kg']
                }
            },
            'de': {
                'name': 'German',
                'ocr_lang': 'de',
                'decimal_separator': ',',
                'fraction_words': ['halb', 'viertel', 'drittel'],
                'unit_synonyms': {
                    'tasse': ['tassen'],
                    'esslöffel': ['el', 'essl'],
                    'teelöffel': ['tl', 'teel'],
                    'gramm': ['g'],
                    'kilogramm': ['kg']
                }
            },
            'it': {
                'name': 'Italian',
                'ocr_lang': 'it',
                'decimal_separator': ',',
                'fraction_words': ['mezzo', 'mezza', 'quarto', 'terzo'],
                'unit_synonyms': {
                    'tazza': ['tazze'],
                    'cucchiaio': ['cucchiai', 'c.'],
                    'cucchiaino': ['cucchiaini', 'cc.'],
                    'grammo': ['grammi', 'g'],
                    'chilogrammo': ['chilogrammi', 'kg']
                }
            },
            'pt': {
                'name': 'Portuguese',
                'ocr_lang': 'pt',
                'decimal_separator': ',',
                'fraction_words': ['meio', 'meia', 'quarto', 'terço'],
                'unit_synonyms': {
                    'xícara': ['xícaras'],
                    'colher': ['colheres', 'c.'],
                    'colherinha': ['colherinhas'],
                    'grama': ['gramas', 'g'],
                    'quilograma': ['quilogramas', 'kg']
                }
            },
            'nl': {
                'name': 'Dutch',
                'ocr_lang': 'nl',
                'decimal_separator': ',',
                'fraction_words': ['half', 'kwart', 'derde'],
                'unit_synonyms': {
                    'kop': ['koppen'],
                    'eetlepel': ['eetlepels', 'el'],
                    'theelepel': ['theelepels', 'tl'],
                    'gram': ['g'],
                    'kilogram': ['kg']
                }
            },
            'ru': {
                'name': 'Russian',
                'ocr_lang': 'ru',
                'decimal_separator': ',',
                'fraction_words': ['половина', 'четверть', 'треть'],
                'unit_synonyms': {
                    'стакан': ['стаканы', 'ст'],
                    'столовая': ['столовых', 'ст.л'],
                    'чайная': ['чайных', 'ч.л'],
                    'грамм': ['граммы', 'г'],
                    'килограмм': ['килограммы', 'кг']
                }
            },
            'zh': {
                'name': 'Chinese',
                'ocr_lang': 'zh',
                'decimal_separator': '.',
                'fraction_words': ['半', '四分之一', '三分之一'],
                'unit_synonyms': {
                    '杯': ['杯子'],
                    '汤匙': ['大勺'],
                    '茶匙': ['小勺'],
                    '克': ['g'],
                    '公斤': ['千克', 'kg']
                }
            },
            'ja': {
                'name': 'Japanese',
                'ocr_lang': 'ja',
                'decimal_separator': '.',
                'fraction_words': ['半分', '四分の一', '三分の一'],
                'unit_synonyms': {
                    'カップ': ['カップ'],
                    '大さじ': ['大匙'],
                    '小さじ': ['小匙'],
                    'グラム': ['g'],
                    'キログラム': ['kg']
                }
            },
            'ko': {
                'name': 'Korean',
                'ocr_lang': 'ko',
                'decimal_separator': '.',
                'fraction_words': ['반', '사분의일', '삼분의일'],
                'unit_synonyms': {
                    '컵': ['컵'],
                    '큰술': ['큰스푼'],
                    '작은술': ['작은스푼'],
                    '그램': ['g'],
                    '킬로그램': ['kg']
                }
            }
        }
        
        return configs
    
    def _initialize_language_parsers(self) -> Dict[str, Any]:
        """Initialize language-specific ingredient parsers."""
        parsers = {}
        
        for lang_code in self.supported_languages:
            # Create language-specific parser configuration
            lang_config = self.language_configs.get(lang_code, {})
            
            # This would typically load language-specific parsing rules
            # For now, we'll use a simplified approach
            parsers[lang_code] = {
                'config': lang_config,
                'parser': self.ingredient_parser  # Use base parser for now
            }
        
        return parsers
    
    def detect_language(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> LanguageDetectionResult:
        """
        Detect language of text with confidence scoring.
        
        Args:
            text: Text to analyze
            metadata: Optional metadata that might contain language hints
            
        Returns:
            Language detection result
        """
        # Check metadata first
        if metadata and 'language' in metadata:
            return LanguageDetectionResult(
                language=metadata['language'],
                confidence=1.0,
                alternative_languages=[],
                detected_from='metadata'
            )
        
        # Use langdetect if available
        if LANGDETECT_AVAILABLE and text.strip():
            try:
                # Clean text for better detection
                clean_text = re.sub(r'[^\w\s]', ' ', text)
                clean_text = ' '.join(clean_text.split())
                
                if len(clean_text) > 10:  # Need minimum text length
                    detected_lang = detect(clean_text)
                    
                    # Get confidence by trying multiple times
                    detections = []
                    for _ in range(5):
                        try:
                            detections.append(detect(clean_text))
                        except LangDetectError:
                            pass
                    
                    # Calculate confidence based on consistency
                    if detections:
                        lang_counts = defaultdict(int)
                        for lang in detections:
                            lang_counts[lang] += 1
                        
                        most_common = max(lang_counts.items(), key=lambda x: x[1])
                        confidence = most_common[1] / len(detections)
                        
                        # Get alternative languages
                        alternatives = [(lang, count / len(detections)) 
                                      for lang, count in lang_counts.items() 
                                      if lang != most_common[0]]
                        alternatives.sort(key=lambda x: x[1], reverse=True)
                        
                        return LanguageDetectionResult(
                            language=most_common[0],
                            confidence=confidence,
                            alternative_languages=alternatives[:3],
                            detected_from='text'
                        )
                
            except LangDetectError:
                pass
        
        # Fallback to simple heuristics
        language = self._detect_language_heuristic(text)
        
        return LanguageDetectionResult(
            language=language,
            confidence=0.5,
            alternative_languages=[],
            detected_from='text'
        )
    
    def _detect_language_heuristic(self, text: str) -> str:
        """Simple heuristic-based language detection."""
        text_lower = text.lower()
        
        # Look for common words/patterns in different languages
        if any(word in text_lower for word in ['cup', 'cups', 'tablespoon', 'teaspoon', 'ounce', 'pound']):
            return 'en'
        elif any(word in text_lower for word in ['taza', 'cucharada', 'gramo', 'kilogramo']):
            return 'es'
        elif any(word in text_lower for word in ['tasse', 'cuillère', 'gramme', 'kilogramme']):
            return 'fr'
        elif any(word in text_lower for word in ['tasse', 'esslöffel', 'teelöffel', 'gramm']):
            return 'de'
        elif any(word in text_lower for word in ['tazza', 'cucchiaio', 'grammo', 'chilogrammo']):
            return 'it'
        elif any(word in text_lower for word in ['xícara', 'colher', 'grama', 'quilograma']):
            return 'pt'
        elif any(word in text_lower for word in ['kop', 'eetlepel', 'theelepel', 'gram']):
            return 'nl'
        elif any(char in text for char in 'абвгдежзийклмнопрстуфхцчшщъыьэюя'):
            return 'ru'
        elif any(char in text for char in '一二三四五六七八九十杯克公斤'):
            return 'zh'
        elif any(char in text for char in 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'):
            return 'ja'
        elif any(char in text for char in 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ'):
            return 'ko'
        
        # Default to English
        return 'en'
    
    def extract_multilingual_text(self, image, language_hint: Optional[str] = None) -> List[MultilingualText]:
        """
        Extract text from image with language detection and multilingual OCR.
        
        Args:
            image: Input image
            language_hint: Optional language hint
            
        Returns:
            List of multilingual text extractions
        """
        multilingual_texts = []
        
        # Use multilingual OCR if available
        if self.multilingual_ocr:
            try:
                # Extract text with EasyOCR (multilingual)
                ocr_results = self.multilingual_ocr.readtext(image)
                
                for bbox, text, confidence in ocr_results:
                    if text.strip() and confidence > 0.3:
                        # Detect language for this text
                        lang_result = self.detect_language(text)
                        
                        multilingual_texts.append(MultilingualText(
                            text=text,
                            language=lang_result.language,
                            confidence=confidence * lang_result.confidence,
                            original_text=text
                        ))
                
            except Exception as e:
                self.logger.warning(f"Multilingual OCR failed: {e}")
        
        # Fallback to standard OCR
        if not multilingual_texts:
            try:
                ocr_result = self.ocr_engine.extract_text(image)
                if ocr_result.text:
                    lang_result = self.detect_language(ocr_result.text)
                    
                    multilingual_texts.append(MultilingualText(
                        text=ocr_result.text,
                        language=lang_result.language,
                        confidence=ocr_result.confidence * lang_result.confidence,
                        original_text=ocr_result.text
                    ))
            except Exception as e:
                self.logger.warning(f"Standard OCR failed: {e}")
        
        return multilingual_texts
    
    def parse_multilingual_ingredients(self, multilingual_texts: List[MultilingualText]) -> List[MultilingualIngredient]:
        """
        Parse ingredients from multilingual text extractions.
        
        Args:
            multilingual_texts: List of multilingual text extractions
            
        Returns:
            List of parsed multilingual ingredients
        """
        multilingual_ingredients = []
        
        for text_obj in multilingual_texts:
            language = text_obj.language
            text = text_obj.text
            
            # Get language-specific parser
            parser_config = self.language_parsers.get(language, self.language_parsers['en'])
            
            # Clean text with language-specific rules
            cleaned_text = self._clean_text_by_language(text, language)
            
            # Parse ingredient
            parsed = self._parse_ingredient_by_language(cleaned_text, language)
            
            if parsed and parsed.ingredient_name:
                # Create multilingual ingredient
                multilingual_ingredient = MultilingualIngredient(
                    ingredient_name=parsed.ingredient_name,
                    quantity=parsed.quantity,
                    unit=parsed.unit,
                    preparation=parsed.preparation,
                    language=language,
                    confidence=parsed.confidence * text_obj.confidence,
                    translations={},
                    normalized_name=self._normalize_ingredient_name(parsed.ingredient_name, language)
                )
                
                # Add translations if available
                multilingual_ingredient.translations = self._get_ingredient_translations(
                    parsed.ingredient_name, language
                )
                
                multilingual_ingredients.append(multilingual_ingredient)
        
        return multilingual_ingredients
    
    def _clean_text_by_language(self, text: str, language: str) -> str:
        """Apply language-specific text cleaning."""
        # Get language config
        lang_config = self.language_configs.get(language, {})
        
        # Apply base cleaning
        cleaned_result = self.text_cleaner.clean_text(text)
        cleaned_text = cleaned_result.cleaned_text
        
        # Apply language-specific corrections
        if language in ['es', 'fr', 'it', 'pt']:
            # Latin languages - handle accented characters
            cleaned_text = self._handle_accented_characters(cleaned_text, language)
        elif language in ['ru']:
            # Cyrillic - handle character set
            cleaned_text = self._handle_cyrillic_characters(cleaned_text)
        elif language in ['zh', 'ja', 'ko']:
            # Asian languages - handle character sets
            cleaned_text = self._handle_asian_characters(cleaned_text, language)
        
        # Handle language-specific decimal separators
        decimal_sep = lang_config.get('decimal_separator', '.')
        if decimal_sep == ',':
            # Convert decimal commas to dots for parsing
            cleaned_text = re.sub(r'(\d),(\d)', r'\1.\2', cleaned_text)
        
        return cleaned_text
    
    def _handle_accented_characters(self, text: str, language: str) -> str:
        """Handle accented characters for Latin languages."""
        # Common OCR errors with accented characters
        corrections = {
            'á': ['a', 'à', 'â', 'ã'],
            'é': ['e', 'è', 'ê', 'ë'],
            'í': ['i', 'ì', 'î', 'ï'],
            'ó': ['o', 'ò', 'ô', 'õ'],
            'ú': ['u', 'ù', 'û', 'ü'],
            'ñ': ['n', 'ñ'],
            'ç': ['c', 'ç']
        }
        
        # Apply corrections (simplified)
        for correct, variants in corrections.items():
            for variant in variants:
                text = text.replace(variant, correct)
        
        return text
    
    def _handle_cyrillic_characters(self, text: str) -> str:
        """Handle Cyrillic characters for Russian."""
        # Common OCR errors with Cyrillic
        corrections = {
            'а': ['a', 'à'],
            'е': ['e', 'è'],
            'о': ['o', 'ò'],
            'р': ['p'],
            'с': ['c'],
            'у': ['y'],
            'х': ['x']
        }
        
        # Apply corrections (simplified)
        for correct, variants in corrections.items():
            for variant in variants:
                text = text.replace(variant, correct)
        
        return text
    
    def _handle_asian_characters(self, text: str, language: str) -> str:
        """Handle Asian characters for Chinese/Japanese/Korean."""
        # This would implement character set specific corrections
        # For now, return as-is
        return text
    
    def _parse_ingredient_by_language(self, text: str, language: str):
        """Parse ingredient text using language-specific rules."""
        # Get language-specific parser
        parser_config = self.language_parsers.get(language, self.language_parsers['en'])
        
        # Use base parser for now (would be enhanced with language-specific rules)
        return parser_config['parser'].parse_ingredient_line(text)
    
    def _normalize_ingredient_name(self, ingredient_name: str, language: str) -> str:
        """Normalize ingredient name to a standard form."""
        # This would map ingredients to standardized names
        # For now, return cleaned version
        return ingredient_name.lower().strip()
    
    def _get_ingredient_translations(self, ingredient_name: str, source_language: str) -> Dict[str, str]:
        """Get translations for ingredient name."""
        # This would use a translation service or dictionary
        # For now, return empty dict
        translations = {}
        
        # Simple hardcoded translations for common ingredients
        common_translations = {
            'en': {
                'flour': {'es': 'harina', 'fr': 'farine', 'de': 'mehl', 'it': 'farina'},
                'sugar': {'es': 'azúcar', 'fr': 'sucre', 'de': 'zucker', 'it': 'zucchero'},
                'salt': {'es': 'sal', 'fr': 'sel', 'de': 'salz', 'it': 'sale'},
                'butter': {'es': 'mantequilla', 'fr': 'beurre', 'de': 'butter', 'it': 'burro'},
                'milk': {'es': 'leche', 'fr': 'lait', 'de': 'milch', 'it': 'latte'},
                'egg': {'es': 'huevo', 'fr': 'œuf', 'de': 'ei', 'it': 'uovo'},
                'water': {'es': 'agua', 'fr': 'eau', 'de': 'wasser', 'it': 'acqua'}
            }
        }
        
        normalized_name = ingredient_name.lower().strip()
        
        if source_language in common_translations:
            lang_dict = common_translations[source_language]
            if normalized_name in lang_dict:
                translations = lang_dict[normalized_name]
        
        return translations
    
    def process_multilingual_recipe(self, image, language_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a recipe image with full multilingual support.
        
        Args:
            image: Input image
            language_hint: Optional language hint
            
        Returns:
            Multilingual processing result
        """
        result = {
            'detected_languages': [],
            'multilingual_texts': [],
            'multilingual_ingredients': [],
            'processing_summary': {},
            'errors': []
        }
        
        try:
            # Extract multilingual text
            multilingual_texts = self.extract_multilingual_text(image, language_hint)
            result['multilingual_texts'] = [asdict(text) for text in multilingual_texts]
            
            # Get detected languages
            detected_languages = list(set(text.language for text in multilingual_texts))
            result['detected_languages'] = detected_languages
            
            # Parse multilingual ingredients
            multilingual_ingredients = self.parse_multilingual_ingredients(multilingual_texts)
            result['multilingual_ingredients'] = [asdict(ingredient) for ingredient in multilingual_ingredients]
            
            # Processing summary
            result['processing_summary'] = {
                'total_text_regions': len(multilingual_texts),
                'total_ingredients': len(multilingual_ingredients),
                'languages_detected': len(detected_languages),
                'primary_language': detected_languages[0] if detected_languages else 'unknown',
                'avg_confidence': np.mean([text.confidence for text in multilingual_texts]) if multilingual_texts else 0.0
            }
            
        except Exception as e:
            error_msg = f"Multilingual processing error: {str(e)}"
            result['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return result
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages with their names."""
        return {code: config['name'] for code, config in self.language_configs.items()}
    
    def update_language_support(self, language_code: str, config: Dict[str, Any]):
        """Update or add support for a new language."""
        self.language_configs[language_code] = config
        
        if language_code not in self.supported_languages:
            self.supported_languages.append(language_code)
        
        # Reinitialize components if needed
        self.language_parsers[language_code] = {
            'config': config,
            'parser': self.ingredient_parser
        }
        
        self.logger.info(f"Updated language support for: {language_code}")


def main():
    """Main multilingual processing script."""
    import argparse
    import cv2
    
    parser = argparse.ArgumentParser(description='Process multilingual recipes')
    parser.add_argument('--image', '-i', required=True, help='Path to recipe image')
    parser.add_argument('--language', '-l', help='Language hint')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--supported-languages', action='store_true', help='Show supported languages')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = MultilingualRecipeProcessor()
    
    if args.supported_languages:
        print("Supported Languages:")
        for code, name in processor.get_supported_languages().items():
            print(f"  {code}: {name}")
        return 0
    
    # Process image
    try:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image: {args.image}")
            return 1
        
        result = processor.process_multilingual_recipe(image, args.language)
        
        # Print results
        print(f"\nMultilingual Recipe Processing Results:")
        print(f"=====================================")
        print(f"Detected Languages: {', '.join(result['detected_languages'])}")
        print(f"Primary Language: {result['processing_summary']['primary_language']}")
        print(f"Text Regions: {result['processing_summary']['total_text_regions']}")
        print(f"Ingredients: {result['processing_summary']['total_ingredients']}")
        print(f"Average Confidence: {result['processing_summary']['avg_confidence']:.3f}")
        
        if result['multilingual_ingredients']:
            print(f"\nExtracted Ingredients:")
            for ingredient in result['multilingual_ingredients']:
                print(f"  - {ingredient['quantity']} {ingredient['unit']} {ingredient['ingredient_name']} ({ingredient['language']})")
                if ingredient['translations']:
                    print(f"    Translations: {ingredient['translations']}")
        
        if result['errors']:
            print(f"\nErrors:")
            for error in result['errors']:
                print(f"  - {error}")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import numpy as np
    exit(main())