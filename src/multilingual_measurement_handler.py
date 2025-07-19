#!/usr/bin/env python3
"""
Multilingual and Measurement System Handler
Handles various languages and measurement systems in recipe text:
- Language detection and OCR optimization
- Measurement system detection and conversion
- Cultural recipe format understanding
- Multilingual ingredient parsing
- Cross-language recipe normalization
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
from enum import Enum
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE = "zh"
    ARABIC = "ar"
    HINDI = "hi"
    DUTCH = "nl"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    DANISH = "da"
    FINNISH = "fi"
    POLISH = "pl"
    TURKISH = "tr"
    GREEK = "el"


class MeasurementSystem(Enum):
    """Measurement systems."""
    METRIC = "metric"
    IMPERIAL = "imperial"
    MIXED = "mixed"
    TRADITIONAL_ASIAN = "traditional_asian"
    TRADITIONAL_EUROPEAN = "traditional_european"


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    primary_language: Language
    confidence: float
    secondary_languages: List[Tuple[Language, float]]
    script_type: str
    text_direction: str


@dataclass
class MeasurementSystemResult:
    """Result of measurement system detection."""
    primary_system: MeasurementSystem
    confidence: float
    detected_units: List[str]
    conversion_needed: bool
    cultural_context: str


@dataclass
class MultilingualParsingResult:
    """Result of multilingual ingredient parsing."""
    ingredient_name: str
    ingredient_name_en: str
    quantity: str
    unit: str
    unit_normalized: str
    preparation: str
    language: Language
    measurement_system: MeasurementSystem
    confidence: float
    cultural_notes: List[str]


class MultilingualMeasurementHandler:
    """Handler for multilingual text and measurement systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multilingual handler."""
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Load language resources
        self.language_patterns = self._load_language_patterns()
        self.ingredient_dictionaries = self._load_ingredient_dictionaries()
        self.measurement_units = self._load_measurement_units()
        self.cultural_contexts = self._load_cultural_contexts()
        
        # OCR language configurations
        self.ocr_language_configs = self._load_ocr_language_configs()
        
        # Conversion factors
        self.conversion_factors = self._load_conversion_factors()
        
        self.logger.info("Multilingual Measurement Handler initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the handler."""
        logger = logging.getLogger('multilingual_measurement_handler')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        Detect the language of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language detection result
        """
        text_lower = text.lower()
        
        # Score each language based on pattern matching
        language_scores = {}
        
        for lang, patterns in self.language_patterns.items():
            score = 0
            pattern_matches = 0
            
            for pattern_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if pattern in text_lower:
                        if pattern_type == 'ingredients':
                            score += 3
                        elif pattern_type == 'units':
                            score += 2
                        elif pattern_type == 'keywords':
                            score += 1
                        pattern_matches += 1
            
            # Normalize by text length and pattern diversity
            if len(text_lower) > 0:
                language_scores[lang] = (score / len(text_lower.split())) * min(pattern_matches, 5)
        
        # Find primary language
        if language_scores:
            primary_lang = max(language_scores.items(), key=lambda x: x[1])
            primary_language = Language(primary_lang[0])
            confidence = min(primary_lang[1], 1.0)
            
            # Get secondary languages
            secondary_languages = []
            for lang, score in sorted(language_scores.items(), key=lambda x: x[1], reverse=True)[1:3]:
                if score > 0.1:
                    secondary_languages.append((Language(lang), min(score, 1.0)))
        else:
            primary_language = Language.ENGLISH
            confidence = 0.1
            secondary_languages = []
        
        # Determine script type and text direction
        script_type = self._detect_script_type(text)
        text_direction = self._detect_text_direction(text, primary_language)
        
        return LanguageDetectionResult(
            primary_language=primary_language,
            confidence=confidence,
            secondary_languages=secondary_languages,
            script_type=script_type,
            text_direction=text_direction
        )
    
    def detect_measurement_system(self, text: str, language: Language) -> MeasurementSystemResult:
        """
        Detect the measurement system used in the text.
        
        Args:
            text: Text to analyze
            language: Detected language
            
        Returns:
            Measurement system detection result
        """
        text_lower = text.lower()
        
        # Count units by system
        system_scores = {
            MeasurementSystem.METRIC: 0,
            MeasurementSystem.IMPERIAL: 0,
            MeasurementSystem.TRADITIONAL_ASIAN: 0,
            MeasurementSystem.TRADITIONAL_EUROPEAN: 0
        }
        
        detected_units = []
        
        for system, units in self.measurement_units.items():
            for unit in units:
                count = text_lower.count(unit.lower())
                if count > 0:
                    system_scores[MeasurementSystem(system)] += count
                    detected_units.extend([unit] * count)
        
        # Determine primary system
        if max(system_scores.values()) > 0:
            primary_system = max(system_scores.items(), key=lambda x: x[1])[0]
            confidence = system_scores[primary_system] / sum(system_scores.values())
        else:
            # Default based on language/region
            primary_system = self._get_default_measurement_system(language)
            confidence = 0.3
        
        # Check if conversion is needed
        conversion_needed = self._should_convert_measurements(primary_system, language)
        
        # Get cultural context
        cultural_context = self._get_cultural_context(language, primary_system)
        
        return MeasurementSystemResult(
            primary_system=primary_system,
            confidence=confidence,
            detected_units=detected_units,
            conversion_needed=conversion_needed,
            cultural_context=cultural_context
        )
    
    def parse_multilingual_ingredient(self, text: str, language: Language, 
                                    measurement_system: MeasurementSystem) -> MultilingualParsingResult:
        """
        Parse ingredient text in multiple languages.
        
        Args:
            text: Ingredient text to parse
            language: Detected language
            measurement_system: Detected measurement system
            
        Returns:
            Multilingual parsing result
        """
        # Clean and normalize text
        cleaned_text = self._clean_ingredient_text(text, language)
        
        # Extract quantity and unit
        quantity, unit, remaining_text = self._extract_quantity_and_unit(cleaned_text, language, measurement_system)
        
        # Extract ingredient name
        ingredient_name = self._extract_ingredient_name(remaining_text, language)
        
        # Translate ingredient name to English
        ingredient_name_en = self._translate_ingredient_to_english(ingredient_name, language)
        
        # Extract preparation method
        preparation = self._extract_preparation_method(remaining_text, language)
        
        # Normalize unit
        unit_normalized = self._normalize_unit(unit, measurement_system)
        
        # Calculate confidence
        confidence = self._calculate_parsing_confidence(
            ingredient_name, quantity, unit, language, measurement_system
        )
        
        # Get cultural notes
        cultural_notes = self._get_cultural_notes(ingredient_name, language, measurement_system)
        
        return MultilingualParsingResult(
            ingredient_name=ingredient_name,
            ingredient_name_en=ingredient_name_en,
            quantity=quantity,
            unit=unit,
            unit_normalized=unit_normalized,
            preparation=preparation,
            language=language,
            measurement_system=measurement_system,
            confidence=confidence,
            cultural_notes=cultural_notes
        )
    
    def get_ocr_language_config(self, language: Language) -> Dict[str, Any]:
        """
        Get OCR configuration for specific language.
        
        Args:
            language: Target language
            
        Returns:
            OCR configuration parameters
        """
        return self.ocr_language_configs.get(language.value, self.ocr_language_configs['en'])
    
    def convert_measurements(self, quantity: str, unit: str, 
                           from_system: MeasurementSystem, 
                           to_system: MeasurementSystem) -> Tuple[str, str]:
        """
        Convert measurements between systems.
        
        Args:
            quantity: Original quantity
            unit: Original unit
            from_system: Source measurement system
            to_system: Target measurement system
            
        Returns:
            Converted quantity and unit
        """
        try:
            # Parse quantity
            qty_value = self._parse_quantity_value(quantity)
            
            # Get conversion factor
            conversion_key = f"{unit.lower()}_{from_system.value}_to_{to_system.value}"
            
            if conversion_key in self.conversion_factors:
                factor = self.conversion_factors[conversion_key]
                converted_qty = qty_value * factor['factor']
                converted_unit = factor['target_unit']
                
                # Format converted quantity
                formatted_qty = self._format_quantity(converted_qty)
                
                return formatted_qty, converted_unit
            else:
                # No conversion available
                return quantity, unit
                
        except Exception as e:
            self.logger.warning(f"Measurement conversion failed: {e}")
            return quantity, unit
    
    def normalize_recipe_format(self, recipe_text: str, target_language: Language = Language.ENGLISH,
                              target_system: MeasurementSystem = MeasurementSystem.METRIC) -> str:
        """
        Normalize recipe format to target language and measurement system.
        
        Args:
            recipe_text: Original recipe text
            target_language: Target language for normalization
            target_system: Target measurement system
            
        Returns:
            Normalized recipe text
        """
        # Detect source language and measurement system
        lang_result = self.detect_language(recipe_text)
        measurement_result = self.detect_measurement_system(recipe_text, lang_result.primary_language)
        
        # Split into lines and process each
        lines = recipe_text.strip().split('\n')
        normalized_lines = []
        
        for line in lines:
            if line.strip():
                # Parse as ingredient
                parsed = self.parse_multilingual_ingredient(
                    line, lang_result.primary_language, measurement_result.primary_system
                )
                
                # Convert measurements if needed
                if measurement_result.primary_system != target_system:
                    converted_qty, converted_unit = self.convert_measurements(
                        parsed.quantity, parsed.unit,
                        measurement_result.primary_system, target_system
                    )
                else:
                    converted_qty, converted_unit = parsed.quantity, parsed.unit_normalized
                
                # Format normalized line
                ingredient_name = parsed.ingredient_name_en if target_language == Language.ENGLISH else parsed.ingredient_name
                
                normalized_line = f"{converted_qty} {converted_unit} {ingredient_name}"
                if parsed.preparation:
                    normalized_line += f", {parsed.preparation}"
                
                normalized_lines.append(normalized_line)
        
        return '\n'.join(normalized_lines)
    
    def _load_language_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load language detection patterns."""
        return {
            'en': {
                'ingredients': ['flour', 'sugar', 'butter', 'eggs', 'milk', 'salt', 'pepper', 'onion', 'garlic', 'chicken', 'beef', 'pork'],
                'units': ['cup', 'cups', 'tablespoon', 'tablespoons', 'tbsp', 'teaspoon', 'teaspoons', 'tsp', 'ounce', 'ounces', 'oz', 'pound', 'pounds', 'lb', 'lbs'],
                'keywords': ['recipe', 'ingredients', 'directions', 'instructions', 'cook', 'bake', 'fry', 'boil', 'mix', 'stir', 'add', 'serve']
            },
            'es': {
                'ingredients': ['harina', 'azúcar', 'mantequilla', 'huevos', 'leche', 'sal', 'pimienta', 'cebolla', 'ajo', 'pollo', 'carne', 'cerdo'],
                'units': ['taza', 'tazas', 'cucharada', 'cucharadas', 'cucharadita', 'cucharaditas', 'onza', 'onzas', 'libra', 'libras', 'gramo', 'gramos', 'kilogramo', 'litro'],
                'keywords': ['receta', 'ingredientes', 'preparación', 'instrucciones', 'cocinar', 'hornear', 'freír', 'hervir', 'mezclar', 'revolver', 'agregar', 'servir']
            },
            'fr': {
                'ingredients': ['farine', 'sucre', 'beurre', 'œufs', 'lait', 'sel', 'poivre', 'oignon', 'ail', 'poulet', 'bœuf', 'porc'],
                'units': ['tasse', 'tasses', 'cuillère', 'cuillères', 'once', 'onces', 'livre', 'livres', 'gramme', 'grammes', 'kilogramme', 'litre'],
                'keywords': ['recette', 'ingrédients', 'préparation', 'instructions', 'cuire', 'cuisson', 'frire', 'bouillir', 'mélanger', 'remuer', 'ajouter', 'servir']
            },
            'de': {
                'ingredients': ['mehl', 'zucker', 'butter', 'eier', 'milch', 'salz', 'pfeffer', 'zwiebel', 'knoblauch', 'huhn', 'rindfleisch', 'schweinefleisch'],
                'units': ['tasse', 'tassen', 'esslöffel', 'teelöffel', 'unze', 'unzen', 'pfund', 'gramm', 'kilogramm', 'liter'],
                'keywords': ['rezept', 'zutaten', 'zubereitung', 'anweisungen', 'kochen', 'backen', 'braten', 'kochen', 'mischen', 'rühren', 'hinzufügen', 'servieren']
            },
            'it': {
                'ingredients': ['farina', 'zucchero', 'burro', 'uova', 'latte', 'sale', 'pepe', 'cipolla', 'aglio', 'pollo', 'manzo', 'maiale'],
                'units': ['tazza', 'tazze', 'cucchiaio', 'cucchiai', 'cucchiaino', 'cucchiaini', 'oncia', 'once', 'libbra', 'grammo', 'grammi', 'chilogrammo', 'litro'],
                'keywords': ['ricetta', 'ingredienti', 'preparazione', 'istruzioni', 'cucinare', 'cuocere', 'friggere', 'bollire', 'mescolare', 'girare', 'aggiungere', 'servire']
            },
            'pt': {
                'ingredients': ['farinha', 'açúcar', 'manteiga', 'ovos', 'leite', 'sal', 'pimenta', 'cebola', 'alho', 'frango', 'carne', 'porco'],
                'units': ['xícara', 'xícaras', 'colher', 'colheres', 'colher de chá', 'onça', 'onças', 'libra', 'libras', 'grama', 'gramas', 'quilograma', 'litro'],
                'keywords': ['receita', 'ingredientes', 'preparação', 'instruções', 'cozinhar', 'assar', 'fritar', 'ferver', 'misturar', 'mexer', 'adicionar', 'servir']
            },
            'ja': {
                'ingredients': ['小麦粉', '砂糖', 'バター', '卵', '牛乳', '塩', 'コショウ', '玉ねぎ', 'にんにく', '鶏肉', '牛肉', '豚肉'],
                'units': ['カップ', '大さじ', '小さじ', 'グラム', 'キログラム', 'リットル', 'ミリリットル'],
                'keywords': ['レシピ', '材料', '作り方', '調理', '料理', '焼く', '炒める', '茹でる', '混ぜる', '加える', '盛る']
            },
            'ko': {
                'ingredients': ['밀가루', '설탕', '버터', '계란', '우유', '소금', '후추', '양파', '마늘', '닭고기', '소고기', '돼지고기'],
                'units': ['컵', '큰술', '작은술', '그램', '킬로그램', '리터', '밀리리터'],
                'keywords': ['레시피', '재료', '만들기', '조리법', '요리', '굽기', '볶기', '끓이기', '섞기', '넣기', '담기']
            },
            'zh': {
                'ingredients': ['面粉', '糖', '黄油', '鸡蛋', '牛奶', '盐', '胡椒', '洋葱', '大蒜', '鸡肉', '牛肉', '猪肉'],
                'units': ['杯', '大勺', '小勺', '克', '公斤', '升', '毫升'],
                'keywords': ['食谱', '配料', '制作', '烹饪', '料理', '烘烤', '炒', '煮', '混合', '加入', '盛装']
            }
        }
    
    def _load_ingredient_dictionaries(self) -> Dict[str, Dict[str, str]]:
        """Load ingredient translation dictionaries."""
        return {
            'es_to_en': {
                'harina': 'flour', 'azúcar': 'sugar', 'mantequilla': 'butter',
                'huevos': 'eggs', 'leche': 'milk', 'sal': 'salt', 'pimienta': 'pepper',
                'cebolla': 'onion', 'ajo': 'garlic', 'pollo': 'chicken', 'carne': 'beef',
                'cerdo': 'pork', 'tomate': 'tomato', 'papa': 'potato', 'zanahoria': 'carrot'
            },
            'fr_to_en': {
                'farine': 'flour', 'sucre': 'sugar', 'beurre': 'butter',
                'œufs': 'eggs', 'lait': 'milk', 'sel': 'salt', 'poivre': 'pepper',
                'oignon': 'onion', 'ail': 'garlic', 'poulet': 'chicken', 'bœuf': 'beef',
                'porc': 'pork', 'tomate': 'tomato', 'pomme de terre': 'potato'
            },
            'de_to_en': {
                'mehl': 'flour', 'zucker': 'sugar', 'butter': 'butter',
                'eier': 'eggs', 'milch': 'milk', 'salz': 'salt', 'pfeffer': 'pepper',
                'zwiebel': 'onion', 'knoblauch': 'garlic', 'huhn': 'chicken',
                'rindfleisch': 'beef', 'schweinefleisch': 'pork'
            },
            'it_to_en': {
                'farina': 'flour', 'zucchero': 'sugar', 'burro': 'butter',
                'uova': 'eggs', 'latte': 'milk', 'sale': 'salt', 'pepe': 'pepper',
                'cipolla': 'onion', 'aglio': 'garlic', 'pollo': 'chicken',
                'manzo': 'beef', 'maiale': 'pork'
            },
            'pt_to_en': {
                'farinha': 'flour', 'açúcar': 'sugar', 'manteiga': 'butter',
                'ovos': 'eggs', 'leite': 'milk', 'sal': 'salt', 'pimenta': 'pepper',
                'cebola': 'onion', 'alho': 'garlic', 'frango': 'chicken',
                'carne': 'beef', 'porco': 'pork'
            }
        }
    
    def _load_measurement_units(self) -> Dict[str, List[str]]:
        """Load measurement units by system."""
        return {
            'metric': [
                'g', 'gram', 'grams', 'gramme', 'grammes', 'gramo', 'gramos',
                'kg', 'kilogram', 'kilograms', 'kilogramme', 'kilogrammes', 'kilogramo', 'kilogramos',
                'ml', 'milliliter', 'milliliters', 'millilitre', 'millilitres', 'mililitro', 'mililitros',
                'l', 'liter', 'liters', 'litre', 'litres', 'litro', 'litros'
            ],
            'imperial': [
                'oz', 'ounce', 'ounces', 'onza', 'onzas', 'once', 'onces',
                'lb', 'lbs', 'pound', 'pounds', 'libra', 'libras', 'livre', 'livres',
                'cup', 'cups', 'taza', 'tazas', 'tasse', 'tasses',
                'tbsp', 'tablespoon', 'tablespoons', 'cucharada', 'cucharadas', 'cuillère', 'cuillères',
                'tsp', 'teaspoon', 'teaspoons', 'cucharadita', 'cucharaditas',
                'pint', 'pints', 'quart', 'quarts', 'gallon', 'gallons'
            ],
            'traditional_asian': [
                'go', 'sho', 'to', 'koku',  # Japanese
                'doe', 'mal', 'toe', 'seok',  # Korean
                'sheng', 'dou', 'dan', 'shi'  # Chinese
            ],
            'traditional_european': [
                'livre', 'once', 'gros', 'grain',  # Old French
                'pfund', 'lot', 'quentchen', 'gran',  # Old German
                'libbra', 'oncia', 'dramma', 'scrupolo'  # Old Italian
            ]
        }
    
    def _load_cultural_contexts(self) -> Dict[str, Dict[str, str]]:
        """Load cultural contexts for different regions."""
        return {
            'en': {
                'baking': 'American/British baking traditions',
                'cooking': 'Western cooking methods',
                'measurements': 'Imperial system common in US/UK'
            },
            'es': {
                'baking': 'Latin American and Spanish baking',
                'cooking': 'Hispanic cooking traditions',
                'measurements': 'Mix of metric and traditional units'
            },
            'fr': {
                'baking': 'French patisserie and baking',
                'cooking': 'French culinary traditions',
                'measurements': 'Metric system with traditional French units'
            },
            'de': {
                'baking': 'German and Austrian baking',
                'cooking': 'Central European cooking',
                'measurements': 'Metric system predominant'
            },
            'ja': {
                'baking': 'Japanese baking adaptations',
                'cooking': 'Traditional Japanese cooking',
                'measurements': 'Traditional Japanese units + metric'
            },
            'ko': {
                'baking': 'Korean baking styles',
                'cooking': 'Korean culinary traditions',
                'measurements': 'Traditional Korean units + metric'
            },
            'zh': {
                'baking': 'Chinese baking adaptations',
                'cooking': 'Traditional Chinese cooking',
                'measurements': 'Traditional Chinese units + metric'
            }
        }
    
    def _load_ocr_language_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load OCR configurations for different languages."""
        return {
            'en': {'language': 'eng', 'script': 'latin', 'direction': 'ltr'},
            'es': {'language': 'spa', 'script': 'latin', 'direction': 'ltr'},
            'fr': {'language': 'fra', 'script': 'latin', 'direction': 'ltr'},
            'de': {'language': 'deu', 'script': 'latin', 'direction': 'ltr'},
            'it': {'language': 'ita', 'script': 'latin', 'direction': 'ltr'},
            'pt': {'language': 'por', 'script': 'latin', 'direction': 'ltr'},
            'ru': {'language': 'rus', 'script': 'cyrillic', 'direction': 'ltr'},
            'ja': {'language': 'jpn', 'script': 'mixed', 'direction': 'ttb'},
            'ko': {'language': 'kor', 'script': 'hangul', 'direction': 'ltr'},
            'zh': {'language': 'chi_sim', 'script': 'chinese', 'direction': 'ttb'},
            'ar': {'language': 'ara', 'script': 'arabic', 'direction': 'rtl'},
            'hi': {'language': 'hin', 'script': 'devanagari', 'direction': 'ltr'}
        }
    
    def _load_conversion_factors(self) -> Dict[str, Dict[str, Any]]:
        """Load conversion factors between measurement systems."""
        return {
            # Volume conversions
            'cup_imperial_to_metric': {'factor': 236.588, 'target_unit': 'ml'},
            'tbsp_imperial_to_metric': {'factor': 14.787, 'target_unit': 'ml'},
            'tsp_imperial_to_metric': {'factor': 4.929, 'target_unit': 'ml'},
            'fl oz_imperial_to_metric': {'factor': 29.574, 'target_unit': 'ml'},
            'pint_imperial_to_metric': {'factor': 473.176, 'target_unit': 'ml'},
            'quart_imperial_to_metric': {'factor': 946.353, 'target_unit': 'ml'},
            'gallon_imperial_to_metric': {'factor': 3785.41, 'target_unit': 'ml'},
            
            # Weight conversions
            'oz_imperial_to_metric': {'factor': 28.35, 'target_unit': 'g'},
            'lb_imperial_to_metric': {'factor': 453.592, 'target_unit': 'g'},
            
            # Reverse conversions
            'ml_metric_to_imperial': {'factor': 0.00423, 'target_unit': 'cup'},
            'g_metric_to_imperial': {'factor': 0.0353, 'target_unit': 'oz'},
            
            # Traditional unit conversions
            'go_traditional_asian_to_metric': {'factor': 180, 'target_unit': 'ml'},
            'sho_traditional_asian_to_metric': {'factor': 1800, 'target_unit': 'ml'},
            'livre_traditional_european_to_metric': {'factor': 489.5, 'target_unit': 'g'}
        }
    
    def _detect_script_type(self, text: str) -> str:
        """Detect script type of the text."""
        # Check for different script types
        latin_count = sum(1 for char in text if ord(char) < 256)
        cyrillic_count = sum(1 for char in text if 0x0400 <= ord(char) <= 0x04FF)
        arabic_count = sum(1 for char in text if 0x0600 <= ord(char) <= 0x06FF)
        chinese_count = sum(1 for char in text if 0x4E00 <= ord(char) <= 0x9FFF)
        japanese_count = sum(1 for char in text if 0x3040 <= ord(char) <= 0x309F or 0x30A0 <= ord(char) <= 0x30FF)
        korean_count = sum(1 for char in text if 0xAC00 <= ord(char) <= 0xD7AF)
        
        script_counts = {
            'latin': latin_count,
            'cyrillic': cyrillic_count,
            'arabic': arabic_count,
            'chinese': chinese_count,
            'japanese': japanese_count,
            'korean': korean_count
        }
        
        return max(script_counts.items(), key=lambda x: x[1])[0]
    
    def _detect_text_direction(self, text: str, language: Language) -> str:
        """Detect text direction based on language."""
        rtl_languages = {Language.ARABIC, Language.HEBREW}
        ttb_languages = {Language.JAPANESE, Language.CHINESE}
        
        if language in rtl_languages:
            return 'rtl'
        elif language in ttb_languages:
            return 'ttb'
        else:
            return 'ltr'
    
    def _get_default_measurement_system(self, language: Language) -> MeasurementSystem:
        """Get default measurement system for a language."""
        imperial_languages = {Language.ENGLISH}
        traditional_asian_languages = {Language.JAPANESE, Language.KOREAN, Language.CHINESE}
        traditional_european_languages = {Language.FRENCH, Language.GERMAN, Language.ITALIAN}
        
        if language in imperial_languages:
            return MeasurementSystem.IMPERIAL
        elif language in traditional_asian_languages:
            return MeasurementSystem.TRADITIONAL_ASIAN
        elif language in traditional_european_languages:
            return MeasurementSystem.TRADITIONAL_EUROPEAN
        else:
            return MeasurementSystem.METRIC
    
    def _should_convert_measurements(self, system: MeasurementSystem, language: Language) -> bool:
        """Determine if measurements should be converted."""
        # Convert to metric if not already metric and not in traditional system context
        return system != MeasurementSystem.METRIC and language not in {Language.ENGLISH}
    
    def _get_cultural_context(self, language: Language, system: MeasurementSystem) -> str:
        """Get cultural context for language and measurement system."""
        contexts = self.cultural_contexts.get(language.value, {})
        return contexts.get('measurements', f"{language.value} culinary context")
    
    def _clean_ingredient_text(self, text: str, language: Language) -> str:
        """Clean ingredient text based on language."""
        # Remove common non-ingredient characters
        cleaned = re.sub(r'[^\w\s\-\./,]', '', text)
        
        # Language-specific cleaning
        if language == Language.CHINESE or language == Language.JAPANESE:
            # Remove common punctuation for CJK languages
            cleaned = re.sub(r'[。，、]', ' ', cleaned)
        
        return cleaned.strip()
    
    def _extract_quantity_and_unit(self, text: str, language: Language, 
                                 system: MeasurementSystem) -> Tuple[str, str, str]:
        """Extract quantity and unit from text."""
        # Common patterns for quantities
        quantity_patterns = [
            r'(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?)',  # Numbers and fractions
            r'(\d+(?:\s+\d+/\d+)?)',  # Mixed numbers
            r'([一二三四五六七八九十百千万]+)',  # Chinese numbers
            r'([一二三四五六七八九十]+)',  # Japanese numbers
        ]
        
        # Get relevant units for the measurement system
        relevant_units = self.measurement_units.get(system.value, [])
        
        for pattern in quantity_patterns:
            # Try to match quantity followed by unit
            for unit in relevant_units:
                unit_pattern = rf'{pattern}\s*{re.escape(unit)}'
                match = re.search(unit_pattern, text, re.IGNORECASE)
                
                if match:
                    quantity = match.group(1)
                    remaining_text = text.replace(match.group(0), '').strip()
                    return quantity, unit, remaining_text
        
        # Try to match quantity without unit
        for pattern in quantity_patterns:
            match = re.search(pattern, text)
            if match:
                quantity = match.group(1)
                remaining_text = text.replace(match.group(0), '').strip()
                return quantity, '', remaining_text
        
        return '', '', text
    
    def _extract_ingredient_name(self, text: str, language: Language) -> str:
        """Extract ingredient name from text."""
        # Remove common preparation words
        prep_words = {
            Language.ENGLISH: ['chopped', 'diced', 'sliced', 'minced', 'grated', 'fresh', 'dried'],
            Language.SPANISH: ['picado', 'cortado', 'rallado', 'fresco', 'seco'],
            Language.FRENCH: ['haché', 'coupé', 'râpé', 'frais', 'sec'],
            Language.GERMAN: ['gehackt', 'geschnitten', 'gerieben', 'frisch', 'getrocknet'],
            Language.ITALIAN: ['tritato', 'tagliato', 'grattugiato', 'fresco', 'secco']
        }
        
        words = text.split()
        clean_words = []
        
        for word in words:
            if word.lower() not in prep_words.get(language, []):
                clean_words.append(word)
        
        return ' '.join(clean_words).strip()
    
    def _translate_ingredient_to_english(self, ingredient: str, language: Language) -> str:
        """Translate ingredient name to English."""
        if language == Language.ENGLISH:
            return ingredient
        
        # Get translation dictionary for the language
        dict_key = f"{language.value}_to_en"
        translation_dict = self.ingredient_dictionaries.get(dict_key, {})
        
        # Try exact match first
        ingredient_lower = ingredient.lower()
        if ingredient_lower in translation_dict:
            return translation_dict[ingredient_lower]
        
        # Try word-by-word translation
        words = ingredient.split()
        translated_words = []
        
        for word in words:
            translated_word = translation_dict.get(word.lower(), word)
            translated_words.append(translated_word)
        
        return ' '.join(translated_words)
    
    def _extract_preparation_method(self, text: str, language: Language) -> str:
        """Extract preparation method from text."""
        prep_words = {
            Language.ENGLISH: ['chopped', 'diced', 'sliced', 'minced', 'grated', 'fresh', 'dried', 'cooked'],
            Language.SPANISH: ['picado', 'cortado', 'rallado', 'fresco', 'seco', 'cocido'],
            Language.FRENCH: ['haché', 'coupé', 'râpé', 'frais', 'sec', 'cuit'],
            Language.GERMAN: ['gehackt', 'geschnitten', 'gerieben', 'frisch', 'getrocknet', 'gekocht'],
            Language.ITALIAN: ['tritato', 'tagliato', 'grattugiato', 'fresco', 'secco', 'cotto']
        }
        
        words = text.split()
        prep_methods = []
        
        for word in words:
            if word.lower() in prep_words.get(language, []):
                prep_methods.append(word)
        
        return ', '.join(prep_methods)
    
    def _normalize_unit(self, unit: str, system: MeasurementSystem) -> str:
        """Normalize unit to standard form."""
        if not unit:
            return unit
        
        # Unit normalization mappings
        unit_mappings = {
            # Volume
            'c': 'cup', 'cups': 'cup', 'taza': 'cup', 'tazas': 'cup',
            'tbsp': 'tablespoon', 'tablespoons': 'tablespoon', 'cucharada': 'tablespoon',
            'tsp': 'teaspoon', 'teaspoons': 'teaspoon', 'cucharadita': 'teaspoon',
            'ml': 'milliliter', 'milliliters': 'milliliter', 'millilitres': 'milliliter',
            'l': 'liter', 'liters': 'liter', 'litres': 'liter',
            
            # Weight
            'g': 'gram', 'grams': 'gram', 'grammes': 'gram',
            'kg': 'kilogram', 'kilograms': 'kilogram', 'kilogrammes': 'kilogram',
            'oz': 'ounce', 'ounces': 'ounce', 'onza': 'ounce',
            'lb': 'pound', 'lbs': 'pound', 'pounds': 'pound', 'libra': 'pound'
        }
        
        return unit_mappings.get(unit.lower(), unit)
    
    def _calculate_parsing_confidence(self, ingredient: str, quantity: str, unit: str,
                                    language: Language, system: MeasurementSystem) -> float:
        """Calculate confidence score for parsing result."""
        confidence = 0.0
        
        # Ingredient name confidence
        if ingredient and len(ingredient) > 2:
            confidence += 0.4
        
        # Quantity confidence
        if quantity:
            try:
                float(quantity.replace('/', '.'))
                confidence += 0.3
            except ValueError:
                confidence += 0.1
        
        # Unit confidence
        if unit:
            relevant_units = self.measurement_units.get(system.value, [])
            if unit.lower() in [u.lower() for u in relevant_units]:
                confidence += 0.3
            else:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_cultural_notes(self, ingredient: str, language: Language, 
                          system: MeasurementSystem) -> List[str]:
        """Get cultural notes for ingredient and context."""
        notes = []
        
        # Add language-specific notes
        if language != Language.ENGLISH:
            notes.append(f"Original language: {language.value}")
        
        # Add measurement system notes
        if system != MeasurementSystem.METRIC:
            notes.append(f"Measurement system: {system.value}")
        
        # Add ingredient-specific cultural notes
        cultural_ingredients = {
            'miso': 'Japanese fermented soybean paste',
            'chorizo': 'Spanish/Mexican spiced sausage',
            'gruyère': 'Swiss cheese',
            'pancetta': 'Italian cured meat',
            'ghee': 'Indian clarified butter'
        }
        
        ingredient_lower = ingredient.lower()
        for cultural_ingredient, note in cultural_ingredients.items():
            if cultural_ingredient in ingredient_lower:
                notes.append(note)
        
        return notes
    
    def _parse_quantity_value(self, quantity: str) -> float:
        """Parse quantity string to numeric value."""
        # Handle fractions
        if '/' in quantity:
            parts = quantity.split('/')
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except ValueError:
                    pass
        
        # Handle mixed numbers (e.g., "1 1/2")
        if ' ' in quantity and '/' in quantity:
            parts = quantity.split(' ')
            if len(parts) == 2:
                try:
                    whole = float(parts[0])
                    fraction_parts = parts[1].split('/')
                    if len(fraction_parts) == 2:
                        fraction = float(fraction_parts[0]) / float(fraction_parts[1])
                        return whole + fraction
                except ValueError:
                    pass
        
        # Handle regular numbers
        try:
            return float(quantity)
        except ValueError:
            return 1.0  # Default value
    
    def _format_quantity(self, quantity: float) -> str:
        """Format quantity for display."""
        if quantity == int(quantity):
            return str(int(quantity))
        else:
            return f"{quantity:.2f}".rstrip('0').rstrip('.')


def main():
    """Main function for multilingual measurement handling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multilingual Recipe Text Handler')
    parser.add_argument('input_text', help='Input recipe text or file')
    parser.add_argument('--detect-language', action='store_true', help='Detect language only')
    parser.add_argument('--detect-measurements', action='store_true', help='Detect measurement system only')
    parser.add_argument('--normalize', action='store_true', help='Normalize recipe format')
    parser.add_argument('--target-language', default='en', help='Target language for normalization')
    parser.add_argument('--target-system', default='metric', help='Target measurement system')
    parser.add_argument('--config', '-c', help='Configuration file (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize handler
    handler = MultilingualMeasurementHandler(config)
    
    # Read input text
    if Path(args.input_text).exists():
        with open(args.input_text, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.input_text
    
    try:
        # Detect language
        if args.detect_language or args.verbose:
            lang_result = handler.detect_language(text)
            print(f"Language Detection:")
            print(f"  Primary: {lang_result.primary_language.value} (confidence: {lang_result.confidence:.3f})")
            print(f"  Script: {lang_result.script_type}")
            print(f"  Direction: {lang_result.text_direction}")
            
            if lang_result.secondary_languages:
                print(f"  Secondary languages:")
                for lang, conf in lang_result.secondary_languages:
                    print(f"    {lang.value}: {conf:.3f}")
            print()
        
        # Detect measurements
        if args.detect_measurements or args.verbose:
            lang_result = handler.detect_language(text)
            measurement_result = handler.detect_measurement_system(text, lang_result.primary_language)
            
            print(f"Measurement System Detection:")
            print(f"  Primary: {measurement_result.primary_system.value} (confidence: {measurement_result.confidence:.3f})")
            print(f"  Cultural context: {measurement_result.cultural_context}")
            print(f"  Conversion needed: {measurement_result.conversion_needed}")
            
            if measurement_result.detected_units:
                print(f"  Detected units: {', '.join(measurement_result.detected_units[:10])}")
            print()
        
        # Normalize recipe
        if args.normalize:
            target_lang = Language(args.target_language)
            target_system = MeasurementSystem(args.target_system)
            
            normalized_text = handler.normalize_recipe_format(text, target_lang, target_system)
            
            print(f"Normalized Recipe ({target_lang.value}, {target_system.value}):")
            print("=" * 50)
            print(normalized_text)
            print()
        
        # Parse individual ingredients
        if args.verbose:
            lang_result = handler.detect_language(text)
            measurement_result = handler.detect_measurement_system(text, lang_result.primary_language)
            
            print(f"Ingredient Parsing:")
            print("-" * 30)
            
            lines = text.strip().split('\n')
            for i, line in enumerate(lines[:5]):  # Show first 5 lines
                if line.strip():
                    parsed = handler.parse_multilingual_ingredient(
                        line, lang_result.primary_language, measurement_result.primary_system
                    )
                    
                    print(f"Line {i+1}: {line}")
                    print(f"  Ingredient: {parsed.ingredient_name} ({parsed.ingredient_name_en})")
                    print(f"  Quantity: {parsed.quantity}")
                    print(f"  Unit: {parsed.unit} -> {parsed.unit_normalized}")
                    print(f"  Preparation: {parsed.preparation}")
                    print(f"  Confidence: {parsed.confidence:.3f}")
                    if parsed.cultural_notes:
                        print(f"  Cultural notes: {', '.join(parsed.cultural_notes)}")
                    print()
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())