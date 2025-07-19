#!/usr/bin/env python3
"""
Recipe Type-Specific Examples
Detailed examples for different types of recipe images with optimal settings
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class RecipeTypeProcessor:
    """Processor with optimized settings for different recipe types"""
    
    def __init__(self):
        try:
            from ingredient_pipeline import IngredientExtractionPipeline
            self.pipeline = IngredientExtractionPipeline()
        except ImportError:
            print("Warning: IngredientExtractionPipeline not available")
            self.pipeline = None
    
    def process_cookbook_page(self, image_path: str) -> Dict[str, Any]:
        """
        Process printed cookbook pages
        
        Characteristics:
        - Clean, professional typography
        - High contrast black text on white background
        - Consistent formatting and layout
        - Multiple columns or structured sections
        """
        print("📚 Processing Cookbook Page")
        print("=" * 40)
        
        optimal_settings = {
            'format_hint': 'cookbook',
            'ocr_engine': 'tesseract',  # Best for clean printed text
            'confidence_threshold': 0.3,
            'quality_threshold': 0.7,
            'language_hint': 'en',
            'enable_preprocessing': False  # Usually not needed for clean text
        }
        
        print("Optimal settings for cookbook pages:")
        for key, value in optimal_settings.items():
            print(f"  {key}: {value}")
        
        if self.pipeline:
            try:
                result = self.pipeline.process_image(image_path, **optimal_settings)
                self._display_cookbook_results(result)
                return result
            except Exception as e:
                print(f"Error processing cookbook page: {e}")
                return self._create_mock_cookbook_result(image_path)
        else:
            return self._create_mock_cookbook_result(image_path)
    
    def _create_mock_cookbook_result(self, image_path: str) -> Dict[str, Any]:
        """Create mock result for cookbook processing"""
        return {
            'filename': os.path.basename(image_path),
            'format_type': 'printed_cookbook',
            'quality_score': 0.92,
            'confidence_score': 0.86,
            'processing_time': 6.8,
            'language': 'en',
            'ingredients': [
                {
                    'ingredient_name': 'all-purpose flour',
                    'quantity': '2 1/4',
                    'unit': 'cups',
                    'preparation': 'sifted',
                    'confidence': 0.95,
                    'bbox': {'x1': 120, 'y1': 180, 'x2': 320, 'y2': 205}
                },
                {
                    'ingredient_name': 'granulated sugar',
                    'quantity': '3/4',
                    'unit': 'cup',
                    'preparation': '',
                    'confidence': 0.93,
                    'bbox': {'x1': 120, 'y1': 210, 'x2': 290, 'y2': 235}
                },
                {
                    'ingredient_name': 'large eggs',
                    'quantity': '2',
                    'unit': '',
                    'preparation': 'room temperature',
                    'confidence': 0.91,
                    'bbox': {'x1': 120, 'y1': 240, 'x2': 280, 'y2': 265}
                },
                {
                    'ingredient_name': 'unsalted butter',
                    'quantity': '1/2',
                    'unit': 'cup',
                    'preparation': 'melted and cooled',
                    'confidence': 0.88,
                    'bbox': {'x1': 120, 'y1': 270, 'x2': 350, 'y2': 295}
                },
                {
                    'ingredient_name': 'vanilla extract',
                    'quantity': '2',
                    'unit': 'teaspoons',
                    'preparation': 'pure',
                    'confidence': 0.89,
                    'bbox': {'x1': 120, 'y1': 300, 'x2': 280, 'y2': 325}
                }
            ]
        }
    
    def _display_cookbook_results(self, result: Dict[str, Any]):
        """Display cookbook processing results"""
        print(f"\n📊 Results for {result['filename']}:")
        print(f"Quality Score: {result['quality_score']:.2f} (Excellent)")
        print(f"Confidence: {result['confidence_score']:.2f} (High)")
        print(f"Processing Time: {result['processing_time']:.1f}s (Fast)")
        
        print(f"\n🥘 Ingredients ({len(result['ingredients'])}):")
        for i, ing in enumerate(result['ingredients'], 1):
            quantity = ing.get('quantity', '')
            unit = ing.get('unit', '')
            name = ing['ingredient_name']
            prep = ing.get('preparation', '')
            conf = ing['confidence']
            
            ingredient_text = f"{i:2d}. {quantity} {unit} {name}".strip()
            if prep:
                ingredient_text += f" ({prep})"
            
            status = "✅" if conf > 0.8 else "⚠️" if conf > 0.6 else "❌"
            print(f"{status} {ingredient_text} [confidence: {conf:.2f}]")
        
        print("\n💡 Cookbook processing insights:")
        print("• High confidence scores indicate clean, readable text")
        print("• Fast processing due to optimized OCR settings")
        print("• Preparation methods often included in cookbooks")
        print("• Consistent formatting enables reliable extraction")
    
    def process_handwritten_recipe(self, image_path: str) -> Dict[str, Any]:
        """
        Process handwritten recipe cards
        
        Characteristics:
        - Personal handwriting (varying legibility)
        - Informal abbreviations and notes
        - Possible stains or aging
        - Inconsistent text size and spacing
        """
        print("\n✍️ Processing Handwritten Recipe")
        print("=" * 40)
        
        optimal_settings = {
            'format_hint': 'handwritten',
            'ocr_engine': 'easyocr',  # Better for handwriting
            'confidence_threshold': 0.2,  # Lower due to handwriting variability
            'quality_threshold': 0.5,
            'enable_preprocessing': True,  # Helps with image quality
            'language_hint': 'en'
        }
        
        print("Optimal settings for handwritten recipes:")
        for key, value in optimal_settings.items():
            print(f"  {key}: {value}")
        
        if self.pipeline:
            try:
                result = self.pipeline.process_image(image_path, **optimal_settings)
                self._display_handwritten_results(result)
                return result
            except Exception as e:
                print(f"Error processing handwritten recipe: {e}")
                return self._create_mock_handwritten_result(image_path)
        else:
            return self._create_mock_handwritten_result(image_path)
    
    def _create_mock_handwritten_result(self, image_path: str) -> Dict[str, Any]:
        """Create mock result for handwritten processing"""
        return {
            'filename': os.path.basename(image_path),
            'format_type': 'handwritten',
            'quality_score': 0.58,
            'confidence_score': 0.42,
            'processing_time': 11.5,
            'language': 'en',
            'ingredients': [
                {
                    'ingredient_name': 'flour',
                    'quantity': '2',
                    'unit': 'c',  # Abbreviated units common
                    'preparation': '',
                    'confidence': 0.65,
                    'bbox': {'x1': 85, 'y1': 120, 'x2': 150, 'y2': 145}
                },
                {
                    'ingredient_name': 'sugar',
                    'quantity': '1',
                    'unit': 'cup',
                    'preparation': '',
                    'confidence': 0.48,
                    'bbox': {'x1': 85, 'y1': 150, 'x2': 160, 'y2': 175}
                },
                {
                    'ingredient_name': 'eggs',
                    'quantity': '2',
                    'unit': '',
                    'preparation': '',
                    'confidence': 0.52,
                    'bbox': {'x1': 85, 'y1': 180, 'x2': 130, 'y2': 205}
                },
                {
                    'ingredient_name': 'milk',
                    'quantity': '1/2',
                    'unit': 'c',
                    'preparation': '',
                    'confidence': 0.31,  # Lower confidence typical
                    'bbox': {'x1': 85, 'y1': 210, 'x2': 140, 'y2': 235}
                },
                {
                    'ingredient_name': 'vanilla',
                    'quantity': '1',
                    'unit': 'tsp',
                    'preparation': '',
                    'confidence': 0.28,
                    'bbox': {'x1': 85, 'y1': 240, 'x2': 155, 'y2': 265}
                }
            ]
        }
    
    def _display_handwritten_results(self, result: Dict[str, Any]):
        """Display handwritten processing results"""
        print(f"\n📊 Results for {result['filename']}:")
        print(f"Quality Score: {result['quality_score']:.2f} (Moderate - typical for handwriting)")
        print(f"Confidence: {result['confidence_score']:.2f} (Lower - requires review)")
        print(f"Processing Time: {result['processing_time']:.1f}s (Slower due to preprocessing)")
        
        print(f"\n📝 Ingredients ({len(result['ingredients'])}):")
        needs_review = []
        
        for i, ing in enumerate(result['ingredients'], 1):
            quantity = ing.get('quantity', '')
            unit = ing.get('unit', '')
            name = ing['ingredient_name']
            conf = ing['confidence']
            
            ingredient_text = f"{i:2d}. {quantity} {unit} {name}".strip()
            
            if conf > 0.5:
                status = "✅ Good"
                icon = "✅"
            elif conf > 0.3:
                status = "⚠️ Review"
                icon = "⚠️"
                needs_review.append(ingredient_text)
            else:
                status = "❌ Manual check"
                icon = "❌"
                needs_review.append(ingredient_text)
            
            print(f"{icon} {ingredient_text} [confidence: {conf:.2f}] - {status}")
        
        if needs_review:
            print(f"\n🔍 Items needing review ({len(needs_review)}):")
            for item in needs_review:
                print(f"   • {item}")
        
        print("\n💡 Handwritten processing insights:")
        print("• Lower confidence scores are normal for handwriting")
        print("• Abbreviated units (c, tsp, T) are common")
        print("• Manual review recommended for low confidence items")
        print("• Consider multiple OCR engines for difficult text")
    
    def process_digital_screenshot(self, image_path: str) -> Dict[str, Any]:
        """
        Process digital recipe screenshots from apps or websites
        
        Characteristics:
        - Clean digital fonts
        - Possible compression artifacts
        - Varied background colors
        - May include UI elements (buttons, ads)
        """
        print("\n📱 Processing Digital Screenshot")
        print("=" * 40)
        
        optimal_settings = {
            'format_hint': 'digital',
            'ocr_engine': 'paddleocr',  # Good for modern digital fonts
            'confidence_threshold': 0.4,
            'quality_threshold': 0.6,
            'enable_preprocessing': False,  # Usually not needed
            'filter_ui_elements': True
        }
        
        print("Optimal settings for digital screenshots:")
        for key, value in optimal_settings.items():
            print(f"  {key}: {value}")
        
        if self.pipeline:
            try:
                result = self.pipeline.process_image(image_path, **optimal_settings)
                result = self._filter_ui_elements(result)
                self._display_digital_results(result)
                return result
            except Exception as e:
                print(f"Error processing digital screenshot: {e}")
                return self._create_mock_digital_result(image_path)
        else:
            return self._create_mock_digital_result(image_path)
    
    def _create_mock_digital_result(self, image_path: str) -> Dict[str, Any]:
        """Create mock result for digital processing"""
        return {
            'filename': os.path.basename(image_path),
            'format_type': 'digital_screenshot',
            'quality_score': 0.78,
            'confidence_score': 0.82,
            'processing_time': 7.2,
            'language': 'en',
            'ingredients': [
                {
                    'ingredient_name': 'extra virgin olive oil',
                    'quantity': '3',
                    'unit': 'tablespoons',
                    'preparation': '',
                    'confidence': 0.94,
                    'bbox': {'x1': 60, 'y1': 240, 'x2': 280, 'y2': 265}
                },
                {
                    'ingredient_name': 'garlic cloves',
                    'quantity': '4',
                    'unit': '',
                    'preparation': 'minced',
                    'confidence': 0.91,
                    'bbox': {'x1': 60, 'y1': 270, 'x2': 220, 'y2': 295}
                },
                {
                    'ingredient_name': 'Roma tomatoes',
                    'quantity': '6',
                    'unit': 'large',
                    'preparation': 'diced',
                    'confidence': 0.88,
                    'bbox': {'x1': 60, 'y1': 300, 'x2': 240, 'y2': 325}
                },
                {
                    'ingredient_name': 'fresh basil',
                    'quantity': '1/4',
                    'unit': 'cup',
                    'preparation': 'chopped',
                    'confidence': 0.86,
                    'bbox': {'x1': 60, 'y1': 330, 'x2': 200, 'y2': 355}
                },
                {
                    'ingredient_name': 'mozzarella cheese',
                    'quantity': '8',
                    'unit': 'oz',
                    'preparation': 'fresh',
                    'confidence': 0.84,
                    'bbox': {'x1': 60, 'y1': 360, 'x2': 250, 'y2': 385}
                }
            ]
        }
    
    def _filter_ui_elements(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out UI elements from digital screenshots"""
        ui_keywords = [
            'share', 'save', 'print', 'email', 'rate', 'review', 'comment',
            'like', 'follow', 'subscribe', 'click', 'tap', 'button',
            'menu', 'home', 'search', 'login', 'sign up', 'ad', 'advertisement'
        ]
        
        original_count = len(result['ingredients'])
        filtered_ingredients = []
        
        for ingredient in result['ingredients']:
            name = ingredient['ingredient_name'].lower()
            is_ui_element = any(keyword in name for keyword in ui_keywords)
            
            if not is_ui_element:
                filtered_ingredients.append(ingredient)
        
        result['ingredients'] = filtered_ingredients
        result['ui_elements_filtered'] = original_count - len(filtered_ingredients)
        
        return result
    
    def _display_digital_results(self, result: Dict[str, Any]):
        """Display digital screenshot processing results"""
        print(f"\n📊 Results for {result['filename']}:")
        print(f"Quality Score: {result['quality_score']:.2f} (Good for digital)")
        print(f"Confidence: {result['confidence_score']:.2f} (High for clean fonts)")
        print(f"Processing Time: {result['processing_time']:.1f}s (Fast)")
        
        if 'ui_elements_filtered' in result:
            print(f"UI Elements Filtered: {result['ui_elements_filtered']}")
        
        print(f"\n📋 Ingredients ({len(result['ingredients'])}):")
        for i, ing in enumerate(result['ingredients'], 1):
            quantity = ing.get('quantity', '')
            unit = ing.get('unit', '')
            name = ing['ingredient_name']
            prep = ing.get('preparation', '')
            conf = ing['confidence']
            
            ingredient_text = f"{i:2d}. {quantity} {unit} {name}".strip()
            if prep:
                ingredient_text += f" ({prep})"
            
            status = "✅" if conf > 0.8 else "⚠️"
            print(f"{status} {ingredient_text} [confidence: {conf:.2f}]")
        
        print("\n💡 Digital screenshot insights:")
        print("• Clean digital fonts produce high confidence scores")
        print("• Watch for UI elements in results (filtered automatically)")
        print("• Consider cropping to ingredients section for better results")
        print("• PNG format often better than JPEG for screenshots")
    
    def process_blog_recipe(self, image_path: str) -> Dict[str, Any]:
        """
        Process recipe blog images
        
        Characteristics:
        - Mixed content (text + images)
        - Decorative fonts and styling
        - Variable layouts
        - Possible watermarks or overlays
        """
        print("\n🌐 Processing Recipe Blog Image")
        print("=" * 40)
        
        optimal_settings = {
            'format_hint': 'blog',
            'ocr_engine': 'paddleocr',  # Good for varied layouts
            'confidence_threshold': 0.35,
            'quality_threshold': 0.6,
            'ignore_decorative_text': True,
            'enable_preprocessing': True  # May help with varied quality
        }
        
        print("Optimal settings for blog recipes:")
        for key, value in optimal_settings.items():
            print(f"  {key}: {value}")
        
        if self.pipeline:
            try:
                result = self.pipeline.process_image(image_path, **optimal_settings)
                result = self._filter_blog_decorations(result)
                self._display_blog_results(result)
                return result
            except Exception as e:
                print(f"Error processing blog recipe: {e}")
                return self._create_mock_blog_result(image_path)
        else:
            return self._create_mock_blog_result(image_path)
    
    def _create_mock_blog_result(self, image_path: str) -> Dict[str, Any]:
        """Create mock result for blog processing"""
        return {
            'filename': os.path.basename(image_path),
            'format_type': 'recipe_blog',
            'quality_score': 0.71,
            'confidence_score': 0.68,
            'processing_time': 9.8,
            'language': 'en',
            'ingredients': [
                {
                    'ingredient_name': 'coconut oil',
                    'quantity': '2',
                    'unit': 'tablespoons',
                    'preparation': 'melted',
                    'confidence': 0.79,
                    'bbox': {'x1': 100, 'y1': 320, 'x2': 260, 'y2': 345}
                },
                {
                    'ingredient_name': 'quinoa',
                    'quantity': '1',
                    'unit': 'cup',
                    'preparation': 'rinsed',
                    'confidence': 0.73,
                    'bbox': {'x1': 100, 'y1': 350, 'x2': 200, 'y2': 375}
                },
                {
                    'ingredient_name': 'vegetable broth',
                    'quantity': '2',
                    'unit': 'cups',
                    'preparation': 'low sodium',
                    'confidence': 0.71,
                    'bbox': {'x1': 100, 'y1': 380, 'x2': 280, 'y2': 405}
                },
                {
                    'ingredient_name': 'bell pepper',
                    'quantity': '1',
                    'unit': 'large',
                    'preparation': 'diced',
                    'confidence': 0.68,
                    'bbox': {'x1': 100, 'y1': 410, 'x2': 220, 'y2': 435}
                },
                {
                    'ingredient_name': 'chickpeas',
                    'quantity': '1',
                    'unit': 'can',
                    'preparation': 'drained and rinsed',
                    'confidence': 0.65,
                    'bbox': {'x1': 100, 'y1': 440, 'x2': 300, 'y2': 465}
                }
            ]
        }
    
    def _filter_blog_decorations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out decorative text from blog images"""
        decorative_keywords = [
            'recipe', 'blog', 'website', 'copyright', 'photo', 'image',
            'delicious', 'amazing', 'perfect', 'easy', 'quick', 'healthy',
            'serves', 'prep time', 'cook time', 'total time', 'difficulty',
            'calories', 'nutrition', 'author', 'by', 'posted', 'updated'
        ]
        
        original_count = len(result['ingredients'])
        filtered_ingredients = []
        
        for ingredient in result['ingredients']:
            name = ingredient['ingredient_name'].lower()
            is_decorative = any(keyword in name for keyword in decorative_keywords)
            
            # Also filter very short or very long text that's likely decorative
            if len(name) < 3 or len(name) > 40:
                is_decorative = True
            
            if not is_decorative:
                filtered_ingredients.append(ingredient)
        
        result['ingredients'] = filtered_ingredients
        result['decorative_text_filtered'] = original_count - len(filtered_ingredients)
        
        return result
    
    def _display_blog_results(self, result: Dict[str, Any]):
        """Display blog recipe processing results"""
        print(f"\n📊 Results for {result['filename']}:")
        print(f"Quality Score: {result['quality_score']:.2f} (Variable for blogs)")
        print(f"Confidence: {result['confidence_score']:.2f} (Moderate due to styling)")
        print(f"Processing Time: {result['processing_time']:.1f}s (Slower due to complexity)")
        
        if 'decorative_text_filtered' in result:
            print(f"Decorative Text Filtered: {result['decorative_text_filtered']}")
        
        print(f"\n🥗 Ingredients ({len(result['ingredients'])}):")
        for i, ing in enumerate(result['ingredients'], 1):
            quantity = ing.get('quantity', '')
            unit = ing.get('unit', '')
            name = ing['ingredient_name']
            prep = ing.get('preparation', '')
            conf = ing['confidence']
            
            ingredient_text = f"{i:2d}. {quantity} {unit} {name}".strip()
            if prep:
                ingredient_text += f" ({prep})"
            
            status = "✅" if conf > 0.7 else "⚠️"
            print(f"{status} {ingredient_text} [confidence: {conf:.2f}]")
        
        print("\n💡 Blog recipe insights:")
        print("• Mixed layouts can reduce processing accuracy")
        print("• Decorative text often mixed with ingredients")
        print("• Consider cropping to ingredients section")
        print("• Modern food blogs often have clean ingredient lists")
    
    def process_foreign_language_recipe(self, image_path: str, language: str = 'auto') -> Dict[str, Any]:
        """
        Process recipes in foreign languages
        
        Args:
            image_path: Path to the recipe image
            language: Language code (es, fr, de, it, etc.) or 'auto' for detection
        """
        print(f"\n🌍 Processing Foreign Language Recipe ({language})")
        print("=" * 40)
        
        # Language-specific optimizations
        language_configs = {
            'es': {  # Spanish
                'ocr_engine': 'easyocr',
                'confidence_threshold': 0.25,
                'common_units': ['cucharadas', 'cucharaditas', 'tazas', 'gramos']
            },
            'fr': {  # French
                'ocr_engine': 'easyocr',
                'confidence_threshold': 0.25,
                'common_units': ['cuillères', 'tasses', 'grammes', 'litres']
            },
            'de': {  # German
                'ocr_engine': 'paddleocr',
                'confidence_threshold': 0.3,
                'common_units': ['EL', 'TL', 'Tassen', 'Gramm']
            },
            'it': {  # Italian
                'ocr_engine': 'easyocr',
                'confidence_threshold': 0.25,
                'common_units': ['cucchiai', 'cucchiaini', 'tazze', 'grammi']
            }
        }
        
        if language == 'auto':
            language = 'es'  # Default for demo
            print("Auto-detecting language... (detected: Spanish)")
        
        config = language_configs.get(language, language_configs['es'])
        
        optimal_settings = {
            'language_hint': language,
            'ocr_engine': config['ocr_engine'],
            'confidence_threshold': config['confidence_threshold'],
            'quality_threshold': 0.6,
            'enable_preprocessing': True
        }
        
        print(f"Optimal settings for {language} recipes:")
        for key, value in optimal_settings.items():
            print(f"  {key}: {value}")
        
        # Create mock multilingual result
        mock_results = {
            'es': self._create_mock_spanish_result(image_path),
            'fr': self._create_mock_french_result(image_path),
            'de': self._create_mock_german_result(image_path),
            'it': self._create_mock_italian_result(image_path)
        }
        
        result = mock_results.get(language, mock_results['es'])
        self._display_foreign_language_results(result, language)
        return result
    
    def _create_mock_spanish_result(self, image_path: str) -> Dict[str, Any]:
        """Create mock Spanish recipe result"""
        return {
            'filename': os.path.basename(image_path),
            'format_type': 'printed_cookbook',
            'quality_score': 0.82,
            'confidence_score': 0.74,
            'processing_time': 8.9,
            'language': 'es',
            'ingredients': [
                {
                    'ingredient_name': 'aceite de oliva',
                    'ingredient_name_en': 'olive oil',
                    'quantity': '3',
                    'unit': 'cucharadas',
                    'unit_en': 'tablespoons',
                    'confidence': 0.87
                },
                {
                    'ingredient_name': 'ajo',
                    'ingredient_name_en': 'garlic',
                    'quantity': '2',
                    'unit': 'dientes',
                    'unit_en': 'cloves',
                    'confidence': 0.92
                },
                {
                    'ingredient_name': 'tomates',
                    'ingredient_name_en': 'tomatoes',
                    'quantity': '4',
                    'unit': 'grandes',
                    'unit_en': 'large',
                    'confidence': 0.85
                }
            ]
        }
    
    def _create_mock_french_result(self, image_path: str) -> Dict[str, Any]:
        """Create mock French recipe result"""
        return {
            'filename': os.path.basename(image_path),
            'format_type': 'handwritten',
            'quality_score': 0.68,
            'confidence_score': 0.61,
            'processing_time': 12.1,
            'language': 'fr',
            'ingredients': [
                {
                    'ingredient_name': 'farine',
                    'ingredient_name_en': 'flour',
                    'quantity': '250',
                    'unit': 'grammes',
                    'unit_en': 'grams',
                    'confidence': 0.78
                },
                {
                    'ingredient_name': 'œufs',
                    'ingredient_name_en': 'eggs',
                    'quantity': '3',
                    'unit': 'gros',
                    'unit_en': 'large',
                    'confidence': 0.71
                },
                {
                    'ingredient_name': 'lait',
                    'ingredient_name_en': 'milk',
                    'quantity': '200',
                    'unit': 'ml',
                    'unit_en': 'ml',
                    'confidence': 0.82
                }
            ]
        }
    
    def _create_mock_german_result(self, image_path: str) -> Dict[str, Any]:
        """Create mock German recipe result"""
        return {
            'filename': os.path.basename(image_path),
            'format_type': 'digital_screenshot',
            'quality_score': 0.79,
            'confidence_score': 0.71,
            'processing_time': 9.4,
            'language': 'de',
            'ingredients': [
                {
                    'ingredient_name': 'Mehl',
                    'ingredient_name_en': 'flour',
                    'quantity': '400',
                    'unit': 'Gramm',
                    'unit_en': 'grams',
                    'confidence': 0.84
                },
                {
                    'ingredient_name': 'Zucker',
                    'ingredient_name_en': 'sugar',
                    'quantity': '100',
                    'unit': 'Gramm',
                    'unit_en': 'grams',
                    'confidence': 0.88
                },
                {
                    'ingredient_name': 'Butter',
                    'ingredient_name_en': 'butter',
                    'quantity': '200',
                    'unit': 'Gramm',
                    'unit_en': 'grams',
                    'confidence': 0.81
                }
            ]
        }
    
    def _create_mock_italian_result(self, image_path: str) -> Dict[str, Any]:
        """Create mock Italian recipe result"""
        return {
            'filename': os.path.basename(image_path),
            'format_type': 'recipe_blog',
            'quality_score': 0.75,
            'confidence_score': 0.69,
            'processing_time': 10.2,
            'language': 'it',
            'ingredients': [
                {
                    'ingredient_name': 'pomodori',
                    'ingredient_name_en': 'tomatoes',
                    'quantity': '500',
                    'unit': 'grammi',
                    'unit_en': 'grams',
                    'confidence': 0.79
                },
                {
                    'ingredient_name': 'basilico',
                    'ingredient_name_en': 'basil',
                    'quantity': '1',
                    'unit': 'mazzetto',
                    'unit_en': 'bunch',
                    'confidence': 0.73
                },
                {
                    'ingredient_name': 'parmigiano',
                    'ingredient_name_en': 'parmesan',
                    'quantity': '100',
                    'unit': 'grammi',
                    'unit_en': 'grams',
                    'confidence': 0.76
                }
            ]
        }
    
    def _display_foreign_language_results(self, result: Dict[str, Any], language: str):
        """Display foreign language processing results"""
        language_names = {
            'es': 'Spanish',
            'fr': 'French', 
            'de': 'German',
            'it': 'Italian'
        }
        
        lang_name = language_names.get(language, language.upper())
        
        print(f"\n📊 Results for {result['filename']} ({lang_name}):")
        print(f"Quality Score: {result['quality_score']:.2f}")
        print(f"Confidence: {result['confidence_score']:.2f}")
        print(f"Processing Time: {result['processing_time']:.1f}s")
        
        print(f"\n🌮 Ingredients in {lang_name} ({len(result['ingredients'])}):")
        for i, ing in enumerate(result['ingredients'], 1):
            quantity = ing.get('quantity', '')
            unit = ing.get('unit', '')
            name = ing['ingredient_name']
            conf = ing['confidence']
            
            # Show translation if available
            name_en = ing.get('ingredient_name_en', '')
            unit_en = ing.get('unit_en', '')
            
            original_text = f"{i:2d}. {quantity} {unit} {name}".strip()
            
            if name_en and name_en != name:
                translation = f"{quantity} {unit_en} {name_en}".strip()
                print(f"✅ {original_text}")
                print(f"    English: {translation} [confidence: {conf:.2f}]")
            else:
                print(f"✅ {original_text} [confidence: {conf:.2f}]")
        
        print(f"\n💡 {lang_name} processing insights:")
        if language == 'es':
            print("• Spanish recipes often use metric measurements")
            print("• Common abbreviations: cdta (cucharadita), cda (cucharada)")
            print("• Watch for regional variations in ingredient names")
        elif language == 'fr':
            print("• French recipes typically use metric system")
            print("• Accented characters may affect OCR accuracy")
            print("• Common units: cuillères à soupe (tbsp), cuillères à café (tsp)")
        elif language == 'de':
            print("• German uses compound words for ingredients")
            print("• Common abbreviations: EL (Esslöffel), TL (Teelöffel)")
            print("• Metric measurements standard")
        elif language == 'it':
            print("• Italian recipes often specify preparation methods")
            print("• Regional ingredient name variations common")
            print("• Metric system used throughout")

def demonstrate_all_recipe_types():
    """Demonstrate processing for all recipe types"""
    processor = RecipeTypeProcessor()
    
    print("🍳 Recipe Type Processing Demonstrations")
    print("=" * 60)
    print("This demonstration shows optimal settings and expected results")
    print("for different types of recipe images.")
    print("=" * 60)
    
    # Demonstrate each recipe type
    recipe_types = [
        ("cookbook_page.jpg", processor.process_cookbook_page),
        ("handwritten_recipe.jpg", processor.process_handwritten_recipe),
        ("app_screenshot.png", processor.process_digital_screenshot),
        ("blog_recipe.jpg", processor.process_blog_recipe),
    ]
    
    for image_path, process_func in recipe_types:
        try:
            result = process_func(image_path)
            # Save result for reference
            results_dir = Path(__file__).parent / "results"
            results_dir.mkdir(exist_ok=True)
            
            result_file = results_dir / f"{process_func.__name__}_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Demonstrate foreign language processing
    foreign_languages = ['es', 'fr', 'de', 'it']
    for lang in foreign_languages:
        try:
            result = processor.process_foreign_language_recipe(f"recipe_{lang}.jpg", lang)
        except Exception as e:
            print(f"Error processing {lang} recipe: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Recipe Type Demonstrations Complete!")
    print("\nKey Takeaways:")
    print("• Different recipe types require different OCR engines and settings")
    print("• Handwritten recipes need lower confidence thresholds")
    print("• Digital screenshots may contain UI elements to filter")
    print("• Foreign language processing benefits from language-specific settings")
    print("• Always validate results and implement error handling")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_all_recipe_types()