#!/usr/bin/env python3
"""
Format-Specific Processor for Various Recipe Text Formats
Handles different recipe formats with specialized preprocessing and detection strategies:
- Printed cookbook pages
- Handwritten recipe cards
- Digital recipe screenshots
- Recipe blog images
- Various ingredient list layouts
- Multiple languages and measurement systems
- Faded or low-quality images
- Mixed text and image content
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
from enum import Enum
import re
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent))

from utils.text_utils import TextPreprocessor


class RecipeFormat(Enum):
    """Enumeration of different recipe formats."""
    PRINTED_COOKBOOK = "printed_cookbook"
    HANDWRITTEN_CARD = "handwritten_card"
    DIGITAL_SCREENSHOT = "digital_screenshot"
    RECIPE_BLOG = "recipe_blog"
    INGREDIENT_LIST_COLUMNS = "ingredient_list_columns"
    INGREDIENT_LIST_BULLETS = "ingredient_list_bullets"
    INGREDIENT_LIST_PARAGRAPH = "ingredient_list_paragraph"
    MULTILINGUAL = "multilingual"
    FADED_LOW_QUALITY = "faded_low_quality"
    MIXED_CONTENT = "mixed_content"
    UNKNOWN = "unknown"


class TextLayout(Enum):
    """Enumeration of text layout types."""
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    BULLET_POINTS = "bullet_points"
    NUMBERED_LIST = "numbered_list"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    MIXED = "mixed"


@dataclass
class FormatAnalysis:
    """Analysis result for recipe format detection."""
    detected_format: RecipeFormat
    confidence: float
    layout_type: TextLayout
    quality_score: float
    language: str
    measurement_system: str
    characteristics: Dict[str, Any]
    preprocessing_recommendations: List[str]


@dataclass
class ProcessingStrategy:
    """Processing strategy for a specific format."""
    format_type: RecipeFormat
    preprocessing_steps: List[str]
    detection_parameters: Dict[str, Any]
    ocr_parameters: Dict[str, Any]
    parsing_parameters: Dict[str, Any]
    expected_accuracy: float


class FormatSpecificProcessor:
    """Main processor for handling various recipe text formats."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the format-specific processor."""
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize text preprocessor
        self.text_preprocessor = TextPreprocessor()
        
        # Format detection parameters
        self.format_classifiers = self._initialize_format_classifiers()
        
        # Language and measurement system detection
        self.language_patterns = self._load_language_patterns()
        self.measurement_systems = self._load_measurement_systems()
        
        # Processing strategies for different formats
        self.processing_strategies = self._initialize_processing_strategies()
        
        self.logger.info("Format-Specific Processor initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the processor."""
        logger = logging.getLogger('format_specific_processor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_recipe_format(self, image_path: str) -> FormatAnalysis:
        """
        Analyze recipe format and determine optimal processing strategy.
        
        Args:
            image_path: Path to the recipe image
            
        Returns:
            Format analysis with processing recommendations
        """
        self.logger.info(f"Analyzing recipe format: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Basic image analysis
        height, width = image.shape[:2]
        
        # Detect format type
        format_type = self._detect_format_type(image)
        
        # Analyze layout
        layout_type = self._analyze_layout(image)
        
        # Assess image quality
        quality_score = self._assess_image_quality(image)
        
        # Detect language
        language = self._detect_language(image)
        
        # Detect measurement system
        measurement_system = self._detect_measurement_system(image)
        
        # Calculate confidence
        confidence = self._calculate_format_confidence(image, format_type)
        
        # Generate characteristics
        characteristics = self._analyze_characteristics(image, format_type)
        
        # Generate preprocessing recommendations
        preprocessing_recommendations = self._generate_preprocessing_recommendations(
            format_type, quality_score, characteristics
        )
        
        return FormatAnalysis(
            detected_format=format_type,
            confidence=confidence,
            layout_type=layout_type,
            quality_score=quality_score,
            language=language,
            measurement_system=measurement_system,
            characteristics=characteristics,
            preprocessing_recommendations=preprocessing_recommendations
        )
    
    def _detect_format_type(self, image: np.ndarray) -> RecipeFormat:
        """Detect the recipe format type."""
        # Image characteristics
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Analyze texture and patterns
        texture_score = self._analyze_texture(gray)
        edge_density = self._calculate_edge_density(gray)
        brightness_variance = np.var(gray)
        
        # Detect text regions for layout analysis
        text_regions = self._detect_text_regions_simple(gray)
        
        # Format classification logic
        if self._is_handwritten(gray, texture_score):
            return RecipeFormat.HANDWRITTEN_CARD
        elif self._is_digital_screenshot(image, edge_density):
            return RecipeFormat.DIGITAL_SCREENSHOT
        elif self._is_recipe_blog(image, text_regions):
            return RecipeFormat.RECIPE_BLOG
        elif self._is_faded_low_quality(gray, brightness_variance):
            return RecipeFormat.FADED_LOW_QUALITY
        elif self._is_mixed_content(image, text_regions):
            return RecipeFormat.MIXED_CONTENT
        elif self._is_printed_cookbook(gray, texture_score, edge_density):
            return RecipeFormat.PRINTED_COOKBOOK
        else:
            return RecipeFormat.UNKNOWN
    
    def _is_handwritten(self, gray: np.ndarray, texture_score: float) -> bool:
        """Detect handwritten text characteristics."""
        # Handwritten text has irregular texture and varying thickness
        return texture_score > 0.7 and self._has_irregular_strokes(gray)
    
    def _is_digital_screenshot(self, image: np.ndarray, edge_density: float) -> bool:
        """Detect digital screenshot characteristics."""
        # Digital screenshots have sharp edges and uniform colors
        return edge_density > 0.5 and self._has_uniform_background(image)
    
    def _is_recipe_blog(self, image: np.ndarray, text_regions: List[Dict]) -> bool:
        """Detect recipe blog characteristics."""
        # Recipe blogs often have mixed content and specific layouts
        return len(text_regions) > 5 and self._has_blog_layout_features(image)
    
    def _is_faded_low_quality(self, gray: np.ndarray, brightness_variance: float) -> bool:
        """Detect faded or low-quality image characteristics."""
        # Low contrast and poor quality indicators
        contrast = np.max(gray) - np.min(gray)
        return contrast < 100 or brightness_variance < 500
    
    def _is_mixed_content(self, image: np.ndarray, text_regions: List[Dict]) -> bool:
        """Detect mixed text and image content."""
        # Look for images within the recipe
        return self._has_embedded_images(image) and len(text_regions) > 3
    
    def _is_printed_cookbook(self, gray: np.ndarray, texture_score: float, edge_density: float) -> bool:
        """Detect printed cookbook characteristics."""
        # Printed text has consistent characteristics
        return texture_score < 0.5 and edge_density < 0.3 and self._has_uniform_text_size(gray)
    
    def _analyze_layout(self, image: np.ndarray) -> TextLayout:
        """Analyze text layout type."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text_regions = self._detect_text_regions_simple(gray)
        
        if not text_regions:
            return TextLayout.MIXED
        
        # Analyze region positions
        y_positions = [region['y'] for region in text_regions]
        x_positions = [region['x'] for region in text_regions]
        
        # Check for column structure
        if self._has_column_structure(x_positions):
            return TextLayout.MULTI_COLUMN
        
        # Check for bullet points or numbered lists
        if self._has_list_structure(text_regions):
            return TextLayout.BULLET_POINTS
        
        # Check for table structure
        if self._has_table_structure(text_regions):
            return TextLayout.TABLE
        
        # Check for paragraph structure
        if self._has_paragraph_structure(text_regions):
            return TextLayout.PARAGRAPH
        
        return TextLayout.SINGLE_COLUMN
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess image quality score (0-1)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Contrast
        contrast = np.max(gray) - np.min(gray)
        
        # Brightness distribution
        brightness_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        brightness_uniformity = 1.0 - np.var(brightness_hist) / np.mean(brightness_hist)
        
        # Noise level (estimated)
        noise_level = self._estimate_noise_level(gray)
        
        # Combine factors
        quality_score = (
            min(sharpness / 500, 1.0) * 0.3 +
            min(contrast / 255, 1.0) * 0.3 +
            brightness_uniformity * 0.2 +
            (1.0 - min(noise_level / 50, 1.0)) * 0.2
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _detect_language(self, image: np.ndarray) -> str:
        """Detect text language."""
        try:
            # Use OCR to get text sample
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text_sample = pytesseract.image_to_string(gray)
            
            # Simple language detection based on patterns
            for lang, patterns in self.language_patterns.items():
                if any(pattern in text_sample.lower() for pattern in patterns):
                    return lang
            
            return "en"  # Default to English
            
        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")
            return "en"
    
    def _detect_measurement_system(self, image: np.ndarray) -> str:
        """Detect measurement system used."""
        try:
            # Use OCR to get text sample
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text_sample = pytesseract.image_to_string(gray).lower()
            
            # Count metric vs imperial units
            metric_count = sum(1 for unit in self.measurement_systems['metric'] if unit in text_sample)
            imperial_count = sum(1 for unit in self.measurement_systems['imperial'] if unit in text_sample)
            
            if metric_count > imperial_count:
                return "metric"
            elif imperial_count > metric_count:
                return "imperial"
            else:
                return "mixed"
                
        except Exception as e:
            self.logger.warning(f"Measurement system detection failed: {e}")
            return "mixed"
    
    def _calculate_format_confidence(self, image: np.ndarray, format_type: RecipeFormat) -> float:
        """Calculate confidence score for format detection."""
        # This is a simplified confidence calculation
        # In practice, you'd use trained classifiers
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Base confidence on format-specific features
        if format_type == RecipeFormat.HANDWRITTEN_CARD:
            return self._calculate_handwritten_confidence(gray)
        elif format_type == RecipeFormat.DIGITAL_SCREENSHOT:
            return self._calculate_digital_confidence(image)
        elif format_type == RecipeFormat.PRINTED_COOKBOOK:
            return self._calculate_printed_confidence(gray)
        else:
            return 0.5  # Default confidence
    
    def _analyze_characteristics(self, image: np.ndarray, format_type: RecipeFormat) -> Dict[str, Any]:
        """Analyze format-specific characteristics."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        
        characteristics = {
            'image_size': (width, height),
            'aspect_ratio': width / height,
            'brightness_mean': np.mean(gray),
            'brightness_std': np.std(gray),
            'contrast': np.max(gray) - np.min(gray),
            'text_regions_count': len(self._detect_text_regions_simple(gray)),
            'edge_density': self._calculate_edge_density(gray),
            'texture_score': self._analyze_texture(gray)
        }
        
        # Add format-specific characteristics
        if format_type == RecipeFormat.HANDWRITTEN_CARD:
            characteristics['stroke_irregularity'] = self._measure_stroke_irregularity(gray)
            characteristics['ink_consistency'] = self._measure_ink_consistency(gray)
        
        elif format_type == RecipeFormat.DIGITAL_SCREENSHOT:
            characteristics['pixel_sharpness'] = self._measure_pixel_sharpness(gray)
            characteristics['compression_artifacts'] = self._detect_compression_artifacts(gray)
        
        elif format_type == RecipeFormat.RECIPE_BLOG:
            characteristics['layout_complexity'] = self._measure_layout_complexity(image)
            characteristics['color_variety'] = self._measure_color_variety(image)
        
        return characteristics
    
    def _generate_preprocessing_recommendations(self, format_type: RecipeFormat, 
                                             quality_score: float, 
                                             characteristics: Dict[str, Any]) -> List[str]:
        """Generate preprocessing recommendations based on format analysis."""
        recommendations = []
        
        # Quality-based recommendations
        if quality_score < 0.5:
            recommendations.append("enhance_contrast")
            recommendations.append("denoise")
        
        if characteristics['brightness_std'] < 30:
            recommendations.append("adaptive_histogram_equalization")
        
        # Format-specific recommendations
        if format_type == RecipeFormat.HANDWRITTEN_CARD:
            recommendations.extend([
                "morphological_closing",
                "skew_correction",
                "stroke_width_normalization"
            ])
        
        elif format_type == RecipeFormat.DIGITAL_SCREENSHOT:
            recommendations.extend([
                "resize_optimization",
                "anti_aliasing",
                "color_normalization"
            ])
        
        elif format_type == RecipeFormat.FADED_LOW_QUALITY:
            recommendations.extend([
                "gamma_correction",
                "unsharp_masking",
                "background_normalization"
            ])
        
        elif format_type == RecipeFormat.MIXED_CONTENT:
            recommendations.extend([
                "region_based_processing",
                "content_aware_enhancement",
                "selective_sharpening"
            ])
        
        return recommendations
    
    def apply_format_specific_preprocessing(self, image: np.ndarray, 
                                          format_analysis: FormatAnalysis) -> np.ndarray:
        """Apply format-specific preprocessing to the image."""
        processed_image = image.copy()
        
        for recommendation in format_analysis.preprocessing_recommendations:
            if recommendation == "enhance_contrast":
                processed_image = self._enhance_contrast(processed_image)
            elif recommendation == "denoise":
                processed_image = self._denoise_image(processed_image)
            elif recommendation == "adaptive_histogram_equalization":
                processed_image = self._adaptive_histogram_equalization(processed_image)
            elif recommendation == "morphological_closing":
                processed_image = self._morphological_closing(processed_image)
            elif recommendation == "skew_correction":
                processed_image = self._correct_skew(processed_image)
            elif recommendation == "stroke_width_normalization":
                processed_image = self._normalize_stroke_width(processed_image)
            elif recommendation == "resize_optimization":
                processed_image = self._optimize_resize(processed_image)
            elif recommendation == "anti_aliasing":
                processed_image = self._apply_anti_aliasing(processed_image)
            elif recommendation == "color_normalization":
                processed_image = self._normalize_colors(processed_image)
            elif recommendation == "gamma_correction":
                processed_image = self._gamma_correction(processed_image)
            elif recommendation == "unsharp_masking":
                processed_image = self._unsharp_masking(processed_image)
            elif recommendation == "background_normalization":
                processed_image = self._normalize_background(processed_image)
            elif recommendation == "region_based_processing":
                processed_image = self._region_based_processing(processed_image)
            elif recommendation == "content_aware_enhancement":
                processed_image = self._content_aware_enhancement(processed_image)
            elif recommendation == "selective_sharpening":
                processed_image = self._selective_sharpening(processed_image)
        
        return processed_image
    
    def get_format_specific_detection_parameters(self, format_analysis: FormatAnalysis) -> Dict[str, Any]:
        """Get optimized detection parameters for the specific format."""
        base_params = {
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'max_detections': 100
        }
        
        format_type = format_analysis.detected_format
        
        if format_type == RecipeFormat.HANDWRITTEN_CARD:
            base_params.update({
                'confidence_threshold': 0.15,  # Lower threshold for handwritten
                'iou_threshold': 0.3,          # More lenient overlap
                'max_detections': 50
            })
        
        elif format_type == RecipeFormat.DIGITAL_SCREENSHOT:
            base_params.update({
                'confidence_threshold': 0.3,   # Higher threshold for clean text
                'iou_threshold': 0.5,          # Standard overlap
                'max_detections': 200
            })
        
        elif format_type == RecipeFormat.FADED_LOW_QUALITY:
            base_params.update({
                'confidence_threshold': 0.1,   # Very low threshold
                'iou_threshold': 0.2,          # Very lenient overlap
                'max_detections': 150
            })
        
        elif format_type == RecipeFormat.MIXED_CONTENT:
            base_params.update({
                'confidence_threshold': 0.2,   # Lower threshold for mixed content
                'iou_threshold': 0.4,          # Moderate overlap
                'max_detections': 300
            })
        
        return base_params
    
    def get_format_specific_ocr_parameters(self, format_analysis: FormatAnalysis) -> Dict[str, Any]:
        """Get optimized OCR parameters for the specific format."""
        base_params = {
            'psm': 6,  # Uniform block of text
            'oem': 3,  # Default OCR Engine Mode
            'language': format_analysis.language,
            'dpi': 300
        }
        
        format_type = format_analysis.detected_format
        
        if format_type == RecipeFormat.HANDWRITTEN_CARD:
            base_params.update({
                'psm': 8,  # Single word
                'oem': 1,  # Neural nets LSTM engine
                'dpi': 600  # Higher DPI for handwritten text
            })
        
        elif format_type == RecipeFormat.DIGITAL_SCREENSHOT:
            base_params.update({
                'psm': 6,  # Uniform block of text
                'oem': 3,  # Default engine
                'dpi': 150  # Lower DPI for digital content
            })
        
        elif format_type == RecipeFormat.RECIPE_BLOG:
            base_params.update({
                'psm': 4,  # Single column of text
                'oem': 3,  # Default engine
                'dpi': 200
            })
        
        elif format_type == RecipeFormat.FADED_LOW_QUALITY:
            base_params.update({
                'psm': 8,  # Single word
                'oem': 1,  # Neural nets LSTM engine
                'dpi': 600  # Higher DPI for better quality
            })
        
        # Add language-specific parameters
        if format_analysis.language != 'en':
            base_params['language'] = format_analysis.language
        
        return base_params
    
    def _initialize_format_classifiers(self) -> Dict[str, Any]:
        """Initialize format classification models."""
        # In a real implementation, these would be trained classifiers
        return {
            'handwritten_classifier': None,
            'digital_classifier': None,
            'quality_classifier': None,
            'layout_classifier': None
        }
    
    def _load_language_patterns(self) -> Dict[str, List[str]]:
        """Load language detection patterns."""
        return {
            'en': ['cup', 'tablespoon', 'teaspoon', 'ounce', 'pound', 'ingredients', 'recipe'],
            'es': ['taza', 'cucharada', 'cucharadita', 'onza', 'libra', 'ingredientes', 'receta'],
            'fr': ['tasse', 'cuillère', 'once', 'livre', 'ingrédients', 'recette'],
            'de': ['tasse', 'esslöffel', 'teelöffel', 'unze', 'pfund', 'zutaten', 'rezept'],
            'it': ['tazza', 'cucchiaio', 'cucchiaino', 'oncia', 'libbra', 'ingredienti', 'ricetta'],
            'pt': ['xícara', 'colher', 'onça', 'libra', 'ingredientes', 'receita'],
            'ru': ['стакан', 'ложка', 'чайная', 'унция', 'фунт', 'ингредиенты', 'рецепт'],
            'ja': ['カップ', 'スプーン', 'オンス', 'ポンド', '材料', 'レシピ'],
            'ko': ['컵', '스푼', '온스', '파운드', '재료', '레시피'],
            'zh': ['杯', '勺', '盎司', '磅', '配料', '食谱']
        }
    
    def _load_measurement_systems(self) -> Dict[str, List[str]]:
        """Load measurement system patterns."""
        return {
            'metric': ['g', 'kg', 'gram', 'grams', 'kilogram', 'kilograms', 'ml', 'l', 'liter', 'liters', 'milliliter', 'milliliters'],
            'imperial': ['oz', 'lb', 'lbs', 'ounce', 'ounces', 'pound', 'pounds', 'cup', 'cups', 'tbsp', 'tsp', 'tablespoon', 'teaspoon', 'pint', 'quart', 'gallon']
        }
    
    def _initialize_processing_strategies(self) -> Dict[RecipeFormat, ProcessingStrategy]:
        """Initialize processing strategies for different formats."""
        return {
            RecipeFormat.PRINTED_COOKBOOK: ProcessingStrategy(
                format_type=RecipeFormat.PRINTED_COOKBOOK,
                preprocessing_steps=['enhance_contrast', 'denoise'],
                detection_parameters={'confidence_threshold': 0.25, 'iou_threshold': 0.45},
                ocr_parameters={'psm': 6, 'oem': 3, 'dpi': 300},
                parsing_parameters={'min_confidence': 0.5},
                expected_accuracy=0.85
            ),
            RecipeFormat.HANDWRITTEN_CARD: ProcessingStrategy(
                format_type=RecipeFormat.HANDWRITTEN_CARD,
                preprocessing_steps=['morphological_closing', 'skew_correction', 'stroke_width_normalization'],
                detection_parameters={'confidence_threshold': 0.15, 'iou_threshold': 0.3},
                ocr_parameters={'psm': 8, 'oem': 1, 'dpi': 600},
                parsing_parameters={'min_confidence': 0.3},
                expected_accuracy=0.65
            ),
            RecipeFormat.DIGITAL_SCREENSHOT: ProcessingStrategy(
                format_type=RecipeFormat.DIGITAL_SCREENSHOT,
                preprocessing_steps=['resize_optimization', 'color_normalization'],
                detection_parameters={'confidence_threshold': 0.3, 'iou_threshold': 0.5},
                ocr_parameters={'psm': 6, 'oem': 3, 'dpi': 150},
                parsing_parameters={'min_confidence': 0.6},
                expected_accuracy=0.9
            ),
            RecipeFormat.FADED_LOW_QUALITY: ProcessingStrategy(
                format_type=RecipeFormat.FADED_LOW_QUALITY,
                preprocessing_steps=['gamma_correction', 'unsharp_masking', 'background_normalization'],
                detection_parameters={'confidence_threshold': 0.1, 'iou_threshold': 0.2},
                ocr_parameters={'psm': 8, 'oem': 1, 'dpi': 600},
                parsing_parameters={'min_confidence': 0.2},
                expected_accuracy=0.55
            )
        }
    
    # Helper methods for format detection
    def _analyze_texture(self, gray: np.ndarray) -> float:
        """Analyze texture using local binary patterns."""
        # Simplified texture analysis
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_score = np.var(laplacian) / 10000  # Normalize
        return min(texture_score, 1.0)
    
    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge density in the image."""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density
    
    def _detect_text_regions_simple(self, gray: np.ndarray) -> List[Dict]:
        """Simple text region detection."""
        # Use morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 10:  # Minimum size filter
                text_regions.append({'x': x, 'y': y, 'w': w, 'h': h})
        
        return text_regions
    
    def _has_irregular_strokes(self, gray: np.ndarray) -> bool:
        """Check for irregular stroke patterns (handwritten indicator)."""
        # Simplified check - look for varying stroke widths
        edges = cv2.Canny(gray, 50, 150)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(edges)
        
        # Analyze stroke width variation
        stroke_widths = []
        for label in range(1, num_labels):
            mask = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 50:
                    x, y, w, h = cv2.boundingRect(contour)
                    stroke_widths.append(min(w, h))
        
        if len(stroke_widths) > 5:
            return np.std(stroke_widths) > np.mean(stroke_widths) * 0.3
        
        return False
    
    def _has_uniform_background(self, image: np.ndarray) -> bool:
        """Check for uniform background (digital screenshot indicator)."""
        # Analyze background uniformity
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sample background regions (corners)
        h, w = gray.shape
        corner_size = min(h, w) // 10
        
        corners = [
            gray[:corner_size, :corner_size],
            gray[:corner_size, -corner_size:],
            gray[-corner_size:, :corner_size],
            gray[-corner_size:, -corner_size:]
        ]
        
        corner_means = [np.mean(corner) for corner in corners]
        return np.std(corner_means) < 20  # Low variation in corners
    
    def _has_blog_layout_features(self, image: np.ndarray) -> bool:
        """Check for blog layout features."""
        # Blog layouts often have specific characteristics
        # This is a simplified check
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check for horizontal lines (separators)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Check for multiple text regions with varying sizes
        text_regions = self._detect_text_regions_simple(gray)
        
        if len(text_regions) > 8:
            sizes = [region['w'] * region['h'] for region in text_regions]
            return np.std(sizes) > np.mean(sizes) * 0.5
        
        return False
    
    def _has_embedded_images(self, image: np.ndarray) -> bool:
        """Check for embedded images within the recipe."""
        # Look for high-frequency content that might be images
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection to find potential image regions
        edges = cv2.Canny(gray, 50, 150)
        
        # Find large connected components that might be images
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        large_regions = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > gray.size * 0.05:  # More than 5% of image area
                large_regions += 1
        
        return large_regions > 0
    
    def _has_uniform_text_size(self, gray: np.ndarray) -> bool:
        """Check for uniform text size (printed text indicator)."""
        text_regions = self._detect_text_regions_simple(gray)
        
        if len(text_regions) > 3:
            heights = [region['h'] for region in text_regions]
            return np.std(heights) < np.mean(heights) * 0.3
        
        return True
    
    def _has_column_structure(self, x_positions: List[int]) -> bool:
        """Check for column structure in text layout."""
        if len(x_positions) < 4:
            return False
        
        # Cluster x positions to find columns
        x_array = np.array(x_positions).reshape(-1, 1)
        
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(x_array)
            
            # Check if we have reasonable separation
            cluster_0_x = x_array[clusters == 0]
            cluster_1_x = x_array[clusters == 1]
            
            if len(cluster_0_x) > 1 and len(cluster_1_x) > 1:
                separation = abs(np.mean(cluster_0_x) - np.mean(cluster_1_x))
                return separation > 100  # Minimum separation for columns
        except:
            pass
        
        return False
    
    def _has_list_structure(self, text_regions: List[Dict]) -> bool:
        """Check for list structure (bullets, numbers)."""
        if len(text_regions) < 3:
            return False
        
        # Check for regular vertical spacing
        y_positions = sorted([region['y'] for region in text_regions])
        
        if len(y_positions) > 2:
            spacings = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
            return np.std(spacings) < np.mean(spacings) * 0.5
        
        return False
    
    def _has_table_structure(self, text_regions: List[Dict]) -> bool:
        """Check for table structure."""
        if len(text_regions) < 6:
            return False
        
        # Check for grid-like arrangement
        x_positions = [region['x'] for region in text_regions]
        y_positions = [region['y'] for region in text_regions]
        
        # Look for alignment in both x and y directions
        x_unique = len(set(x_positions))
        y_unique = len(set(y_positions))
        
        return x_unique >= 2 and y_unique >= 3
    
    def _has_paragraph_structure(self, text_regions: List[Dict]) -> bool:
        """Check for paragraph structure."""
        if len(text_regions) < 2:
            return False
        
        # Paragraphs have similar x positions but varying y positions
        x_positions = [region['x'] for region in text_regions]
        
        return np.std(x_positions) < 50  # Similar x positions
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level in the image."""
        # Use Laplacian to estimate noise
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = np.var(laplacian)
        return noise_level / 1000  # Normalize
    
    def _calculate_handwritten_confidence(self, gray: np.ndarray) -> float:
        """Calculate confidence for handwritten text detection."""
        # Simplified confidence calculation
        irregularity = self._has_irregular_strokes(gray)
        texture_score = self._analyze_texture(gray)
        
        if irregularity and texture_score > 0.5:
            return 0.8
        elif irregularity or texture_score > 0.7:
            return 0.6
        else:
            return 0.3
    
    def _calculate_digital_confidence(self, image: np.ndarray) -> float:
        """Calculate confidence for digital screenshot detection."""
        uniform_bg = self._has_uniform_background(image)
        edge_density = self._calculate_edge_density(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        if uniform_bg and edge_density > 0.3:
            return 0.9
        elif uniform_bg or edge_density > 0.5:
            return 0.7
        else:
            return 0.4
    
    def _calculate_printed_confidence(self, gray: np.ndarray) -> float:
        """Calculate confidence for printed text detection."""
        uniform_text = self._has_uniform_text_size(gray)
        texture_score = self._analyze_texture(gray)
        
        if uniform_text and texture_score < 0.3:
            return 0.8
        elif uniform_text or texture_score < 0.5:
            return 0.6
        else:
            return 0.4
    
    def _measure_stroke_irregularity(self, gray: np.ndarray) -> float:
        """Measure stroke irregularity for handwritten text."""
        # Simplified measurement
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        irregularities = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                # Calculate contour irregularity
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(contour)
                
                if hull_area > 0:
                    irregularities.append(1.0 - (contour_area / hull_area))
        
        return np.mean(irregularities) if irregularities else 0.0
    
    def _measure_ink_consistency(self, gray: np.ndarray) -> float:
        """Measure ink consistency for handwritten text."""
        # Analyze intensity variations
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = gray[edges > 0]
        
        if len(edge_pixels) > 100:
            return 1.0 - (np.std(edge_pixels) / 255.0)
        
        return 0.5
    
    def _measure_pixel_sharpness(self, gray: np.ndarray) -> float:
        """Measure pixel sharpness for digital content."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian) / 10000
    
    def _detect_compression_artifacts(self, gray: np.ndarray) -> float:
        """Detect compression artifacts in digital images."""
        # Look for block artifacts (8x8 DCT blocks)
        h, w = gray.shape
        block_size = 8
        
        artifacts = 0
        total_blocks = 0
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                
                # Check for blocking artifacts
                horizontal_diff = np.mean(np.abs(np.diff(block, axis=1)))
                vertical_diff = np.mean(np.abs(np.diff(block, axis=0)))
                
                if horizontal_diff < 5 or vertical_diff < 5:
                    artifacts += 1
                
                total_blocks += 1
        
        return artifacts / total_blocks if total_blocks > 0 else 0.0
    
    def _measure_layout_complexity(self, image: np.ndarray) -> float:
        """Measure layout complexity for blog-style content."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text_regions = self._detect_text_regions_simple(gray)
        
        if len(text_regions) == 0:
            return 0.0
        
        # Calculate complexity based on region size variation
        areas = [region['w'] * region['h'] for region in text_regions]
        aspect_ratios = [region['w'] / region['h'] for region in text_regions]
        
        area_variation = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
        aspect_variation = np.std(aspect_ratios) / np.mean(aspect_ratios) if np.mean(aspect_ratios) > 0 else 0
        
        return (area_variation + aspect_variation) / 2
    
    def _measure_color_variety(self, image: np.ndarray) -> float:
        """Measure color variety in the image."""
        # Convert to LAB color space for better color analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate color histogram
        hist_l = cv2.calcHist([lab], [0], None, [256], [0, 256])
        hist_a = cv2.calcHist([lab], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([lab], [2], None, [256], [0, 256])
        
        # Calculate entropy (measure of color variety)
        def entropy(hist):
            hist = hist.flatten()
            hist = hist[hist > 0]
            hist = hist / np.sum(hist)
            return -np.sum(hist * np.log2(hist))
        
        return (entropy(hist_l) + entropy(hist_a) + entropy(hist_b)) / 3
    
    # Preprocessing methods
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Denoise the image."""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def _adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization."""
        # Convert to YUV color space
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # Apply CLAHE to Y channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        
        # Convert back to BGR
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    def _morphological_closing(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological closing to connect text strokes."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR
        return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    
    def _correct_skew(self, image: np.ndarray) -> np.ndarray:
        """Correct skew in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect skew angle
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                angles.append(angle)
            
            if angles:
                skew_angle = np.median(angles)
                
                # Correct skew if angle is significant
                if abs(skew_angle) > 0.5:
                    h, w = image.shape[:2]
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                    return cv2.warpAffine(image, rotation_matrix, (w, h), 
                                        flags=cv2.INTER_CUBIC, 
                                        borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def _normalize_stroke_width(self, image: np.ndarray) -> np.ndarray:
        """Normalize stroke width for handwritten text."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to normalize stroke width
        kernel_size = 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply opening followed by closing
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        normalized = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    
    def _optimize_resize(self, image: np.ndarray) -> np.ndarray:
        """Optimize image size for digital content."""
        h, w = image.shape[:2]
        
        # Target size for optimal OCR (around 300 DPI equivalent)
        target_height = 800
        
        if h != target_height:
            scale_factor = target_height / h
            new_width = int(w * scale_factor)
            
            return cv2.resize(image, (new_width, target_height), 
                            interpolation=cv2.INTER_CUBIC)
        
        return image
    
    def _apply_anti_aliasing(self, image: np.ndarray) -> np.ndarray:
        """Apply anti-aliasing to smooth edges."""
        # Use Gaussian blur for anti-aliasing
        return cv2.GaussianBlur(image, (3, 3), 0.5)
    
    def _normalize_colors(self, image: np.ndarray) -> np.ndarray:
        """Normalize colors for consistent processing."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Normalize L channel
        l_channel = lab[:, :, 0]
        l_normalized = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
        lab[:, :, 0] = l_normalized
        
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _gamma_correction(self, image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        """Apply gamma correction to enhance faded images."""
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        
        # Apply gamma correction
        return cv2.LUT(image, table)
    
    def _unsharp_masking(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking to enhance details."""
        # Create Gaussian blur
        gaussian = cv2.GaussianBlur(image, (9, 9), 2.0)
        
        # Create unsharp mask
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    
    def _normalize_background(self, image: np.ndarray) -> np.ndarray:
        """Normalize background for better text contrast."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Estimate background using morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Subtract background
        normalized = cv2.subtract(gray, background)
        
        # Enhance contrast
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    
    def _region_based_processing(self, image: np.ndarray) -> np.ndarray:
        """Apply region-based processing for mixed content."""
        # This is a simplified version - would need more sophisticated region detection
        return self._enhance_contrast(image)
    
    def _content_aware_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply content-aware enhancement."""
        # This is a simplified version - would need more sophisticated content analysis
        return self._adaptive_histogram_equalization(image)
    
    def _selective_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply selective sharpening to text regions."""
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        # Apply sharpening
        return cv2.filter2D(image, -1, kernel)


def main():
    """Main function for format-specific processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Format-Specific Recipe Text Processor')
    parser.add_argument('input_image', help='Input recipe image path')
    parser.add_argument('--output', '-o', help='Output directory for processed image')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze format, do not process')
    parser.add_argument('--config', '-c', help='Configuration file (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize processor
    processor = FormatSpecificProcessor(config)
    
    try:
        # Analyze format
        format_analysis = processor.analyze_recipe_format(args.input_image)
        
        # Print analysis results
        print(f"Format Analysis Results:")
        print(f"========================")
        print(f"Detected Format: {format_analysis.detected_format.value}")
        print(f"Confidence: {format_analysis.confidence:.3f}")
        print(f"Layout Type: {format_analysis.layout_type.value}")
        print(f"Quality Score: {format_analysis.quality_score:.3f}")
        print(f"Language: {format_analysis.language}")
        print(f"Measurement System: {format_analysis.measurement_system}")
        
        if args.verbose:
            print(f"\nCharacteristics:")
            for key, value in format_analysis.characteristics.items():
                print(f"  {key}: {value}")
            
            print(f"\nPreprocessing Recommendations:")
            for rec in format_analysis.preprocessing_recommendations:
                print(f"  - {rec}")
        
        # Process image if not analyze-only
        if not args.analyze_only:
            # Load and process image
            image = cv2.imread(args.input_image)
            if image is None:
                print(f"Error: Could not load image {args.input_image}")
                return 1
            
            # Apply format-specific preprocessing
            processed_image = processor.apply_format_specific_preprocessing(image, format_analysis)
            
            # Save processed image
            if args.output:
                output_path = Path(args.output)
                output_path.mkdir(parents=True, exist_ok=True)
                
                input_name = Path(args.input_image).stem
                output_file = output_path / f"{input_name}_processed.jpg"
                
                cv2.imwrite(str(output_file), processed_image)
                print(f"Processed image saved to: {output_file}")
            
            # Get optimized parameters
            detection_params = processor.get_format_specific_detection_parameters(format_analysis)
            ocr_params = processor.get_format_specific_ocr_parameters(format_analysis)
            
            print(f"\nOptimized Detection Parameters:")
            for key, value in detection_params.items():
                print(f"  {key}: {value}")
            
            print(f"\nOptimized OCR Parameters:")
            for key, value in ocr_params.items():
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())