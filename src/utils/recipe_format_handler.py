"""
Recipe format handler for different types of recipe sources.
Handles handwritten, printed, and digital recipe formats with specialized processing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json


class RecipeFormat(Enum):
    """Recipe format types."""
    HANDWRITTEN = "handwritten"
    PRINTED = "printed"
    DIGITAL = "digital"
    UNKNOWN = "unknown"


@dataclass
class RecipeMetadata:
    """Metadata for recipe images."""
    format_type: RecipeFormat
    confidence: float
    characteristics: Dict[str, any]
    preprocessing_recommendations: List[str]
    ocr_settings: Dict[str, any]


class RecipeFormatClassifier:
    """Classify recipe images by format type."""
    
    def __init__(self):
        """Initialize the classifier."""
        self.feature_extractors = {
            'edge_density': self._calculate_edge_density,
            'text_regularity': self._calculate_text_regularity,
            'color_variance': self._calculate_color_variance,
            'line_straightness': self._calculate_line_straightness,
            'font_consistency': self._calculate_font_consistency
        }
    
    def classify_recipe_format(self, image: np.ndarray) -> RecipeMetadata:
        """
        Classify recipe format based on image characteristics.
        
        Args:
            image: Input image
            
        Returns:
            Recipe metadata with format classification
        """
        features = self._extract_features(image)
        format_type, confidence = self._classify_features(features)
        
        # Get format-specific characteristics
        characteristics = self._analyze_format_characteristics(image, format_type)
        
        # Generate preprocessing recommendations
        preprocessing_recs = self._get_preprocessing_recommendations(format_type, characteristics)
        
        # Generate OCR settings
        ocr_settings = self._get_ocr_settings(format_type, characteristics)
        
        return RecipeMetadata(
            format_type=format_type,
            confidence=confidence,
            characteristics=characteristics,
            preprocessing_recommendations=preprocessing_recs,
            ocr_settings=ocr_settings
        )
    
    def _extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features for format classification."""
        features = {}
        
        for feature_name, extractor in self.feature_extractors.items():
            try:
                features[feature_name] = extractor(image)
            except Exception as e:
                print(f"Failed to extract {feature_name}: {e}")
                features[feature_name] = 0.0
        
        return features
    
    def _calculate_edge_density(self, image: np.ndarray) -> float:
        """Calculate edge density (higher for handwritten)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        
        return edge_pixels / total_pixels
    
    def _calculate_text_regularity(self, image: np.ndarray) -> float:
        """Calculate text line regularity (higher for printed)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find horizontal lines using morphology
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Calculate line consistency
        line_pixels = np.sum(horizontal_lines < 200)
        total_text_pixels = np.sum(gray < 200)
        
        if total_text_pixels == 0:
            return 0.0
        
        return line_pixels / total_text_pixels
    
    def _calculate_color_variance(self, image: np.ndarray) -> float:
        """Calculate color variance (higher for digital)."""
        if len(image.shape) == 3:
            # Calculate variance across color channels
            variance = np.var(image, axis=(0, 1))
            return np.mean(variance)
        else:
            return np.var(image)
    
    def _calculate_line_straightness(self, image: np.ndarray) -> float:
        """Calculate line straightness (higher for printed/digital)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect lines using HoughLines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0.0
        
        # Calculate how many lines are nearly horizontal or vertical
        straight_lines = 0
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            if (angle < 10 or angle > 170) or (80 < angle < 100):
                straight_lines += 1
        
        return straight_lines / len(lines) if len(lines) > 0 else 0.0
    
    def _calculate_font_consistency(self, image: np.ndarray) -> float:
        """Calculate font consistency (higher for printed/digital)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find text regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours (characters)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 5:
            return 0.0
        
        # Calculate consistency in character heights
        heights = []
        for contour in contours:
            _, _, _, h = cv2.boundingRect(contour)
            if h > 10:  # Filter out noise
                heights.append(h)
        
        if len(heights) < 3:
            return 0.0
        
        # Lower coefficient of variation indicates more consistency
        mean_height = np.mean(heights)
        std_height = np.std(heights)
        cv = std_height / mean_height if mean_height > 0 else 1.0
        
        return max(0, 1 - cv)  # Higher score for more consistency
    
    def _classify_features(self, features: Dict[str, float]) -> Tuple[RecipeFormat, float]:
        """Classify format based on extracted features."""
        
        # Define thresholds for classification
        thresholds = {
            'handwritten': {
                'edge_density': (0.05, float('inf')),
                'text_regularity': (0.0, 0.3),
                'line_straightness': (0.0, 0.4),
                'font_consistency': (0.0, 0.5)
            },
            'printed': {
                'edge_density': (0.02, 0.08),
                'text_regularity': (0.3, 0.8),
                'line_straightness': (0.4, 0.9),
                'font_consistency': (0.5, 1.0)
            },
            'digital': {
                'color_variance': (100, float('inf')),
                'text_regularity': (0.6, 1.0),
                'line_straightness': (0.7, 1.0),
                'font_consistency': (0.7, 1.0)
            }
        }
        
        # Calculate scores for each format
        scores = {}
        
        for format_name, format_thresholds in thresholds.items():
            score = 0
            count = 0
            
            for feature, (min_val, max_val) in format_thresholds.items():
                if feature in features:
                    value = features[feature]
                    if min_val <= value <= max_val:
                        # Calculate normalized score within range
                        if max_val == float('inf'):
                            normalized_score = min(1.0, (value - min_val) / (min_val + 0.1))
                        else:
                            range_size = max_val - min_val
                            normalized_score = 1.0 if range_size == 0 else min(1.0, (value - min_val) / range_size)
                        score += normalized_score
                    count += 1
            
            scores[format_name] = score / count if count > 0 else 0
        
        # Determine best match
        best_format = max(scores, key=scores.get)
        confidence = scores[best_format]
        
        # Convert to enum
        format_map = {
            'handwritten': RecipeFormat.HANDWRITTEN,
            'printed': RecipeFormat.PRINTED,
            'digital': RecipeFormat.DIGITAL
        }
        
        if confidence < 0.3:
            return RecipeFormat.UNKNOWN, confidence
        
        return format_map.get(best_format, RecipeFormat.UNKNOWN), confidence
    
    def _analyze_format_characteristics(self, image: np.ndarray, format_type: RecipeFormat) -> Dict[str, any]:
        """Analyze format-specific characteristics."""
        characteristics = {}
        
        if format_type == RecipeFormat.HANDWRITTEN:
            characteristics.update(self._analyze_handwritten_characteristics(image))
        elif format_type == RecipeFormat.PRINTED:
            characteristics.update(self._analyze_printed_characteristics(image))
        elif format_type == RecipeFormat.DIGITAL:
            characteristics.update(self._analyze_digital_characteristics(image))
        
        return characteristics
    
    def _analyze_handwritten_characteristics(self, image: np.ndarray) -> Dict[str, any]:
        """Analyze handwritten recipe characteristics."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return {
            'has_lined_paper': self._detect_lined_paper(gray),
            'ink_type': self._detect_ink_type(image),
            'writing_pressure_variation': self._analyze_pressure_variation(gray),
            'text_slant': self._analyze_text_slant(gray),
            'margin_consistency': self._analyze_margins(gray)
        }
    
    def _analyze_printed_characteristics(self, image: np.ndarray) -> Dict[str, any]:
        """Analyze printed recipe characteristics."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return {
            'print_quality': self._assess_print_quality(gray),
            'has_columns': self._detect_columns(gray),
            'font_size_variation': self._analyze_font_sizes(gray),
            'page_layout': self._analyze_page_layout(gray),
            'scan_artifacts': self._detect_scan_artifacts(gray)
        }
    
    def _analyze_digital_characteristics(self, image: np.ndarray) -> Dict[str, any]:
        """Analyze digital recipe characteristics."""
        return {
            'screen_type': self._detect_screen_type(image),
            'ui_elements': self._detect_ui_elements(image),
            'font_rendering': self._analyze_font_rendering(image),
            'background_type': self._analyze_background(image),
            'compression_artifacts': self._detect_compression_artifacts(image)
        }
    
    def _get_preprocessing_recommendations(self, format_type: RecipeFormat, 
                                         characteristics: Dict[str, any]) -> List[str]:
        """Get preprocessing recommendations based on format."""
        recommendations = []
        
        if format_type == RecipeFormat.HANDWRITTEN:
            recommendations.extend([
                "Apply stronger noise reduction",
                "Use adaptive thresholding",
                "Correct for rotation and skew",
                "Enhance contrast for faded ink"
            ])
            
            if characteristics.get('has_lined_paper'):
                recommendations.append("Remove ruled lines")
            
            if characteristics.get('writing_pressure_variation', 0) > 0.5:
                recommendations.append("Apply pressure normalization")
        
        elif format_type == RecipeFormat.PRINTED:
            recommendations.extend([
                "Apply moderate noise reduction",
                "Use global thresholding",
                "Correct minor rotation",
                "Remove print artifacts"
            ])
            
            if characteristics.get('scan_artifacts'):
                recommendations.append("Remove scan lines and artifacts")
            
            if characteristics.get('print_quality', 1.0) < 0.7:
                recommendations.append("Enhance text clarity")
        
        elif format_type == RecipeFormat.DIGITAL:
            recommendations.extend([
                "Minimal preprocessing required",
                "Remove UI elements if present",
                "Correct screen glare",
                "Handle font antialiasing"
            ])
            
            if characteristics.get('compression_artifacts'):
                recommendations.append("Remove JPEG artifacts")
        
        return recommendations
    
    def _get_ocr_settings(self, format_type: RecipeFormat, 
                         characteristics: Dict[str, any]) -> Dict[str, any]:
        """Get OCR settings optimized for format."""
        base_settings = {
            'confidence_threshold': 0.5,
            'preprocessing': True,
            'language': 'en'
        }
        
        if format_type == RecipeFormat.HANDWRITTEN:
            base_settings.update({
                'confidence_threshold': 0.3,  # Lower threshold for handwriting
                'character_whitelist': None,  # Allow all characters
                'psm_mode': 6,  # Assume uniform block of text
                'preprocessing_aggressive': True
            })
        
        elif format_type == RecipeFormat.PRINTED:
            base_settings.update({
                'confidence_threshold': 0.6,  # Higher threshold for clean print
                'character_whitelist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,/- ()',
                'psm_mode': 4,  # Assume single column of text
                'preprocessing_aggressive': False
            })
        
        elif format_type == RecipeFormat.DIGITAL:
            base_settings.update({
                'confidence_threshold': 0.7,  # Highest threshold for digital text
                'character_whitelist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,/- ()',
                'psm_mode': 3,  # Fully automatic page segmentation
                'preprocessing_aggressive': False
            })
        
        return base_settings
    
    # Helper methods for characteristic analysis
    def _detect_lined_paper(self, gray: np.ndarray) -> bool:
        """Detect if image shows lined paper."""
        # Look for horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Count horizontal line pixels
        line_pixels = np.sum(horizontal_lines < 200)
        return line_pixels > (gray.shape[0] * gray.shape[1] * 0.01)
    
    def _detect_ink_type(self, image: np.ndarray) -> str:
        """Detect ink type (pen, pencil, marker)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Analyze text pixel intensities
        text_pixels = gray[gray < 200]
        if len(text_pixels) == 0:
            return "unknown"
        
        mean_intensity = np.mean(text_pixels)
        intensity_std = np.std(text_pixels)
        
        if mean_intensity < 50 and intensity_std < 30:
            return "pen"
        elif mean_intensity < 100 and intensity_std > 40:
            return "pencil"
        else:
            return "marker"
    
    def _analyze_pressure_variation(self, gray: np.ndarray) -> float:
        """Analyze writing pressure variation."""
        text_pixels = gray[gray < 200]
        if len(text_pixels) == 0:
            return 0.0
        
        # Coefficient of variation in text pixel intensities
        return np.std(text_pixels) / np.mean(text_pixels) if np.mean(text_pixels) > 0 else 0.0
    
    def _analyze_text_slant(self, gray: np.ndarray) -> float:
        """Analyze text slant angle."""
        # Use Hough lines to detect text orientation
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is None:
            return 0.0
        
        angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi - 90
            if -45 < angle < 45:  # Only consider reasonable text angles
                angles.append(angle)
        
        return np.mean(angles) if angles else 0.0
    
    def _analyze_margins(self, gray: np.ndarray) -> float:
        """Analyze margin consistency."""
        # Find text regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Project onto horizontal axis
        horizontal_projection = np.sum(binary, axis=0)
        
        # Find left and right margins
        left_margin = 0
        right_margin = len(horizontal_projection) - 1
        
        threshold = np.max(horizontal_projection) * 0.1
        
        for i, val in enumerate(horizontal_projection):
            if val > threshold:
                left_margin = i
                break
        
        for i in range(len(horizontal_projection) - 1, -1, -1):
            if horizontal_projection[i] > threshold:
                right_margin = i
                break
        
        # Calculate margin consistency (placeholder implementation)
        return 0.5  # Default consistency score
    
    def _assess_print_quality(self, gray: np.ndarray) -> float:
        """Assess print quality."""
        # Calculate local contrast and sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range
        return min(1.0, variance / 1000)
    
    def _detect_columns(self, gray: np.ndarray) -> bool:
        """Detect if text is in columns."""
        # Simplified column detection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Vertical projection
        vertical_projection = np.sum(binary, axis=1)
        
        # Look for gaps that might indicate columns
        mean_projection = np.mean(vertical_projection)
        gaps = vertical_projection < (mean_projection * 0.1)
        
        # Count vertical gaps
        gap_regions = 0
        in_gap = False
        
        for is_gap in gaps:
            if is_gap and not in_gap:
                gap_regions += 1
                in_gap = True
            elif not is_gap:
                in_gap = False
        
        return gap_regions > 2
    
    def _analyze_font_sizes(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze font size variation."""
        # Find contours representing characters
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        heights = []
        for contour in contours:
            _, _, _, h = cv2.boundingRect(contour)
            if 5 < h < 100:  # Filter reasonable character heights
                heights.append(h)
        
        if not heights:
            return {'mean_height': 0, 'height_variation': 0}
        
        return {
            'mean_height': np.mean(heights),
            'height_variation': np.std(heights) / np.mean(heights)
        }
    
    def _analyze_page_layout(self, gray: np.ndarray) -> str:
        """Analyze page layout type."""
        # Simplified layout analysis
        return "single_column"  # Default
    
    def _detect_scan_artifacts(self, gray: np.ndarray) -> bool:
        """Detect scanning artifacts."""
        # Look for horizontal scan lines
        kernel = np.ones((1, 20), np.uint8)
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Count artifact pixels
        artifact_pixels = np.sum(horizontal_lines < 250)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        return artifact_pixels > (total_pixels * 0.005)
    
    def _detect_screen_type(self, image: np.ndarray) -> str:
        """Detect screen type for digital images."""
        # Simplified screen type detection
        return "unknown"
    
    def _detect_ui_elements(self, image: np.ndarray) -> List[str]:
        """Detect UI elements in digital recipes."""
        # Placeholder for UI element detection
        return []
    
    def _analyze_font_rendering(self, image: np.ndarray) -> Dict[str, any]:
        """Analyze font rendering characteristics."""
        return {'antialiased': True, 'subpixel_rendering': False}
    
    def _analyze_background(self, image: np.ndarray) -> str:
        """Analyze background type."""
        return "solid"
    
    def _detect_compression_artifacts(self, image: np.ndarray) -> bool:
        """Detect JPEG compression artifacts."""
        # Simplified artifact detection
        return False


class RecipeFormatProcessor:
    """Process recipes based on their format type."""
    
    def __init__(self):
        """Initialize the processor."""
        self.classifier = RecipeFormatClassifier()
    
    def process_recipe_image(self, image: np.ndarray, 
                           metadata: Optional[RecipeMetadata] = None) -> Tuple[np.ndarray, RecipeMetadata]:
        """
        Process recipe image based on its format.
        
        Args:
            image: Input image
            metadata: Optional pre-classified metadata
            
        Returns:
            Tuple of (processed_image, metadata)
        """
        if metadata is None:
            metadata = self.classifier.classify_recipe_format(image)
        
        # Apply format-specific preprocessing
        processed_image = self._apply_format_preprocessing(image, metadata)
        
        return processed_image, metadata
    
    def _apply_format_preprocessing(self, image: np.ndarray, 
                                   metadata: RecipeMetadata) -> np.ndarray:
        """Apply format-specific preprocessing."""
        processed = image.copy()
        
        for recommendation in metadata.preprocessing_recommendations:
            if "noise reduction" in recommendation:
                processed = self._apply_noise_reduction(processed, metadata.format_type)
            elif "thresholding" in recommendation:
                processed = self._apply_thresholding(processed, metadata.format_type)
            elif "rotation" in recommendation:
                processed = self._correct_rotation(processed)
            elif "contrast" in recommendation:
                processed = self._enhance_contrast(processed)
        
        return processed
    
    def _apply_noise_reduction(self, image: np.ndarray, format_type: RecipeFormat) -> np.ndarray:
        """Apply format-appropriate noise reduction."""
        if format_type == RecipeFormat.HANDWRITTEN:
            # Stronger noise reduction for handwritten
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif format_type == RecipeFormat.PRINTED:
            # Moderate noise reduction for printed
            return cv2.GaussianBlur(image, (3, 3), 0)
        else:
            # Minimal for digital
            return image
    
    def _apply_thresholding(self, image: np.ndarray, format_type: RecipeFormat) -> np.ndarray:
        """Apply format-appropriate thresholding."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if format_type == RecipeFormat.HANDWRITTEN:
            # Adaptive thresholding for handwritten
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        else:
            # Global thresholding for printed/digital
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
    
    def _correct_rotation(self, image: np.ndarray) -> np.ndarray:
        """Correct image rotation."""
        # Simplified rotation correction
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect rotation using Hough lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = theta * 180 / np.pi - 90
                angles.append(angle)
            
            # Find median angle
            median_angle = np.median(angles) if angles else 0
            
            # Correct rotation if significant
            if abs(median_angle) > 1:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                return cv2.warpAffine(image, M, (w, h), borderValue=255)
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast."""
        if len(image.shape) == 3:
            # Convert to LAB and enhance L channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)


def main():
    """Example usage of recipe format handling."""
    
    # Example: Classify recipe format
    classifier = RecipeFormatClassifier()
    processor = RecipeFormatProcessor()
    
    print("Recipe Format Handler")
    print("Usage examples:")
    print("1. Classify recipe format:")
    print("   metadata = classifier.classify_recipe_format(image)")
    print("2. Process based on format:")
    print("   processed_image, metadata = processor.process_recipe_image(image)")
    print("3. Get format-specific OCR settings:")
    print("   ocr_settings = metadata.ocr_settings")


if __name__ == "__main__":
    main()