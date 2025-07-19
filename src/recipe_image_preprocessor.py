#!/usr/bin/env python3
"""
Recipe Image Preprocessor
Advanced image preprocessing pipeline for recipe cards and cookbook pages
with layout detection, rotation correction, and quality enhancement.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from PIL import Image, ImageEnhance, ImageFilter
import json

try:
    from skimage import morphology, measure, filters
    from skimage.transform import hough_line, hough_line_peaks
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


@dataclass
class PreprocessingResult:
    """Result of image preprocessing."""
    success: bool
    processed_image: Optional[np.ndarray]
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    rotation_angle: float
    quality_score: float
    detected_regions: List[Dict[str, Any]]
    preprocessing_steps: List[str]
    metadata: Dict[str, Any]
    processing_time: float


class RecipeImagePreprocessor:
    """Advanced image preprocessor for recipe cards and cookbook pages."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize recipe image preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Preprocessing parameters
        self.max_image_size = self.config.get('max_image_size', 2048)
        self.min_image_size = self.config.get('min_image_size', 300)
        self.rotation_detection_enabled = self.config.get('rotation_detection', True)
        self.noise_reduction_enabled = self.config.get('noise_reduction', True)
        self.contrast_enhancement_enabled = self.config.get('contrast_enhancement', True)
        self.layout_detection_enabled = self.config.get('layout_detection', True)
        
        # Quality thresholds
        self.min_quality_score = self.config.get('min_quality_score', 0.3)
        self.blur_threshold = self.config.get('blur_threshold', 100)
        self.brightness_threshold = self.config.get('brightness_threshold', 30)
        
        self.logger.info("Initialized RecipeImagePreprocessor")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for preprocessor."""
        logger = logging.getLogger('recipe_image_preprocessor')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def preprocess_image(self, image_path: str) -> PreprocessingResult:
        """
        Comprehensive image preprocessing for recipe cards.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PreprocessingResult with processed image and metadata
        """
        import time
        start_time = time.time()
        
        preprocessing_steps = []
        metadata = {}
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return PreprocessingResult(
                    success=False,
                    processed_image=None,
                    original_size=(0, 0),
                    processed_size=(0, 0),
                    rotation_angle=0.0,
                    quality_score=0.0,
                    detected_regions=[],
                    preprocessing_steps=["Failed to load image"],
                    metadata={"error": "Could not load image"},
                    processing_time=time.time() - start_time
                )
            
            original_size = (image.shape[1], image.shape[0])
            processed_image = image.copy()
            preprocessing_steps.append("Image loaded")
            
            # Step 1: Initial quality assessment
            quality_score = self._assess_image_quality(processed_image)
            preprocessing_steps.append(f"Quality assessment: {quality_score:.3f}")
            
            if quality_score < self.min_quality_score:
                self.logger.warning(f"Low quality image: {quality_score:.3f}")
            
            # Step 2: Resize if needed
            if self._needs_resizing(processed_image):
                processed_image = self._resize_image(processed_image)
                preprocessing_steps.append("Image resized")
            
            # Step 3: Color space conversion and normalization
            processed_image = self._normalize_colors(processed_image)
            preprocessing_steps.append("Colors normalized")
            
            # Step 4: Noise reduction
            if self.noise_reduction_enabled:
                processed_image = self._reduce_noise(processed_image)
                preprocessing_steps.append("Noise reduced")
            
            # Step 5: Rotation correction
            rotation_angle = 0.0
            if self.rotation_detection_enabled:
                processed_image, rotation_angle = self._correct_rotation(processed_image)
                preprocessing_steps.append(f"Rotation corrected: {rotation_angle:.2f}°")
            
            # Step 6: Layout detection
            detected_regions = []
            if self.layout_detection_enabled:
                detected_regions = self._detect_layout_regions(processed_image)
                preprocessing_steps.append(f"Layout regions detected: {len(detected_regions)}")
            
            # Step 7: Contrast and brightness enhancement
            if self.contrast_enhancement_enabled:
                processed_image = self._enhance_contrast(processed_image)
                preprocessing_steps.append("Contrast enhanced")
            
            # Step 8: Final quality assessment
            final_quality_score = self._assess_image_quality(processed_image)
            preprocessing_steps.append(f"Final quality: {final_quality_score:.3f}")
            
            # Metadata
            metadata = {
                "original_format": Path(image_path).suffix.lower(),
                "original_file_size": Path(image_path).stat().st_size,
                "quality_improvement": final_quality_score - quality_score,
                "rotation_applied": abs(rotation_angle) > 1.0,
                "regions_detected": len(detected_regions),
                "preprocessing_enabled": {
                    "rotation_detection": self.rotation_detection_enabled,
                    "noise_reduction": self.noise_reduction_enabled,
                    "contrast_enhancement": self.contrast_enhancement_enabled,
                    "layout_detection": self.layout_detection_enabled
                }
            }
            
            processing_time = time.time() - start_time
            
            return PreprocessingResult(
                success=True,
                processed_image=processed_image,
                original_size=original_size,
                processed_size=(processed_image.shape[1], processed_image.shape[0]),
                rotation_angle=rotation_angle,
                quality_score=final_quality_score,
                detected_regions=detected_regions,
                preprocessing_steps=preprocessing_steps,
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return PreprocessingResult(
                success=False,
                processed_image=None,
                original_size=(0, 0),
                processed_size=(0, 0),
                rotation_angle=0.0,
                quality_score=0.0,
                detected_regions=[],
                preprocessing_steps=preprocessing_steps + [f"Error: {str(e)}"],
                metadata={"error": str(e)},
                processing_time=time.time() - start_time
            )
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """
        Assess image quality using multiple metrics.
        
        Args:
            image: Input image
            
        Returns:
            Quality score between 0 and 1
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_normalized = min(blur_score / self.blur_threshold, 1.0)
        
        # Brightness assessment
        brightness = np.mean(gray)
        brightness_normalized = min(brightness / 255.0, 1.0)
        
        # Contrast assessment
        contrast = np.std(gray)
        contrast_normalized = min(contrast / 128.0, 1.0)
        
        # Noise assessment (using local standard deviation)
        kernel = np.ones((3, 3), np.uint8)
        mean_filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel / 9)
        noise_variance = np.mean((gray.astype(np.float32) - mean_filtered) ** 2)
        noise_normalized = max(0, 1.0 - noise_variance / 1000.0)
        
        # Combined quality score
        quality_score = (
            blur_normalized * 0.3 +
            brightness_normalized * 0.2 +
            contrast_normalized * 0.3 +
            noise_normalized * 0.2
        )
        
        return min(max(quality_score, 0.0), 1.0)
    
    def _needs_resizing(self, image: np.ndarray) -> bool:
        """Check if image needs resizing."""
        height, width = image.shape[:2]
        max_dim = max(height, width)
        min_dim = min(height, width)
        
        return max_dim > self.max_image_size or min_dim < self.min_image_size
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        max_dim = max(height, width)
        
        if max_dim > self.max_image_size:
            # Scale down
            scale_factor = self.max_image_size / max_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        elif min(height, width) < self.min_image_size:
            # Scale up
            scale_factor = self.min_image_size / min(height, width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return image
    
    def _normalize_colors(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize colors and correct white balance.
        
        Args:
            image: Input image
            
        Returns:
            Color-normalized image
        """
        # Convert to LAB color space for better color processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge([l, a, b])
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return normalized
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction filters.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Apply bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Apply median blur for salt-and-pepper noise
        denoised = cv2.medianBlur(denoised, 3)
        
        return denoised
    
    def _correct_rotation(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct image rotation.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (corrected_image, rotation_angle)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return image, 0.0
        
        # Calculate rotation angle from detected lines
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Convert to degrees
            angle = np.degrees(theta) - 90
            # Normalize to [-45, 45] range
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
            angles.append(angle)
        
        if not angles:
            return image, 0.0
        
        # Use median angle to avoid outliers
        rotation_angle = np.median(angles)
        
        # Only correct if rotation is significant
        if abs(rotation_angle) < 1.0:
            return image, rotation_angle
        
        # Apply rotation correction
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Calculate new dimensions
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_width = int(height * sin_angle + width * cos_angle)
        new_height = int(height * cos_angle + width * sin_angle)
        
        # Adjust translation
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
        
        # Apply rotation
        corrected_image = cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return corrected_image, rotation_angle
    
    def _detect_layout_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect different layout regions in the recipe image.
        
        Args:
            image: Input image
            
        Returns:
            List of detected regions with metadata
        """
        regions = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and classify regions
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Minimum area threshold
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = w / h
            
            # Classify region type based on characteristics
            region_type = "text"
            confidence = 0.8
            
            if aspect_ratio > 5:
                region_type = "title"
                confidence = 0.9
            elif aspect_ratio < 0.5:
                region_type = "list"
                confidence = 0.7
            elif area > 10000:
                region_type = "paragraph"
                confidence = 0.8
            
            regions.append({
                "type": region_type,
                "bbox": [x, y, x + w, y + h],
                "area": area,
                "aspect_ratio": aspect_ratio,
                "confidence": confidence
            })
        
        # Sort regions by vertical position (top to bottom)
        regions.sort(key=lambda r: r["bbox"][1])
        
        return regions
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast for better OCR performance.
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to PIL Image for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(1.2)  # Increase contrast by 20%
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.1)  # Increase sharpness by 10%
        
        # Convert back to OpenCV format
        enhanced_cv = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
        
        return enhanced_cv
    
    def save_processed_image(self, result: PreprocessingResult, output_path: str) -> bool:
        """
        Save processed image to file.
        
        Args:
            result: Preprocessing result
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        if not result.success or result.processed_image is None:
            return False
        
        try:
            cv2.imwrite(output_path, result.processed_image)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save processed image: {e}")
            return False
    
    def batch_preprocess(self, image_paths: List[str], output_dir: str) -> List[PreprocessingResult]:
        """
        Batch process multiple images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory for processed images
            
        Returns:
            List of preprocessing results
        """
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.preprocess_image(image_path)
            results.append(result)
            
            if result.success:
                # Save processed image
                filename = Path(image_path).stem + "_processed.jpg"
                output_file = output_path / filename
                self.save_processed_image(result, str(output_file))
        
        return results
    
    def get_preprocessing_report(self, results: List[PreprocessingResult]) -> Dict[str, Any]:
        """
        Generate preprocessing report for batch processing.
        
        Args:
            results: List of preprocessing results
            
        Returns:
            Comprehensive preprocessing report
        """
        if not results:
            return {"error": "No results provided"}
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # Calculate statistics
        total_processing_time = sum(r.processing_time for r in results)
        average_quality = sum(r.quality_score for r in successful) / len(successful) if successful else 0
        
        # Rotation statistics
        rotations = [r.rotation_angle for r in successful if abs(r.rotation_angle) > 1.0]
        
        # Region detection statistics
        total_regions = sum(len(r.detected_regions) for r in successful)
        
        return {
            "summary": {
                "total_images": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(results) if results else 0,
                "total_processing_time": total_processing_time,
                "average_processing_time": total_processing_time / len(results) if results else 0
            },
            "quality_metrics": {
                "average_quality_score": average_quality,
                "quality_distribution": {
                    "high_quality": len([r for r in successful if r.quality_score > 0.7]),
                    "medium_quality": len([r for r in successful if 0.4 <= r.quality_score <= 0.7]),
                    "low_quality": len([r for r in successful if r.quality_score < 0.4])
                }
            },
            "rotation_correction": {
                "images_rotated": len(rotations),
                "average_rotation": sum(rotations) / len(rotations) if rotations else 0,
                "max_rotation": max(rotations) if rotations else 0
            },
            "layout_detection": {
                "total_regions_detected": total_regions,
                "average_regions_per_image": total_regions / len(successful) if successful else 0
            },
            "failed_images": [
                {
                    "index": i,
                    "error": r.metadata.get("error", "Unknown error"),
                    "steps_completed": len(r.preprocessing_steps)
                }
                for i, r in enumerate(results) if not r.success
            ]
        }


def main():
    """Main preprocessing script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recipe image preprocessing')
    parser.add_argument('--input', '-i', required=True, help='Input image or directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--config', '-c', help='Configuration file (JSON)')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    parser.add_argument('--report', action='store_true', help='Generate processing report')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize preprocessor
    preprocessor = RecipeImagePreprocessor(config)
    
    try:
        if args.batch:
            # Batch processing
            input_path = Path(args.input)
            if not input_path.is_dir():
                print(f"Error: {args.input} is not a directory")
                return 1
            
            # Find all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_paths = [
                str(p) for p in input_path.rglob('*') 
                if p.suffix.lower() in image_extensions
            ]
            
            if not image_paths:
                print(f"No image files found in {args.input}")
                return 1
            
            print(f"Found {len(image_paths)} images to process")
            
            # Process images
            results = preprocessor.batch_preprocess(image_paths, args.output)
            
            # Generate report
            if args.report:
                report = preprocessor.get_preprocessing_report(results)
                report_file = Path(args.output) / "preprocessing_report.json"
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"Report saved to: {report_file}")
            
            # Print summary
            successful = len([r for r in results if r.success])
            print(f"Processing completed: {successful}/{len(results)} successful")
            
        else:
            # Single image processing
            result = preprocessor.preprocess_image(args.input)
            
            if result.success:
                # Save processed image
                output_path = Path(args.output)
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / f"{Path(args.input).stem}_processed.jpg"
                
                if preprocessor.save_processed_image(result, str(output_file)):
                    print(f"Processed image saved to: {output_file}")
                    print(f"Quality score: {result.quality_score:.3f}")
                    print(f"Rotation corrected: {result.rotation_angle:.2f}°")
                    print(f"Regions detected: {len(result.detected_regions)}")
                    print(f"Processing time: {result.processing_time:.2f}s")
                else:
                    print("Failed to save processed image")
                    return 1
            else:
                print("Image preprocessing failed")
                print(f"Error: {result.metadata.get('error', 'Unknown error')}")
                return 1
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())