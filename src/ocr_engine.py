#!/usr/bin/env python3
"""
Multi-engine OCR system for text extraction from recipe images.
Supports EasyOCR, Tesseract, and PaddleOCR with fallback mechanisms.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re

# OCR engine imports with graceful fallbacks
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


@dataclass
class OCRResult:
    """OCR extraction result with confidence and metadata."""
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    engine_used: str = ""
    processing_time: float = 0.0
    language: str = "en"


class OCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    @abstractmethod
    def extract_text(self, image: np.ndarray, **kwargs) -> OCRResult:
        """Extract text from image."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if engine is available."""
        pass


class EasyOCREngine(OCREngine):
    """EasyOCR implementation."""
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = False):
        """
        Initialize EasyOCR engine.
        
        Args:
            languages: List of language codes
            gpu: Whether to use GPU acceleration
        """
        self.languages = languages
        self.gpu = gpu
        self.reader = None
        
        if self.is_available():
            try:
                self.reader = easyocr.Reader(languages, gpu=gpu)
            except Exception as e:
                logging.warning(f"Failed to initialize EasyOCR: {e}")
                self.reader = None
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available."""
        return EASYOCR_AVAILABLE
    
    def extract_text(self, image: np.ndarray, **kwargs) -> OCRResult:
        """Extract text using EasyOCR."""
        if not self.reader:
            return OCRResult("", 0.0, engine_used="easyocr_failed")
        
        try:
            import time
            start_time = time.time()
            
            # EasyOCR expects RGB format
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Extract text
            results = self.reader.readtext(image_rgb)
            processing_time = time.time() - start_time
            
            if not results:
                return OCRResult("", 0.0, engine_used="easyocr", processing_time=processing_time)
            
            # Combine all text with confidence weighting
            all_text = []
            total_confidence = 0.0
            
            for bbox, text, confidence in results:
                if confidence > 0.1:  # Filter very low confidence
                    all_text.append(text)
                    total_confidence += confidence
            
            combined_text = " ".join(all_text)
            avg_confidence = total_confidence / len(results) if results else 0.0
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                engine_used="easyocr",
                processing_time=processing_time
            )
            
        except Exception as e:
            logging.error(f"EasyOCR extraction failed: {e}")
            return OCRResult("", 0.0, engine_used="easyocr_error")


class TesseractEngine(OCREngine):
    """Tesseract OCR implementation."""
    
    def __init__(self, config: str = "--oem 3 --psm 6"):
        """
        Initialize Tesseract engine.
        
        Args:
            config: Tesseract configuration string
        """
        self.config = config
    
    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        return TESSERACT_AVAILABLE
    
    def extract_text(self, image: np.ndarray, **kwargs) -> OCRResult:
        """Extract text using Tesseract."""
        if not self.is_available():
            return OCRResult("", 0.0, engine_used="tesseract_unavailable")
        
        try:
            import time
            start_time = time.time()
            
            # Get custom config if provided
            config = kwargs.get('config', self.config)
            
            # Extract text
            text = pytesseract.image_to_string(image, config=config).strip()
            
            # Get confidence data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            avg_confidence = avg_confidence / 100.0  # Convert to 0-1 range
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=text,
                confidence=avg_confidence,
                engine_used="tesseract",
                processing_time=processing_time
            )
            
        except Exception as e:
            logging.error(f"Tesseract extraction failed: {e}")
            return OCRResult("", 0.0, engine_used="tesseract_error")


class PaddleOCREngine(OCREngine):
    """PaddleOCR implementation."""
    
    def __init__(self, lang: str = 'en', use_gpu: bool = False):
        """
        Initialize PaddleOCR engine.
        
        Args:
            lang: Language code
            use_gpu: Whether to use GPU acceleration
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr = None
        
        if self.is_available():
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)
            except Exception as e:
                logging.warning(f"Failed to initialize PaddleOCR: {e}")
                self.ocr = None
    
    def is_available(self) -> bool:
        """Check if PaddleOCR is available."""
        return PADDLEOCR_AVAILABLE
    
    def extract_text(self, image: np.ndarray, **kwargs) -> OCRResult:
        """Extract text using PaddleOCR."""
        if not self.ocr:
            return OCRResult("", 0.0, engine_used="paddleocr_failed")
        
        try:
            import time
            start_time = time.time()
            
            # PaddleOCR expects BGR format (OpenCV default)
            results = self.ocr.ocr(image, cls=True)
            processing_time = time.time() - start_time
            
            if not results or not results[0]:
                return OCRResult("", 0.0, engine_used="paddleocr", processing_time=processing_time)
            
            # Extract text and confidence
            all_text = []
            total_confidence = 0.0
            
            for line in results[0]:
                if line and len(line) >= 2:
                    text, confidence = line[1]
                    if confidence > 0.1:  # Filter very low confidence
                        all_text.append(text)
                        total_confidence += confidence
            
            combined_text = " ".join(all_text)
            avg_confidence = total_confidence / len(results[0]) if results[0] else 0.0
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                engine_used="paddleocr",
                processing_time=processing_time
            )
            
        except Exception as e:
            logging.error(f"PaddleOCR extraction failed: {e}")
            return OCRResult("", 0.0, engine_used="paddleocr_error")


class MultiEngineOCR:
    """Multi-engine OCR system with fallback support."""
    
    def __init__(self, engines: Optional[List[str]] = None, primary_engine: str = "easyocr"):
        """
        Initialize multi-engine OCR system.
        
        Args:
            engines: List of engine names to use
            primary_engine: Primary engine to try first
        """
        self.primary_engine = primary_engine
        self.engines = {}
        
        # Default engines
        if engines is None:
            engines = ["easyocr", "tesseract", "paddleocr"]
        
        # Initialize available engines
        self._initialize_engines(engines)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        if not self.engines:
            self.logger.warning("No OCR engines available!")
    
    def _initialize_engines(self, engine_names: List[str]):
        """Initialize OCR engines."""
        engine_classes = {
            "easyocr": EasyOCREngine,
            "tesseract": TesseractEngine,
            "paddleocr": PaddleOCREngine
        }
        
        for engine_name in engine_names:
            if engine_name in engine_classes:
                try:
                    engine = engine_classes[engine_name]()
                    if engine.is_available():
                        self.engines[engine_name] = engine
                        self.logger.info(f"Initialized {engine_name} engine")
                    else:
                        self.logger.warning(f"{engine_name} engine not available")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize {engine_name}: {e}")
    
    def extract_text(self, image: np.ndarray, 
                    engine: Optional[str] = None,
                    fallback: bool = True,
                    min_confidence: float = 0.3) -> OCRResult:
        """
        Extract text from image using specified or best available engine.
        
        Args:
            image: Input image
            engine: Specific engine to use (None for primary)
            fallback: Whether to try other engines if primary fails
            min_confidence: Minimum confidence threshold for success
            
        Returns:
            OCR result
        """
        if not self.engines:
            return OCRResult("", 0.0, engine_used="no_engines_available")
        
        # Preprocess image for better OCR
        processed_image = self._preprocess_image(image)
        
        # Determine engine order
        if engine and engine in self.engines:
            engine_order = [engine]
            if fallback:
                engine_order.extend([e for e in self.engines.keys() if e != engine])
        else:
            # Use primary engine first, then others
            engine_order = [self.primary_engine] if self.primary_engine in self.engines else []
            engine_order.extend([e for e in self.engines.keys() if e != self.primary_engine])
        
        best_result = None
        
        for engine_name in engine_order:
            if engine_name not in self.engines:
                continue
            
            try:
                result = self.engines[engine_name].extract_text(processed_image)
                
                # Check if result is good enough
                if result.confidence >= min_confidence and result.text.strip():
                    self.logger.info(f"OCR successful with {engine_name}: confidence={result.confidence:.2f}")
                    return result
                
                # Keep track of best result so far
                if best_result is None or result.confidence > best_result.confidence:
                    best_result = result
                
                # If fallback is disabled, return first result
                if not fallback:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Engine {engine_name} failed: {e}")
                continue
        
        # Return best result if no engine met the confidence threshold
        if best_result:
            self.logger.info(f"Best OCR result from {best_result.engine_used}: confidence={best_result.confidence:.2f}")
            return best_result
        
        return OCRResult("", 0.0, engine_used="all_engines_failed")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply preprocessing steps
        processed = gray
        
        # 1. Resize if too small
        h, w = processed.shape
        if h < 32 or w < 32:
            scale = max(32/h, 32/w, 2.0)
            new_h, new_w = int(h * scale), int(w * scale)
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 2. Enhance contrast
        processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=10)
        
        # 3. Noise reduction
        processed = cv2.medianBlur(processed, 3)
        
        # 4. Adaptive thresholding for better text contrast
        processed = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return processed
    
    def extract_text_with_preprocessing_variants(self, image: np.ndarray,
                                               engine: Optional[str] = None) -> List[OCRResult]:
        """
        Extract text with multiple preprocessing variants and return all results.
        
        Args:
            image: Input image
            engine: Specific engine to use
            
        Returns:
            List of OCR results from different preprocessing variants
        """
        preprocessing_variants = [
            ("original", image),
            ("grayscale", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image),
            ("enhanced", self._preprocess_image(image)),
            ("inverted", 255 - self._preprocess_image(image))
        ]
        
        results = []
        
        for variant_name, variant_image in preprocessing_variants:
            result = self.extract_text(variant_image, engine=engine, fallback=False)
            result.engine_used += f"_{variant_name}"
            results.append(result)
        
        return results
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engine names."""
        return list(self.engines.keys())
    
    def get_engine_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available engines."""
        info = {}
        
        for engine_name, engine in self.engines.items():
            info[engine_name] = {
                "available": engine.is_available(),
                "class": engine.__class__.__name__
            }
        
        return info


def main():
    """Example usage of OCR engine."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize OCR system
    ocr = MultiEngineOCR()
    
    print("Multi-Engine OCR System")
    print(f"Available engines: {ocr.get_available_engines()}")
    print("\nEngine information:")
    for name, info in ocr.get_engine_info().items():
        print(f"  {name}: {info}")
    
    print("\nUsage examples:")
    print("1. Extract text with best engine:")
    print("   result = ocr.extract_text(image)")
    print("2. Extract with specific engine:")
    print("   result = ocr.extract_text(image, engine='easyocr')")
    print("3. Extract with preprocessing variants:")
    print("   results = ocr.extract_text_with_preprocessing_variants(image)")


if __name__ == "__main__":
    main()