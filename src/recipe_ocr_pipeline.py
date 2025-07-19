#!/usr/bin/env python3
"""
Complete OCR pipeline for recipe ingredient extraction.
Integrates YOLOv8 text detection, multi-engine OCR, and structured parsing.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import argparse

# Import pipeline components
from text_detection import TextDetector, TextRegion
from ocr_engine import MultiEngineOCR, OCRResult
from ingredient_parser import IngredientParser, ParsedIngredient
from text_cleaner import TextCleaner, CleaningResult


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    source_image_path: str
    processing_time: float
    text_regions_detected: int
    ingredients_extracted: int
    pipeline_config: Dict[str, Any]
    text_regions: List[Dict[str, Any]]
    ingredients: List[Dict[str, Any]]
    pipeline_metadata: Dict[str, Any]
    confidence_summary: Dict[str, float]
    error_log: List[str]


class RecipeOCRPipeline:
    """Complete OCR pipeline for recipe ingredient extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OCR pipeline.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize pipeline components
        self._initialize_components()
        
        self.logger.info("Recipe OCR Pipeline initialized successfully")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default pipeline configuration."""
        return {
            # Text detection settings
            "text_detection": {
                "model_path": "yolov8n.pt",
                "confidence_threshold": 0.25,
                "device": "auto",
                "target_classes": [0, 1],  # ingredient_line, ingredient_block
                "merge_overlapping": True,
                "iou_threshold": 0.5,
                "min_region_area": 100,
                "region_padding": 5
            },
            
            # OCR settings
            "ocr": {
                "engines": ["easyocr", "tesseract", "paddleocr"],
                "primary_engine": "easyocr",
                "enable_fallback": True,
                "min_confidence": 0.3,
                "preprocessing_variants": False,
                "tesseract_config": "--oem 3 --psm 6"
            },
            
            # Text cleaning settings
            "text_cleaning": {
                "enabled": True,
                "aggressive_mode": False,
                "min_improvement_threshold": 0.1
            },
            
            # Ingredient parsing settings
            "ingredient_parsing": {
                "min_confidence": 0.3,
                "normalize_units": True,
                "extract_preparations": True,
                "validate_ingredients": True
            },
            
            # Output settings
            "output": {
                "save_annotated_image": True,
                "save_region_images": False,
                "include_debug_info": False,
                "output_format": "json",
                "confidence_threshold": 0.0
            },
            
            # Performance settings
            "performance": {
                "max_image_size": 2048,
                "enable_gpu": True,
                "parallel_processing": False
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get("logging", {}).get("level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_components(self):
        """Initialize pipeline components."""
        try:
            # Initialize text detector
            self.text_detector = TextDetector(
                model_path=self.config["text_detection"]["model_path"],
                confidence_threshold=self.config["text_detection"]["confidence_threshold"],
                device=self.config["text_detection"]["device"]
            )
            
            # Initialize OCR engine
            self.ocr_engine = MultiEngineOCR(
                engines=self.config["ocr"]["engines"],
                primary_engine=self.config["ocr"]["primary_engine"]
            )
            
            # Initialize text cleaner
            self.text_cleaner = TextCleaner() if self.config["text_cleaning"]["enabled"] else None
            
            # Initialize ingredient parser
            self.ingredient_parser = IngredientParser()
            
            self.logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def process_image(self, image_path: str, output_dir: Optional[str] = None) -> PipelineResult:
        """
        Process a recipe image through the complete OCR pipeline.
        
        Args:
            image_path: Path to input image
            output_dir: Optional output directory for results
            
        Returns:
            Complete pipeline result
        """
        start_time = time.time()
        image_path = str(Path(image_path).resolve())
        
        self.logger.info(f"Processing image: {image_path}")
        
        # Initialize result tracking
        error_log = []
        text_regions_data = []
        ingredients_data = []
        
        try:
            # Load and validate image
            image = self._load_image(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Step 1: Detect text regions
            self.logger.info("Step 1: Detecting text regions...")
            text_regions = self._detect_text_regions(image, error_log)
            
            # Step 2: Extract and process each text region
            self.logger.info(f"Step 2: Processing {len(text_regions)} text regions...")
            for i, region in enumerate(text_regions):
                try:
                    region_result = self._process_text_region(image, region, i, output_dir)
                    text_regions_data.append(region_result)
                    
                    # Add to ingredients if successfully parsed
                    if region_result.get("parsed_ingredient", {}).get("is_valid", False):
                        ingredients_data.append(region_result["parsed_ingredient"])
                        
                except Exception as e:
                    error_msg = f"Failed to process text region {i}: {e}"
                    self.logger.warning(error_msg)
                    error_log.append(error_msg)
                    continue
            
            # Step 3: Post-processing and validation
            self.logger.info("Step 3: Post-processing results...")
            ingredients_data = self._post_process_ingredients(ingredients_data, error_log)
            
            # Step 4: Generate outputs
            if output_dir:
                self._save_outputs(image, image_path, text_regions, text_regions_data, 
                                 ingredients_data, output_dir)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate confidence summary
            confidence_summary = self._calculate_confidence_summary(text_regions_data, ingredients_data)
            
            # Create pipeline result
            result = PipelineResult(
                source_image_path=image_path,
                processing_time=processing_time,
                text_regions_detected=len(text_regions),
                ingredients_extracted=len(ingredients_data),
                pipeline_config=self.config,
                text_regions=text_regions_data,
                ingredients=ingredients_data,
                pipeline_metadata=self._get_pipeline_metadata(),
                confidence_summary=confidence_summary,
                error_log=error_log
            )
            
            self.logger.info(f"Pipeline completed successfully in {processing_time:.2f}s")
            self.logger.info(f"Detected {len(text_regions)} regions, extracted {len(ingredients_data)} ingredients")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            error_log.append(f"Pipeline error: {e}")
            
            # Return partial result
            return PipelineResult(
                source_image_path=image_path,
                processing_time=time.time() - start_time,
                text_regions_detected=0,
                ingredients_extracted=0,
                pipeline_config=self.config,
                text_regions=[],
                ingredients=[],
                pipeline_metadata=self._get_pipeline_metadata(),
                confidence_summary={},
                error_log=error_log
            )
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Resize if too large
            max_size = self.config["performance"]["max_image_size"]
            h, w = image.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                self.logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h}")
            
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _detect_text_regions(self, image: np.ndarray, error_log: List[str]) -> List[TextRegion]:
        """Detect text regions using YOLOv8."""
        try:
            # Detect text regions
            regions = self.text_detector.detect_text_regions(
                image, 
                target_classes=self.config["text_detection"]["target_classes"]
            )
            
            # Filter by size
            regions = self.text_detector.filter_regions_by_size(
                regions,
                min_area=self.config["text_detection"]["min_region_area"]
            )
            
            # Merge overlapping regions if enabled
            if self.config["text_detection"]["merge_overlapping"]:
                regions = self.text_detector.merge_overlapping_regions(
                    regions,
                    iou_threshold=self.config["text_detection"]["iou_threshold"]
                )
            
            self.logger.info(f"Detected {len(regions)} text regions")
            return regions
            
        except Exception as e:
            error_msg = f"Text detection failed: {e}"
            self.logger.error(error_msg)
            error_log.append(error_msg)
            return []
    
    def _process_text_region(self, image: np.ndarray, region: TextRegion, 
                           region_id: int, output_dir: Optional[str]) -> Dict[str, Any]:
        """Process a single text region through OCR and parsing."""
        
        # Extract region image
        region_image = self.text_detector.extract_text_region_image(
            image, region, padding=self.config["text_detection"]["region_padding"]
        )
        
        # Save region image if requested
        if output_dir and self.config["output"]["save_region_images"]:
            region_path = Path(output_dir) / f"region_{region_id:03d}.jpg"
            cv2.imwrite(str(region_path), region_image)
        
        # Apply OCR
        if self.config["ocr"]["preprocessing_variants"]:
            ocr_results = self.ocr_engine.extract_text_with_preprocessing_variants(
                region_image, engine=self.config["ocr"]["primary_engine"]
            )
            # Select best result
            ocr_result = max(ocr_results, key=lambda x: x.confidence)
        else:
            ocr_result = self.ocr_engine.extract_text(
                region_image,
                engine=self.config["ocr"]["primary_engine"],
                fallback=self.config["ocr"]["enable_fallback"],
                min_confidence=self.config["ocr"]["min_confidence"]
            )
        
        # Apply text cleaning if enabled
        cleaning_result = None
        cleaned_text = ocr_result.text
        
        if self.text_cleaner and ocr_result.text:
            cleaning_result = self.text_cleaner.clean_text(
                ocr_result.text,
                aggressive=self.config["text_cleaning"]["aggressive_mode"]
            )
            
            # Use cleaned text if improvement is significant
            if cleaning_result.confidence_improvement > self.config["text_cleaning"]["min_improvement_threshold"]:
                cleaned_text = cleaning_result.cleaned_text
        
        # Parse ingredient
        parsed_ingredient = None
        if cleaned_text and cleaned_text.strip():
            parsed_ingredient = self.ingredient_parser.parse_ingredient_line(cleaned_text)
        
        # Create region result
        region_result = {
            "region_id": region_id,
            "bbox": region.bbox,
            "class_id": region.class_id,
            "class_name": region.class_name,
            "detection_confidence": region.confidence,
            "area": region.area,
            "ocr_result": {
                "raw_text": ocr_result.text,
                "confidence": ocr_result.confidence,
                "engine_used": ocr_result.engine_used,
                "processing_time": ocr_result.processing_time
            },
            "text_cleaning": {
                "applied": cleaning_result is not None,
                "cleaned_text": cleaned_text,
                "corrections_made": cleaning_result.corrections_made if cleaning_result else [],
                "confidence_improvement": cleaning_result.confidence_improvement if cleaning_result else 0.0
            } if cleaning_result else None,
            "parsed_ingredient": parsed_ingredient.to_dict() if parsed_ingredient else None
        }
        
        return region_result
    
    def _post_process_ingredients(self, ingredients_data: List[Dict[str, Any]], 
                                error_log: List[str]) -> List[Dict[str, Any]]:
        """Post-process and validate ingredient data."""
        if not ingredients_data:
            return ingredients_data
        
        # Filter by confidence threshold
        confidence_threshold = self.config["output"]["confidence_threshold"]
        filtered_ingredients = [
            ing for ing in ingredients_data 
            if ing.get("confidence", 0) >= confidence_threshold
        ]
        
        # Remove duplicates (similar ingredient names)
        unique_ingredients = self._remove_duplicate_ingredients(filtered_ingredients)
        
        # Validate and normalize if enabled
        if self.config["ingredient_parsing"]["validate_ingredients"]:
            validated_ingredients = self._validate_ingredients(unique_ingredients, error_log)
        else:
            validated_ingredients = unique_ingredients
        
        self.logger.info(f"Post-processing: {len(ingredients_data)} -> {len(validated_ingredients)} ingredients")
        
        return validated_ingredients
    
    def _remove_duplicate_ingredients(self, ingredients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate ingredients based on similarity."""
        if len(ingredients) <= 1:
            return ingredients
        
        unique_ingredients = []
        seen_names = set()
        
        # Sort by confidence (highest first)
        sorted_ingredients = sorted(ingredients, key=lambda x: x.get("confidence", 0), reverse=True)
        
        for ingredient in sorted_ingredients:
            name = ingredient.get("ingredient_name", "").lower().strip()
            if not name:
                continue
            
            # Check for similar names
            is_duplicate = False
            for seen_name in seen_names:
                if self._are_similar_ingredients(name, seen_name):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_ingredients.append(ingredient)
                seen_names.add(name)
        
        return unique_ingredients
    
    def _are_similar_ingredients(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """Check if two ingredient names are similar."""
        import difflib
        similarity = difflib.SequenceMatcher(None, name1, name2).ratio()
        return similarity >= threshold
    
    def _validate_ingredients(self, ingredients: List[Dict[str, Any]], 
                            error_log: List[str]) -> List[Dict[str, Any]]:
        """Validate ingredient data."""
        validated = []
        
        for ingredient in ingredients:
            try:
                # Check for required fields
                if not ingredient.get("ingredient_name"):
                    continue
                
                # Validate quantity if present
                quantity = ingredient.get("quantity")
                if quantity:
                    try:
                        qty_float = float(quantity)
                        if qty_float <= 0 or qty_float > 1000:  # Reasonable bounds
                            error_log.append(f"Unusual quantity: {quantity}")
                    except ValueError:
                        # Non-numeric quantities are okay (e.g., "to taste")
                        pass
                
                # Validate unit if present
                unit = ingredient.get("unit")
                if unit and len(unit) > 20:  # Suspiciously long unit
                    error_log.append(f"Unusual unit: {unit}")
                
                validated.append(ingredient)
                
            except Exception as e:
                error_log.append(f"Ingredient validation error: {e}")
                continue
        
        return validated
    
    def _calculate_confidence_summary(self, text_regions: List[Dict[str, Any]], 
                                    ingredients: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence summary statistics."""
        if not text_regions:
            return {}
        
        # Detection confidences
        detection_confidences = [r.get("detection_confidence", 0) for r in text_regions]
        
        # OCR confidences
        ocr_confidences = [r.get("ocr_result", {}).get("confidence", 0) for r in text_regions]
        
        # Parsing confidences
        parsing_confidences = [
            r.get("parsed_ingredient", {}).get("confidence", 0) 
            for r in text_regions 
            if r.get("parsed_ingredient")
        ]
        
        # Overall confidences for valid ingredients
        ingredient_confidences = [ing.get("confidence", 0) for ing in ingredients]
        
        summary = {
            "avg_detection_confidence": sum(detection_confidences) / len(detection_confidences) if detection_confidences else 0,
            "avg_ocr_confidence": sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0,
            "avg_parsing_confidence": sum(parsing_confidences) / len(parsing_confidences) if parsing_confidences else 0,
            "avg_ingredient_confidence": sum(ingredient_confidences) / len(ingredient_confidences) if ingredient_confidences else 0,
            "high_confidence_regions": sum(1 for c in detection_confidences if c > 0.7),
            "medium_confidence_regions": sum(1 for c in detection_confidences if 0.3 <= c <= 0.7),
            "low_confidence_regions": sum(1 for c in detection_confidences if c < 0.3)
        }
        
        return summary
    
    def _get_pipeline_metadata(self) -> Dict[str, Any]:
        """Get pipeline metadata."""
        return {
            "pipeline_version": "1.0.0",
            "text_detector_model": self.config["text_detection"]["model_path"],
            "ocr_engines": self.ocr_engine.get_available_engines(),
            "text_cleaning_enabled": self.config["text_cleaning"]["enabled"],
            "processing_timestamp": time.time()
        }
    
    def _save_outputs(self, image: np.ndarray, image_path: str, text_regions: List[TextRegion],
                     text_regions_data: List[Dict[str, Any]], ingredients_data: List[Dict[str, Any]],
                     output_dir: str):
        """Save pipeline outputs."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(image_path).stem
        
        # Save annotated image
        if self.config["output"]["save_annotated_image"]:
            annotated_image = self.text_detector.visualize_detections(
                image, text_regions
            )
            annotated_path = output_path / f"{image_name}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated_image)
            self.logger.info(f"Saved annotated image: {annotated_path}")
        
        # Save structured results
        results = {
            "source_image": image_path,
            "text_regions": text_regions_data,
            "ingredients": ingredients_data,
            "metadata": self._get_pipeline_metadata()
        }
        
        if self.config["output"]["include_debug_info"]:
            results["pipeline_config"] = self.config
        
        # Save as JSON
        results_path = output_path / f"{image_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Saved results: {results_path}")
    
    def process_batch(self, image_paths: List[str], output_dir: str) -> List[PipelineResult]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths
            output_dir: Output directory for all results
            
        Returns:
            List of pipeline results
        """
        results = []
        
        self.logger.info(f"Processing batch of {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            # Create individual output directory
            image_output_dir = Path(output_dir) / f"image_{i+1:03d}_{Path(image_path).stem}"
            
            try:
                result = self.process_image(image_path, str(image_output_dir))
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                continue
        
        # Save batch summary
        self._save_batch_summary(results, output_dir)
        
        return results
    
    def _save_batch_summary(self, results: List[PipelineResult], output_dir: str):
        """Save batch processing summary."""
        summary = {
            "total_images": len(results),
            "successful_extractions": sum(1 for r in results if r.ingredients_extracted > 0),
            "total_ingredients_extracted": sum(r.ingredients_extracted for r in results),
            "total_processing_time": sum(r.processing_time for r in results),
            "average_processing_time": sum(r.processing_time for r in results) / len(results) if results else 0,
            "average_ingredients_per_image": sum(r.ingredients_extracted for r in results) / len(results) if results else 0,
            "error_summary": {}
        }
        
        # Count error types
        for result in results:
            for error in result.error_log:
                error_type = error.split(":")[0] if ":" in error else "Unknown"
                summary["error_summary"][error_type] = summary["error_summary"].get(error_type, 0) + 1
        
        summary_path = Path(output_dir) / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Saved batch summary: {summary_path}")


def main():
    """Command-line interface for the OCR pipeline."""
    parser = argparse.ArgumentParser(description='Recipe OCR Pipeline')
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('--output-dir', '-o', default='output', help='Output directory')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--confidence', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--ocr-engine', default='easyocr', choices=['easyocr', 'tesseract', 'paddleocr'],
                        help='Primary OCR engine')
    parser.add_argument('--aggressive-cleaning', action='store_true',
                        help='Enable aggressive text cleaning')
    parser.add_argument('--save-regions', action='store_true',
                        help='Save individual text region images')
    parser.add_argument('--debug', action='store_true',
                        help='Include debug information in output')
    parser.add_argument('--batch', action='store_true',
                        help='Process directory of images')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override config with command line arguments
    config.setdefault("text_detection", {})["model_path"] = args.model
    config.setdefault("text_detection", {})["confidence_threshold"] = args.confidence
    config.setdefault("ocr", {})["primary_engine"] = args.ocr_engine
    config.setdefault("text_cleaning", {})["aggressive_mode"] = args.aggressive_cleaning
    config.setdefault("output", {})["save_region_images"] = args.save_regions
    config.setdefault("output", {})["include_debug_info"] = args.debug
    
    # Initialize pipeline
    pipeline = RecipeOCRPipeline(config)
    
    # Process input
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # Batch processing
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            str(p) for p in input_path.rglob('*') 
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            print(f"No images found in {input_path}")
            return
        
        print(f"Found {len(image_paths)} images for batch processing")
        results = pipeline.process_batch(image_paths, args.output_dir)
        
        print(f"\nBatch processing completed:")
        print(f"  Images processed: {len(results)}")
        print(f"  Total ingredients extracted: {sum(r.ingredients_extracted for r in results)}")
        print(f"  Average processing time: {sum(r.processing_time for r in results) / len(results):.2f}s")
        
    else:
        # Single image processing
        if not input_path.exists():
            print(f"Image not found: {input_path}")
            return
        
        result = pipeline.process_image(str(input_path), args.output_dir)
        
        print(f"\nProcessing completed:")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Text regions detected: {result.text_regions_detected}")
        print(f"  Ingredients extracted: {result.ingredients_extracted}")
        
        if result.ingredients_extracted > 0:
            print(f"\nExtracted ingredients:")
            for i, ingredient in enumerate(result.ingredients, 1):
                quantity = ingredient.get('quantity', '')
                unit = ingredient.get('unit', '')
                name = ingredient.get('ingredient_name', '')
                confidence = ingredient.get('confidence', 0)
                
                print(f"  {i}. {quantity} {unit} {name} (confidence: {confidence:.2f})")
        
        if result.error_log:
            print(f"\nWarnings/Errors:")
            for error in result.error_log:
                print(f"  - {error}")


if __name__ == "__main__":
    main()