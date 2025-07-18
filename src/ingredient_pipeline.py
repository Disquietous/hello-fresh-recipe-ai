#!/usr/bin/env python3
"""
HelloFresh Recipe AI - Ingredient Extraction Pipeline
Complete pipeline for extracting structured ingredient data from recipe images.

Pipeline stages:
1. YOLOv8 text region detection
2. OCR text extraction
3. Text preprocessing
4. Ingredient parsing
5. Structured data output
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import pytesseract
from PIL import Image

from utils.text_utils import IngredientParser, TextPreprocessor, RecipeDataValidator


class IngredientExtractionPipeline:
    """Complete pipeline for extracting structured ingredient data from recipe images."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ingredient extraction pipeline.
        
        Args:
            config (Dict, optional): Pipeline configuration
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.text_detector = None
        self.ocr_engine = None
        self.text_preprocessor = TextPreprocessor()
        self.ingredient_parser = IngredientParser()
        self.validator = RecipeDataValidator()
        
        # Load models
        self._initialize_models()
    
    def _get_default_config(self) -> Dict:
        """Get default pipeline configuration."""
        return {
            'text_detection': {
                'model_path': 'yolov8n.pt',
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'ocr': {
                'engine': 'easyocr',  # 'easyocr', 'paddleocr', 'tesseract'
                'languages': ['en'],
                'gpu': True
            },
            'preprocessing': {
                'enhance_contrast': True,
                'remove_noise': True,
                'correct_skew': True,
                'resize_min_height': 32
            },
            'parsing': {
                'min_confidence': 0.5,
                'fuzzy_match_threshold': 0.8,
                'validate_units': True
            },
            'output': {
                'save_annotated_image': True,
                'save_cropped_regions': False,
                'output_format': 'json'  # 'json', 'csv', 'xml'
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline."""
        logger = logging.getLogger('IngredientPipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_models(self):
        """Initialize YOLOv8 and OCR models."""
        try:
            # Initialize text detection model
            model_path = self.config['text_detection']['model_path']
            self.text_detector = YOLO(model_path)
            self.logger.info(f"Loaded text detection model: {model_path}")
            
            # Initialize OCR engine
            ocr_engine = self.config['ocr']['engine']
            if ocr_engine == 'easyocr':
                self.ocr_engine = easyocr.Reader(
                    self.config['ocr']['languages'],
                    gpu=self.config['ocr']['gpu']
                )
            elif ocr_engine == 'paddleocr':
                from paddleocr import PaddleOCR
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=self.config['ocr']['gpu']
                )
            elif ocr_engine == 'tesseract':
                # Tesseract doesn't need initialization
                self.ocr_engine = 'tesseract'
            
            self.logger.info(f"Initialized OCR engine: {ocr_engine}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Stage 1: Detect text regions using YOLOv8.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Dict]: Detected text regions with bounding boxes
        """
        self.logger.info("Stage 1: Detecting text regions...")
        
        results = self.text_detector(
            image,
            conf=self.config['text_detection']['confidence_threshold'],
            iou=self.config['text_detection']['iou_threshold']
        )
        
        text_regions = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else None
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                region = {
                    'region_id': i,
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class_id': int(classes[i]) if classes is not None else 0,
                    'class_name': results[0].names.get(int(classes[i]), 'text') if classes is not None else 'text'
                }
                text_regions.append(region)
        
        self.logger.info(f"Found {len(text_regions)} text regions")
        return text_regions
    
    def extract_text_from_regions(self, image: np.ndarray, regions: List[Dict]) -> List[Dict]:
        """
        Stage 2: Extract text from detected regions using OCR.
        
        Args:
            image (np.ndarray): Input image
            regions (List[Dict]): Text regions from detection
            
        Returns:
            List[Dict]: Regions with extracted text
        """
        self.logger.info("Stage 2: Extracting text from regions...")
        
        processed_regions = []
        
        for region in regions:
            bbox = region['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extract region of interest
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # Stage 3: Preprocess the text region
            preprocessed_roi = self._preprocess_text_region(roi)
            
            # Extract text using OCR
            extracted_text = self._extract_text_with_ocr(preprocessed_roi)
            
            if extracted_text.strip():
                region['extracted_text'] = extracted_text
                region['text_confidence'] = self._calculate_text_confidence(extracted_text)
                processed_regions.append(region)
        
        self.logger.info(f"Successfully extracted text from {len(processed_regions)} regions")
        return processed_regions
    
    def _preprocess_text_region(self, roi: np.ndarray) -> np.ndarray:
        """
        Stage 3: Preprocess text region for better OCR.
        
        Args:
            roi (np.ndarray): Region of interest
            
        Returns:
            np.ndarray: Preprocessed image
        """
        if self.config['preprocessing']['enhance_contrast']:
            roi = self.text_preprocessor.enhance_contrast(roi)
        
        if self.config['preprocessing']['remove_noise']:
            roi = self.text_preprocessor.remove_noise(roi)
        
        if self.config['preprocessing']['correct_skew']:
            roi = self.text_preprocessor.correct_skew(roi)
        
        # Ensure minimum height for OCR
        min_height = self.config['preprocessing']['resize_min_height']
        if roi.shape[0] < min_height:
            scale_factor = min_height / roi.shape[0]
            new_width = int(roi.shape[1] * scale_factor)
            roi = cv2.resize(roi, (new_width, min_height), interpolation=cv2.INTER_CUBIC)
        
        return roi
    
    def _extract_text_with_ocr(self, roi: np.ndarray) -> str:
        """Extract text using the configured OCR engine."""
        try:
            if self.config['ocr']['engine'] == 'easyocr':
                results = self.ocr_engine.readtext(roi)
                return " ".join([result[1] for result in results])
            
            elif self.config['ocr']['engine'] == 'paddleocr':
                results = self.ocr_engine.ocr(roi, cls=True)
                if results and results[0]:
                    return " ".join([line[1][0] for line in results[0]])
                return ""
            
            elif self.config['ocr']['engine'] == 'tesseract':
                return pytesseract.image_to_string(roi, config='--psm 7')
            
        except Exception as e:
            self.logger.warning(f"OCR extraction failed: {e}")
            return ""
        
        return ""
    
    def _calculate_text_confidence(self, text: str) -> float:
        """Calculate confidence score for extracted text."""
        if not text.strip():
            return 0.0
        
        # Simple heuristics for text quality
        score = 1.0
        
        # Penalize very short text
        if len(text.strip()) < 3:
            score *= 0.5
        
        # Penalize text with many special characters
        special_chars = sum(1 for c in text if not c.isalnum() and c != ' ')
        if special_chars > len(text) * 0.3:
            score *= 0.7
        
        # Bonus for common ingredient words
        common_words = ['cup', 'tbsp', 'tsp', 'oz', 'lb', 'gram', 'kg', 'ml', 'liter']
        if any(word in text.lower() for word in common_words):
            score = min(1.0, score * 1.2)
        
        return score
    
    def parse_ingredients(self, regions: List[Dict]) -> List[Dict]:
        """
        Stage 4: Parse ingredient information from extracted text.
        
        Args:
            regions (List[Dict]): Regions with extracted text
            
        Returns:
            List[Dict]: Parsed ingredient data
        """
        self.logger.info("Stage 4: Parsing ingredient information...")
        
        parsed_ingredients = []
        
        for region in regions:
            text = region.get('extracted_text', '')
            if not text.strip():
                continue
            
            # Parse ingredient using the parser
            ingredient_data = self.ingredient_parser.parse_ingredient_line(text)
            
            # Add region metadata
            ingredient_data['region_id'] = region['region_id']
            ingredient_data['bbox'] = region['bbox']
            ingredient_data['detection_confidence'] = region['confidence']
            ingredient_data['text_confidence'] = region.get('text_confidence', 0.0)
            ingredient_data['class_name'] = region.get('class_name', 'text')
            
            # Calculate overall confidence
            ingredient_data['overall_confidence'] = (
                ingredient_data['confidence_score'] * 0.4 +
                ingredient_data['detection_confidence'] * 0.3 +
                ingredient_data['text_confidence'] * 0.3
            )
            
            # Filter by minimum confidence
            if ingredient_data['overall_confidence'] >= self.config['parsing']['min_confidence']:
                parsed_ingredients.append(ingredient_data)
        
        self.logger.info(f"Parsed {len(parsed_ingredients)} ingredients")
        return parsed_ingredients
    
    def create_structured_output(self, ingredients: List[Dict], image_path: str) -> Dict:
        """
        Stage 5: Create structured output with validation.
        
        Args:
            ingredients (List[Dict]): Parsed ingredient data
            image_path (str): Source image path
            
        Returns:
            Dict: Structured ingredient data
        """
        self.logger.info("Stage 5: Creating structured output...")
        
        # Validate ingredients
        validation_results = self.validator.validate_recipe(ingredients)
        
        # Create structured output
        structured_data = {
            'source_image': image_path,
            'pipeline_version': '1.0',
            'processing_timestamp': None,  # Will be set when saving
            'configuration': self.config,
            'detection_summary': {
                'total_regions_detected': len(ingredients),
                'high_confidence_ingredients': len([ing for ing in ingredients if ing['overall_confidence'] >= 0.8]),
                'medium_confidence_ingredients': len([ing for ing in ingredients if 0.6 <= ing['overall_confidence'] < 0.8]),
                'low_confidence_ingredients': len([ing for ing in ingredients if ing['overall_confidence'] < 0.6])
            },
            'validation_results': validation_results,
            'ingredients': []
        }
        
        # Format ingredients for output
        for ingredient in ingredients:
            formatted_ingredient = {
                'ingredient_name': ingredient.get('ingredient_name', ''),
                'quantity': ingredient.get('amount', ''),
                'unit': ingredient.get('unit', ''),
                'unit_category': ingredient.get('unit_category', ''),
                'confidence_scores': {
                    'overall': round(ingredient['overall_confidence'], 3),
                    'ingredient_recognition': round(ingredient['confidence_score'], 3),
                    'text_detection': round(ingredient['detection_confidence'], 3),
                    'ocr_quality': round(ingredient['text_confidence'], 3)
                },
                'raw_text': ingredient.get('original_text', ''),
                'bounding_box': {
                    'x1': int(ingredient['bbox'][0]),
                    'y1': int(ingredient['bbox'][1]),
                    'x2': int(ingredient['bbox'][2]),
                    'y2': int(ingredient['bbox'][3])
                },
                'region_id': ingredient['region_id'],
                'detection_class': ingredient.get('class_name', 'text')
            }
            structured_data['ingredients'].append(formatted_ingredient)
        
        return structured_data
    
    def process_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path (str): Path to input image
            output_dir (str, optional): Directory to save outputs
            
        Returns:
            Dict: Structured ingredient data
        """
        self.logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        try:
            # Run pipeline stages
            text_regions = self.detect_text_regions(image)
            regions_with_text = self.extract_text_from_regions(image, text_regions)
            parsed_ingredients = self.parse_ingredients(regions_with_text)
            structured_output = self.create_structured_output(parsed_ingredients, image_path)
            
            # Save outputs if directory specified
            if output_dir:
                self._save_outputs(image, structured_output, output_dir, Path(image_path).stem)
            
            return structured_output
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            raise
    
    def _save_outputs(self, image: np.ndarray, data: Dict, output_dir: str, base_name: str):
        """Save pipeline outputs to files."""
        import datetime
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        data['processing_timestamp'] = datetime.datetime.now().isoformat()
        
        # Save JSON data
        json_path = output_path / f"{base_name}_ingredients.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Saved ingredient data: {json_path}")
        
        # Save annotated image if configured
        if self.config['output']['save_annotated_image']:
            annotated_image = self._create_annotated_image(image, data['ingredients'])
            img_path = output_path / f"{base_name}_annotated.jpg"
            cv2.imwrite(str(img_path), annotated_image)
            self.logger.info(f"Saved annotated image: {img_path}")
        
        # Save cropped regions if configured
        if self.config['output']['save_cropped_regions']:
            crops_dir = output_path / f"{base_name}_crops"
            crops_dir.mkdir(exist_ok=True)
            self._save_cropped_regions(image, data['ingredients'], crops_dir)
    
    def _create_annotated_image(self, image: np.ndarray, ingredients: List[Dict]) -> np.ndarray:
        """Create annotated image with bounding boxes and labels."""
        annotated = image.copy()
        
        for ingredient in ingredients:
            bbox = ingredient['bounding_box']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # Choose color based on confidence
            confidence = ingredient['confidence_scores']['overall']
            if confidence >= 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence >= 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            name = ingredient['ingredient_name'] or 'unknown'
            quantity = ingredient['quantity'] or ''
            unit = ingredient['unit'] or ''
            label = f"{quantity} {unit} {name}".strip()
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1-25), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated
    
    def _save_cropped_regions(self, image: np.ndarray, ingredients: List[Dict], crops_dir: Path):
        """Save cropped regions for inspection."""
        for ingredient in ingredients:
            bbox = ingredient['bounding_box']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                region_id = ingredient['region_id']
                crop_path = crops_dir / f"region_{region_id:03d}.jpg"
                cv2.imwrite(str(crop_path), crop)


def main():
    """Main function for running the pipeline from command line."""
    parser = argparse.ArgumentParser(description='HelloFresh Recipe AI - Ingredient Extraction Pipeline')
    parser.add_argument('image_path', help='Path to recipe image')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--config', help='Path to custom configuration file')
    parser.add_argument('--text-model', default='yolov8n.pt', help='YOLOv8 model for text detection')
    parser.add_argument('--ocr-engine', default='easyocr', choices=['easyocr', 'paddleocr', 'tesseract'],
                       help='OCR engine to use')
    parser.add_argument('--confidence', type=float, default=0.25, help='Text detection confidence threshold')
    parser.add_argument('--min-ingredient-confidence', type=float, default=0.5, 
                       help='Minimum confidence for ingredient parsing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger('IngredientPipeline').setLevel(logging.DEBUG)
    
    # Load custom config if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Create config from command line arguments
        config = {
            'text_detection': {
                'model_path': args.text_model,
                'confidence_threshold': args.confidence
            },
            'ocr': {
                'engine': args.ocr_engine
            },
            'parsing': {
                'min_confidence': args.min_ingredient_confidence
            }
        }
    
    # Initialize and run pipeline
    try:
        pipeline = IngredientExtractionPipeline(config)
        results = pipeline.process_image(args.image_path, args.output_dir)
        
        # Print summary
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Processing Summary:")
        print(f"   - Total ingredients detected: {results['detection_summary']['total_regions_detected']}")
        print(f"   - High confidence: {results['detection_summary']['high_confidence_ingredients']}")
        print(f"   - Medium confidence: {results['detection_summary']['medium_confidence_ingredients']}")
        print(f"   - Low confidence: {results['detection_summary']['low_confidence_ingredients']}")
        print(f"   - Validation score: {results['validation_results']['validation_score']:.2f}")
        print(f"📁 Results saved to: {args.output_dir}")
        
        # Show ingredients
        if results['ingredients']:
            print(f"\n🥘 Detected Ingredients:")
            for ing in results['ingredients']:
                conf = ing['confidence_scores']['overall']
                print(f"   - {ing['quantity']} {ing['unit']} {ing['ingredient_name']} (confidence: {conf:.2f})")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())