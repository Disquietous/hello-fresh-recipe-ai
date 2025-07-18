#!/usr/bin/env python3
"""
HelloFresh Recipe AI - Text Detection and Recognition Script
Detects and recognizes ingredient names, amounts, and units from recipe images.
"""

import argparse
import os
import sys
import re
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import pytesseract
from PIL import Image
import json


class IngredientTextDetector:
    """Text detection and recognition class for recipe ingredients."""
    
    def __init__(self, text_model_path="yolov8n.pt", ocr_engine="easyocr"):
        """
        Initialize the text detector.
        
        Args:
            text_model_path (str): Path to YOLO model for text detection
            ocr_engine (str): OCR engine to use ('easyocr', 'tesseract', 'paddleocr')
        """
        self.text_model = YOLO(text_model_path)
        self.ocr_engine = ocr_engine
        
        # Initialize OCR engine
        if ocr_engine == "easyocr":
            self.ocr_reader = easyocr.Reader(['en'])
        elif ocr_engine == "paddleocr":
            from paddleocr import PaddleOCR
            self.ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')
        
        # Common units and measurements
        self.units = {
            'weight': ['g', 'kg', 'gram', 'grams', 'kilogram', 'kilograms', 'oz', 'ounce', 'ounces', 'lb', 'lbs', 'pound', 'pounds'],
            'volume': ['ml', 'l', 'liter', 'liters', 'milliliter', 'milliliters', 'cup', 'cups', 'tbsp', 'tablespoon', 'tablespoons', 'tsp', 'teaspoon', 'teaspoons', 'fl oz', 'pint', 'pints', 'quart', 'quarts', 'gallon', 'gallons'],
            'count': ['piece', 'pieces', 'item', 'items', 'clove', 'cloves', 'slice', 'slices', 'whole', 'half', 'quarter']
        }
        
        # Common ingredient keywords
        self.ingredient_keywords = [
            'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'shrimp', 'egg', 'eggs',
            'onion', 'onions', 'garlic', 'tomato', 'tomatoes', 'carrot', 'carrots', 'potato', 'potatoes',
            'rice', 'pasta', 'bread', 'flour', 'sugar', 'salt', 'pepper', 'oil', 'butter',
            'milk', 'cream', 'cheese', 'yogurt', 'apple', 'banana', 'orange', 'lemon',
            'broccoli', 'spinach', 'lettuce', 'cucumber', 'bell pepper', 'mushroom', 'mushrooms'
        ]
    
    def detect_text_regions(self, image_path, conf_threshold=0.25):
        """
        Detect text regions in image using YOLO.
        
        Args:
            image_path (str): Path to input image
            conf_threshold (float): Confidence threshold for detections
            
        Returns:
            list: List of text region bounding boxes
        """
        results = self.text_model(image_path, conf=conf_threshold)
        
        text_regions = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                text_regions.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'region_id': i
                })
        
        return text_regions
    
    def extract_text_from_region(self, image, bbox):
        """
        Extract text from a specific region using OCR.
        
        Args:
            image (numpy.ndarray): Input image
            bbox (list): Bounding box [x1, y1, x2, y2]
            
        Returns:
            str: Extracted text
        """
        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return ""
        
        # Preprocess ROI for better OCR
        roi = self.preprocess_text_region(roi)
        
        try:
            if self.ocr_engine == "easyocr":
                results = self.ocr_reader.readtext(roi)
                text = " ".join([result[1] for result in results])
            elif self.ocr_engine == "tesseract":
                text = pytesseract.image_to_string(roi, config='--psm 7')
            elif self.ocr_engine == "paddleocr":
                results = self.ocr_reader.ocr(roi, cls=True)
                if results and results[0]:
                    text = " ".join([line[1][0] for line in results[0]])
                else:
                    text = ""
            else:
                text = ""
                
            return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def preprocess_text_region(self, roi):
        """
        Preprocess text region for better OCR accuracy.
        
        Args:
            roi (numpy.ndarray): Region of interest
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.medianBlur(thresh, 3)
        
        # Resize for better OCR (if too small)
        h, w = denoised.shape
        if h < 30 or w < 100:
            scale_factor = max(30/h, 100/w, 1.0)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            denoised = cv2.resize(denoised, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return denoised
    
    def parse_ingredient_text(self, text):
        """
        Parse ingredient text to extract name, amount, and unit.
        
        Args:
            text (str): Raw ingredient text
            
        Returns:
            dict: Parsed ingredient information
        """
        text = text.lower().strip()
        if not text:
            return None
        
        # Pattern to match amounts and units
        amount_pattern = r'(\d+(?:\.\d+)?(?:/\d+)?)\s*([a-z]+)?'
        matches = re.findall(amount_pattern, text)
        
        amount = None
        unit = None
        ingredient_name = text
        
        # Extract amount and unit
        for match in matches:
            potential_amount, potential_unit = match
            
            # Check if potential_unit is a valid unit
            if potential_unit:
                all_units = []
                for unit_type in self.units.values():
                    all_units.extend(unit_type)
                
                if potential_unit in all_units:
                    amount = potential_amount
                    unit = potential_unit
                    # Remove amount and unit from ingredient name
                    ingredient_name = re.sub(f'{re.escape(potential_amount)}\\s*{re.escape(potential_unit)}', '', text).strip()
                    break
        
        # If no unit found, try to extract just the number
        if amount is None:
            number_match = re.search(r'(\d+(?:\.\d+)?(?:/\d+)?)', text)
            if number_match:
                amount = number_match.group(1)
                ingredient_name = re.sub(re.escape(amount), '', text).strip()
        
        # Clean up ingredient name
        ingredient_name = re.sub(r'[^\w\s]', '', ingredient_name).strip()
        
        # Validate if it's likely an ingredient
        is_ingredient = any(keyword in ingredient_name for keyword in self.ingredient_keywords)
        
        return {
            'raw_text': text,
            'ingredient_name': ingredient_name,
            'amount': amount,
            'unit': unit,
            'confidence_score': 1.0 if is_ingredient else 0.5
        }
    
    def detect_ingredients(self, image_path, output_path=None):
        """
        Detect and extract ingredient information from image.
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save annotated image
            
        Returns:
            dict: Detected ingredients information
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        # Detect text regions
        text_regions = self.detect_text_regions(image_path)
        
        ingredients = []
        annotated_image = image.copy()
        
        # Process each text region
        for region in text_regions:
            bbox = region['bbox']
            
            # Extract text from region
            text = self.extract_text_from_region(image, bbox)
            
            if text:
                # Parse ingredient information
                ingredient_info = self.parse_ingredient_text(text)
                
                if ingredient_info:
                    ingredient_info['bbox'] = bbox
                    ingredient_info['region_confidence'] = region['confidence']
                    ingredients.append(ingredient_info)
                    
                    # Draw bounding box and text on image
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{ingredient_info['ingredient_name']}"
                    if ingredient_info['amount']:
                        label = f"{ingredient_info['amount']} {ingredient_info['unit'] or ''} {label}".strip()
                    
                    cv2.putText(annotated_image, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save annotated image
        if output_path:
            cv2.imwrite(output_path, annotated_image)
        
        return {
            'image_path': image_path,
            'ingredients': ingredients,
            'total_ingredients': len(ingredients)
        }
    
    def process_recipe_list(self, image_path, output_path=None):
        """
        Process a complete recipe ingredient list.
        
        Args:
            image_path (str): Path to recipe image
            output_path (str): Path to save results JSON
            
        Returns:
            dict: Complete recipe information
        """
        results = self.detect_ingredients(image_path)
        
        if results:
            # Group ingredients by confidence
            high_confidence = [ing for ing in results['ingredients'] if ing['confidence_score'] >= 0.8]
            medium_confidence = [ing for ing in results['ingredients'] if 0.5 <= ing['confidence_score'] < 0.8]
            low_confidence = [ing for ing in results['ingredients'] if ing['confidence_score'] < 0.5]
            
            recipe_data = {
                'source_image': image_path,
                'total_detected': results['total_ingredients'],
                'high_confidence_ingredients': high_confidence,
                'medium_confidence_ingredients': medium_confidence,
                'low_confidence_ingredients': low_confidence,
                'processing_summary': {
                    'high_confidence_count': len(high_confidence),
                    'medium_confidence_count': len(medium_confidence),
                    'low_confidence_count': len(low_confidence)
                }
            }
            
            # Save results to JSON
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(recipe_data, f, indent=2)
            
            return recipe_data
        
        return None


def main():
    parser = argparse.ArgumentParser(description='HelloFresh Recipe AI - Ingredient Text Detection')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('--model', default='yolov8n.pt', help='Path to text detection model')
    parser.add_argument('--output-image', help='Output path for annotated image')
    parser.add_argument('--output-json', help='Output path for ingredient data JSON')
    parser.add_argument('--ocr-engine', default='easyocr', 
                       choices=['easyocr', 'tesseract', 'paddleocr'],
                       help='OCR engine to use')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = IngredientTextDetector(args.model, args.ocr_engine)
    
    # Check if input exists
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    print(f"Processing recipe image: {args.input}")
    print(f"Using OCR engine: {args.ocr_engine}")
    
    # Process recipe
    results = detector.process_recipe_list(
        args.input, 
        args.output_json
    )
    
    # Also create annotated image
    if args.output_image or not args.output_json:
        output_img = args.output_image or "results/annotated_recipe.jpg"
        detector.detect_ingredients(args.input, output_img)
        print(f"Annotated image saved to: {output_img}")
    
    if results:
        print(f"\nDetected {results['total_detected']} potential ingredients:")
        print(f"  High confidence: {results['processing_summary']['high_confidence_count']}")
        print(f"  Medium confidence: {results['processing_summary']['medium_confidence_count']}")
        print(f"  Low confidence: {results['processing_summary']['low_confidence_count']}")
        
        # Print high confidence ingredients
        if results['high_confidence_ingredients']:
            print("\nHigh Confidence Ingredients:")
            for ing in results['high_confidence_ingredients']:
                amount_str = f"{ing['amount']} {ing['unit'] or ''}".strip() if ing['amount'] else ""
                print(f"  - {amount_str} {ing['ingredient_name']}".strip())
    else:
        print("No ingredients detected.")


if __name__ == "__main__":
    main()