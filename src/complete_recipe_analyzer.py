#!/usr/bin/env python3
"""
Complete Recipe Analysis Engine
Integrates image preprocessing, text detection, OCR, ingredient parsing,
and nutritional analysis into a comprehensive recipe analysis system.
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import cv2
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from recipe_image_preprocessor import RecipeImagePreprocessor, PreprocessingResult
from text_detection import TextDetector
from ocr_engine import OCREngine
from enhanced_ingredient_parser import EnhancedIngredientParser, EnhancedIngredient
from ingredient_normalizer import IngredientNormalizer
from food_database_integration import FoodDatabaseIntegration

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class RecipeText:
    """Extracted text from recipe image."""
    text: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    region_type: str  # 'title', 'ingredient', 'instruction', 'other'
    ocr_engine: str


@dataclass
class RecipeIngredient:
    """Analyzed recipe ingredient."""
    original_text: str
    parsed_ingredient: EnhancedIngredient
    confidence: float
    nutritional_info: Optional[Dict[str, Any]]
    scaling_factor: float = 1.0


@dataclass
class RecipeNutrition:
    """Recipe nutritional information."""
    total_calories: Optional[float]
    calories_per_serving: Optional[float]
    total_protein_g: Optional[float]
    protein_per_serving_g: Optional[float]
    total_carbs_g: Optional[float]
    carbs_per_serving_g: Optional[float]
    total_fat_g: Optional[float]
    fat_per_serving_g: Optional[float]
    total_fiber_g: Optional[float]
    fiber_per_serving_g: Optional[float]
    total_sodium_mg: Optional[float]
    sodium_per_serving_mg: Optional[float]
    servings: Optional[int]
    calculation_method: str
    confidence: float


@dataclass
class RecipeAnalysisResult:
    """Complete recipe analysis result."""
    # Metadata
    analysis_id: str
    timestamp: str
    processing_time: float
    success: bool
    
    # Input information
    image_path: str
    image_size: Tuple[int, int]
    
    # Processing results
    preprocessing_result: PreprocessingResult
    extracted_texts: List[RecipeText]
    analyzed_ingredients: List[RecipeIngredient]
    nutritional_analysis: Optional[RecipeNutrition]
    
    # Recipe structure
    recipe_title: Optional[str]
    servings: Optional[int]
    instructions: List[str]
    
    # Analysis metadata
    confidence_scores: Dict[str, float]
    processing_steps: List[str]
    errors: List[str]
    warnings: List[str]


class CompleteRecipeAnalyzer:
    """Complete recipe analysis engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize complete recipe analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize components
        self.preprocessor = RecipeImagePreprocessor(self.config.get('preprocessing', {}))
        self.text_detector = TextDetector(self.config.get('text_detection', {}))
        self.ocr_engine = OCREngine(self.config.get('ocr', {}))
        self.ingredient_parser = EnhancedIngredientParser(self.config.get('ingredient_parsing', {}))
        self.normalizer = IngredientNormalizer(self.config.get('normalization', {}))
        self.database_integration = FoodDatabaseIntegration(self.config.get('database', {}))
        
        # Analysis settings
        self.min_text_confidence = self.config.get('min_text_confidence', 0.5)
        self.min_ingredient_confidence = self.config.get('min_ingredient_confidence', 0.3)
        self.nutrition_calculation_enabled = self.config.get('nutrition_calculation', True)
        
        # Region classification settings
        self.ingredient_keywords = self.config.get('ingredient_keywords', [
            'ingredients', 'ingredient', 'you will need', 'recipe requires',
            'for this recipe', 'shopping list', 'grocery list'
        ])
        
        self.instruction_keywords = self.config.get('instruction_keywords', [
            'instructions', 'method', 'directions', 'steps', 'how to make',
            'preparation', 'cooking method', 'recipe steps'
        ])
        
        self.logger.info("Initialized CompleteRecipeAnalyzer")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for analyzer."""
        logger = logging.getLogger('complete_recipe_analyzer')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def analyze_recipe(self, image_path: str) -> RecipeAnalysisResult:
        """
        Perform complete recipe analysis on an image.
        
        Args:
            image_path: Path to the recipe image
            
        Returns:
            Complete recipe analysis result
        """
        analysis_id = str(uuid.uuid4())
        start_time = time.time()
        processing_steps = []
        errors = []
        warnings = []
        
        try:
            # Step 1: Image preprocessing
            self.logger.info(f"Starting recipe analysis for: {image_path}")
            processing_steps.append("Starting image preprocessing")
            
            preprocessing_result = self.preprocessor.preprocess_image(image_path)
            if not preprocessing_result.success:
                errors.append("Image preprocessing failed")
                return self._create_failed_result(
                    analysis_id, image_path, preprocessing_result,
                    processing_steps, errors, warnings, time.time() - start_time
                )
            
            processing_steps.append("Image preprocessing completed")
            
            # Step 2: Text detection
            processing_steps.append("Starting text detection")
            
            detected_regions = self.text_detector.detect_text_regions(
                preprocessing_result.processed_image
            )
            
            if not detected_regions:
                warnings.append("No text regions detected")
            
            processing_steps.append(f"Text detection completed: {len(detected_regions)} regions")
            
            # Step 3: OCR extraction
            processing_steps.append("Starting OCR extraction")
            
            extracted_texts = []
            for region in detected_regions:
                # Extract region from image
                x1, y1, x2, y2 = region.bbox
                region_image = preprocessing_result.processed_image[y1:y2, x1:x2]
                
                # Perform OCR
                ocr_results = self.ocr_engine.extract_text_from_image(region_image)
                
                for ocr_result in ocr_results:
                    if ocr_result.confidence >= self.min_text_confidence:
                        # Classify region type
                        region_type = self._classify_text_region(ocr_result.text)
                        
                        extracted_texts.append(RecipeText(
                            text=ocr_result.text,
                            confidence=ocr_result.confidence,
                            bbox=[x1, y1, x2, y2],
                            region_type=region_type,
                            ocr_engine=ocr_result.engine
                        ))
            
            processing_steps.append(f"OCR extraction completed: {len(extracted_texts)} texts")
            
            # Step 4: Recipe structure analysis
            processing_steps.append("Analyzing recipe structure")
            
            recipe_title = self._extract_recipe_title(extracted_texts)
            servings = self._extract_servings(extracted_texts)
            instructions = self._extract_instructions(extracted_texts)
            
            processing_steps.append("Recipe structure analysis completed")
            
            # Step 5: Ingredient analysis
            processing_steps.append("Starting ingredient analysis")
            
            ingredient_texts = [
                text for text in extracted_texts 
                if text.region_type == 'ingredient'
            ]
            
            analyzed_ingredients = []
            for ingredient_text in ingredient_texts:
                try:
                    parsed_ingredient = self.ingredient_parser.parse_ingredient(ingredient_text.text)
                    
                    if parsed_ingredient.confidence >= self.min_ingredient_confidence:
                        # Calculate nutritional info if enabled
                        nutritional_info = None
                        if self.nutrition_calculation_enabled and parsed_ingredient.nutritional_info:
                            nutritional_info = parsed_ingredient.nutritional_info
                        
                        analyzed_ingredients.append(RecipeIngredient(
                            original_text=ingredient_text.text,
                            parsed_ingredient=parsed_ingredient,
                            confidence=parsed_ingredient.confidence,
                            nutritional_info=nutritional_info
                        ))
                    else:
                        warnings.append(f"Low confidence ingredient: {ingredient_text.text}")
                        
                except Exception as e:
                    errors.append(f"Failed to parse ingredient '{ingredient_text.text}': {str(e)}")
            
            processing_steps.append(f"Ingredient analysis completed: {len(analyzed_ingredients)} ingredients")
            
            # Step 6: Nutritional analysis
            nutritional_analysis = None
            if self.nutrition_calculation_enabled and analyzed_ingredients:
                processing_steps.append("Starting nutritional analysis")
                
                nutritional_analysis = self._calculate_recipe_nutrition(
                    analyzed_ingredients, servings
                )
                
                processing_steps.append("Nutritional analysis completed")
            
            # Step 7: Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                preprocessing_result, extracted_texts, analyzed_ingredients, nutritional_analysis
            )
            
            processing_time = time.time() - start_time
            
            # Create result
            result = RecipeAnalysisResult(
                analysis_id=analysis_id,
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time,
                success=True,
                image_path=image_path,
                image_size=preprocessing_result.original_size,
                preprocessing_result=preprocessing_result,
                extracted_texts=extracted_texts,
                analyzed_ingredients=analyzed_ingredients,
                nutritional_analysis=nutritional_analysis,
                recipe_title=recipe_title,
                servings=servings,
                instructions=instructions,
                confidence_scores=confidence_scores,
                processing_steps=processing_steps,
                errors=errors,
                warnings=warnings
            )
            
            self.logger.info(f"Recipe analysis completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Recipe analysis failed: {e}")
            errors.append(f"Analysis failed: {str(e)}")
            
            return self._create_failed_result(
                analysis_id, image_path, preprocessing_result if 'preprocessing_result' in locals() else None,
                processing_steps, errors, warnings, time.time() - start_time
            )
    
    def _create_failed_result(self, analysis_id: str, image_path: str, 
                            preprocessing_result: Optional[PreprocessingResult],
                            processing_steps: List[str], errors: List[str], 
                            warnings: List[str], processing_time: float) -> RecipeAnalysisResult:
        """Create a failed analysis result."""
        return RecipeAnalysisResult(
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            success=False,
            image_path=image_path,
            image_size=(0, 0),
            preprocessing_result=preprocessing_result,
            extracted_texts=[],
            analyzed_ingredients=[],
            nutritional_analysis=None,
            recipe_title=None,
            servings=None,
            instructions=[],
            confidence_scores={},
            processing_steps=processing_steps,
            errors=errors,
            warnings=warnings
        )
    
    def _classify_text_region(self, text: str) -> str:
        """
        Classify text region as title, ingredient, instruction, or other.
        
        Args:
            text: Text to classify
            
        Returns:
            Region type
        """
        text_lower = text.lower()
        
        # Check for ingredient indicators
        for keyword in self.ingredient_keywords:
            if keyword in text_lower:
                return 'ingredient'
        
        # Check for instruction indicators
        for keyword in self.instruction_keywords:
            if keyword in text_lower:
                return 'instruction'
        
        # Check if text looks like an ingredient (contains quantities/units)
        if self._looks_like_ingredient(text):
            return 'ingredient'
        
        # Check if text looks like instructions (contains verbs)
        if self._looks_like_instruction(text):
            return 'instruction'
        
        # Check if text looks like a title (short, no quantities)
        if self._looks_like_title(text):
            return 'title'
        
        return 'other'
    
    def _looks_like_ingredient(self, text: str) -> bool:
        """Check if text looks like an ingredient."""
        # Common units and measurements
        units = ['cup', 'cups', 'tbsp', 'tsp', 'lb', 'oz', 'g', 'kg', 'ml', 'l']
        quantities = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '/', '.']
        
        text_lower = text.lower()
        
        # Check for units
        for unit in units:
            if f' {unit} ' in text_lower or f' {unit}s ' in text_lower:
                return True
        
        # Check for quantities at the beginning
        if any(char in text[:10] for char in quantities):
            return True
        
        return False
    
    def _looks_like_instruction(self, text: str) -> bool:
        """Check if text looks like cooking instructions."""
        instruction_verbs = [
            'mix', 'stir', 'add', 'combine', 'heat', 'cook', 'bake', 'fry',
            'boil', 'simmer', 'whisk', 'blend', 'chop', 'slice', 'dice',
            'preheat', 'place', 'remove', 'serve', 'garnish', 'season'
        ]
        
        text_lower = text.lower()
        
        # Check for instruction verbs
        for verb in instruction_verbs:
            if verb in text_lower:
                return True
        
        # Check for numbered steps
        if text.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
            return True
        
        return False
    
    def _looks_like_title(self, text: str) -> bool:
        """Check if text looks like a recipe title."""
        # Titles are usually short and don't contain quantities
        if len(text) > 100:
            return False
        
        # Check for title indicators
        title_words = ['recipe', 'cake', 'cookies', 'soup', 'salad', 'pasta', 'bread']
        text_lower = text.lower()
        
        for word in title_words:
            if word in text_lower:
                return True
        
        # Check if it's the first significant text (likely a title)
        return len(text.split()) <= 6
    
    def _extract_recipe_title(self, extracted_texts: List[RecipeText]) -> Optional[str]:
        """Extract recipe title from extracted texts."""
        # Look for title regions first
        title_texts = [text for text in extracted_texts if text.region_type == 'title']
        
        if title_texts:
            # Return the highest confidence title
            return max(title_texts, key=lambda x: x.confidence).text
        
        # Fallback: look for short text at the top of the image
        other_texts = [text for text in extracted_texts if text.region_type == 'other']
        if other_texts:
            # Sort by vertical position (top first)
            other_texts.sort(key=lambda x: x.bbox[1])
            
            # Return the first short text
            for text in other_texts:
                if len(text.text.split()) <= 6:
                    return text.text
        
        return None
    
    def _extract_servings(self, extracted_texts: List[RecipeText]) -> Optional[int]:
        """Extract number of servings from extracted texts."""
        serving_patterns = [
            r'serves?\s+(\d+)', r'(\d+)\s+servings?', r'makes?\s+(\d+)',
            r'yield:?\s+(\d+)', r'portions?:?\s+(\d+)'
        ]
        
        import re
        
        for text in extracted_texts:
            text_lower = text.text.lower()
            for pattern in serving_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        return int(match.group(1))
                    except ValueError:
                        continue
        
        return None
    
    def _extract_instructions(self, extracted_texts: List[RecipeText]) -> List[str]:
        """Extract cooking instructions from extracted texts."""
        instruction_texts = [
            text.text for text in extracted_texts 
            if text.region_type == 'instruction'
        ]
        
        # Sort instructions by position (top to bottom)
        instruction_regions = [
            text for text in extracted_texts 
            if text.region_type == 'instruction'
        ]
        instruction_regions.sort(key=lambda x: x.bbox[1])
        
        return [text.text for text in instruction_regions]
    
    def _calculate_recipe_nutrition(self, ingredients: List[RecipeIngredient], 
                                  servings: Optional[int]) -> Optional[RecipeNutrition]:
        """Calculate total recipe nutrition from ingredients."""
        if not ingredients:
            return None
        
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0
        total_fiber = 0
        total_sodium = 0
        
        valid_ingredients = 0
        
        for ingredient in ingredients:
            if ingredient.nutritional_info:
                nutrition = ingredient.nutritional_info
                
                if nutrition.get('calories'):
                    total_calories += nutrition['calories'] * ingredient.scaling_factor
                    valid_ingredients += 1
                
                if nutrition.get('protein_g'):
                    total_protein += nutrition['protein_g'] * ingredient.scaling_factor
                
                if nutrition.get('carbs_g'):
                    total_carbs += nutrition['carbs_g'] * ingredient.scaling_factor
                
                if nutrition.get('fat_g'):
                    total_fat += nutrition['fat_g'] * ingredient.scaling_factor
                
                if nutrition.get('fiber_g'):
                    total_fiber += nutrition['fiber_g'] * ingredient.scaling_factor
                
                if nutrition.get('sodium_mg'):
                    total_sodium += nutrition['sodium_mg'] * ingredient.scaling_factor
        
        if valid_ingredients == 0:
            return None
        
        # Calculate per-serving values
        calories_per_serving = total_calories / servings if servings else None
        protein_per_serving = total_protein / servings if servings else None
        carbs_per_serving = total_carbs / servings if servings else None
        fat_per_serving = total_fat / servings if servings else None
        fiber_per_serving = total_fiber / servings if servings else None
        sodium_per_serving = total_sodium / servings if servings else None
        
        # Calculate confidence based on ingredient coverage
        confidence = valid_ingredients / len(ingredients)
        
        return RecipeNutrition(
            total_calories=total_calories,
            calories_per_serving=calories_per_serving,
            total_protein_g=total_protein,
            protein_per_serving_g=protein_per_serving,
            total_carbs_g=total_carbs,
            carbs_per_serving_g=carbs_per_serving,
            total_fat_g=total_fat,
            fat_per_serving_g=fat_per_serving,
            total_fiber_g=total_fiber,
            fiber_per_serving_g=fiber_per_serving,
            total_sodium_mg=total_sodium,
            sodium_per_serving_mg=sodium_per_serving,
            servings=servings,
            calculation_method="ingredient_sum",
            confidence=confidence
        )
    
    def _calculate_confidence_scores(self, preprocessing_result: PreprocessingResult,
                                   extracted_texts: List[RecipeText],
                                   analyzed_ingredients: List[RecipeIngredient],
                                   nutritional_analysis: Optional[RecipeNutrition]) -> Dict[str, float]:
        """Calculate confidence scores for different aspects of the analysis."""
        scores = {}
        
        # Image preprocessing confidence
        scores['preprocessing'] = preprocessing_result.quality_score
        
        # Text extraction confidence
        if extracted_texts:
            scores['text_extraction'] = sum(text.confidence for text in extracted_texts) / len(extracted_texts)
        else:
            scores['text_extraction'] = 0.0
        
        # Ingredient parsing confidence
        if analyzed_ingredients:
            scores['ingredient_parsing'] = sum(ing.confidence for ing in analyzed_ingredients) / len(analyzed_ingredients)
        else:
            scores['ingredient_parsing'] = 0.0
        
        # Nutritional analysis confidence
        if nutritional_analysis:
            scores['nutritional_analysis'] = nutritional_analysis.confidence
        else:
            scores['nutritional_analysis'] = 0.0
        
        # Overall confidence
        valid_scores = [score for score in scores.values() if score > 0]
        scores['overall'] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        return scores
    
    def scale_recipe(self, result: RecipeAnalysisResult, scale_factor: float) -> RecipeAnalysisResult:
        """
        Scale recipe quantities by a given factor.
        
        Args:
            result: Original recipe analysis result
            scale_factor: Scaling factor (e.g., 2.0 for double recipe)
            
        Returns:
            Scaled recipe analysis result
        """
        if not result.success:
            return result
        
        # Scale ingredients
        scaled_ingredients = []
        for ingredient in result.analyzed_ingredients:
            scaled_ingredient = RecipeIngredient(
                original_text=ingredient.original_text,
                parsed_ingredient=ingredient.parsed_ingredient,
                confidence=ingredient.confidence,
                nutritional_info=ingredient.nutritional_info,
                scaling_factor=ingredient.scaling_factor * scale_factor
            )
            scaled_ingredients.append(scaled_ingredient)
        
        # Scale nutritional analysis
        scaled_nutrition = None
        if result.nutritional_analysis:
            nutrition = result.nutritional_analysis
            scaled_nutrition = RecipeNutrition(
                total_calories=nutrition.total_calories * scale_factor if nutrition.total_calories else None,
                calories_per_serving=nutrition.calories_per_serving,  # Per serving stays the same
                total_protein_g=nutrition.total_protein_g * scale_factor if nutrition.total_protein_g else None,
                protein_per_serving_g=nutrition.protein_per_serving_g,
                total_carbs_g=nutrition.total_carbs_g * scale_factor if nutrition.total_carbs_g else None,
                carbs_per_serving_g=nutrition.carbs_per_serving_g,
                total_fat_g=nutrition.total_fat_g * scale_factor if nutrition.total_fat_g else None,
                fat_per_serving_g=nutrition.fat_per_serving_g,
                total_fiber_g=nutrition.total_fiber_g * scale_factor if nutrition.total_fiber_g else None,
                fiber_per_serving_g=nutrition.fiber_per_serving_g,
                total_sodium_mg=nutrition.total_sodium_mg * scale_factor if nutrition.total_sodium_mg else None,
                sodium_per_serving_mg=nutrition.sodium_per_serving_mg,
                servings=int(nutrition.servings * scale_factor) if nutrition.servings else None,
                calculation_method=nutrition.calculation_method,
                confidence=nutrition.confidence
            )
        
        # Create scaled result
        scaled_result = RecipeAnalysisResult(
            analysis_id=result.analysis_id + f"_scaled_{scale_factor}",
            timestamp=datetime.now().isoformat(),
            processing_time=result.processing_time,
            success=result.success,
            image_path=result.image_path,
            image_size=result.image_size,
            preprocessing_result=result.preprocessing_result,
            extracted_texts=result.extracted_texts,
            analyzed_ingredients=scaled_ingredients,
            nutritional_analysis=scaled_nutrition,
            recipe_title=result.recipe_title,
            servings=int(result.servings * scale_factor) if result.servings else None,
            instructions=result.instructions,
            confidence_scores=result.confidence_scores,
            processing_steps=result.processing_steps + [f"Recipe scaled by factor {scale_factor}"],
            errors=result.errors,
            warnings=result.warnings
        )
        
        return scaled_result
    
    def export_analysis_result(self, result: RecipeAnalysisResult, output_path: str) -> bool:
        """
        Export analysis result to JSON file.
        
        Args:
            result: Recipe analysis result
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to dictionary
            result_dict = asdict(result)
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export analysis result: {e}")
            return False
    
    def batch_analyze_recipes(self, image_paths: List[str], output_dir: str) -> List[RecipeAnalysisResult]:
        """
        Batch analyze multiple recipe images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory for results
            
        Returns:
            List of analysis results
        """
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"Analyzing recipe {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.analyze_recipe(image_path)
            results.append(result)
            
            # Export individual result
            filename = f"{Path(image_path).stem}_analysis.json"
            output_file = output_path / filename
            self.export_analysis_result(result, str(output_file))
        
        # Export batch summary
        batch_summary = self._generate_batch_summary(results)
        summary_file = output_path / "batch_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        return results
    
    def _generate_batch_summary(self, results: List[RecipeAnalysisResult]) -> Dict[str, Any]:
        """Generate batch analysis summary."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        return {
            "summary": {
                "total_recipes": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(results) if results else 0,
                "total_processing_time": sum(r.processing_time for r in results),
                "average_processing_time": sum(r.processing_time for r in results) / len(results) if results else 0
            },
            "confidence_statistics": {
                "average_overall_confidence": sum(r.confidence_scores.get('overall', 0) for r in successful) / len(successful) if successful else 0,
                "average_preprocessing_confidence": sum(r.confidence_scores.get('preprocessing', 0) for r in successful) / len(successful) if successful else 0,
                "average_text_extraction_confidence": sum(r.confidence_scores.get('text_extraction', 0) for r in successful) / len(successful) if successful else 0,
                "average_ingredient_parsing_confidence": sum(r.confidence_scores.get('ingredient_parsing', 0) for r in successful) / len(successful) if successful else 0
            },
            "content_statistics": {
                "recipes_with_titles": len([r for r in successful if r.recipe_title]),
                "recipes_with_servings": len([r for r in successful if r.servings]),
                "recipes_with_instructions": len([r for r in successful if r.instructions]),
                "recipes_with_nutrition": len([r for r in successful if r.nutritional_analysis]),
                "total_ingredients_found": sum(len(r.analyzed_ingredients) for r in successful),
                "average_ingredients_per_recipe": sum(len(r.analyzed_ingredients) for r in successful) / len(successful) if successful else 0
            },
            "failed_analyses": [
                {
                    "image_path": r.image_path,
                    "errors": r.errors,
                    "processing_time": r.processing_time
                }
                for r in failed
            ]
        }


def main():
    """Main recipe analysis script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete recipe analysis')
    parser.add_argument('--input', '-i', required=True, help='Input image or directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--config', '-c', help='Configuration file (JSON)')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    parser.add_argument('--scale', type=float, help='Scale recipe by factor')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize analyzer
    analyzer = CompleteRecipeAnalyzer(config)
    
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
            
            print(f"Found {len(image_paths)} images to analyze")
            
            # Analyze images
            results = analyzer.batch_analyze_recipes(image_paths, args.output)
            
            # Print summary
            successful = len([r for r in results if r.success])
            print(f"Analysis completed: {successful}/{len(results)} successful")
            
        else:
            # Single image analysis
            result = analyzer.analyze_recipe(args.input)
            
            # Scale if requested
            if args.scale:
                result = analyzer.scale_recipe(result, args.scale)
            
            if result.success:
                # Export result
                output_path = Path(args.output)
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / f"{Path(args.input).stem}_analysis.json"
                
                if analyzer.export_analysis_result(result, str(output_file)):
                    print(f"Analysis result saved to: {output_file}")
                    
                    # Print summary
                    print(f"Recipe Analysis Summary:")
                    print(f"  Title: {result.recipe_title or 'Not detected'}")
                    print(f"  Servings: {result.servings or 'Not detected'}")
                    print(f"  Ingredients: {len(result.analyzed_ingredients)}")
                    print(f"  Instructions: {len(result.instructions)}")
                    print(f"  Overall confidence: {result.confidence_scores.get('overall', 0):.3f}")
                    
                    if result.nutritional_analysis:
                        nutrition = result.nutritional_analysis
                        print(f"  Total calories: {nutrition.total_calories:.1f}" if nutrition.total_calories else "")
                        print(f"  Calories per serving: {nutrition.calories_per_serving:.1f}" if nutrition.calories_per_serving else "")
                    
                    print(f"  Processing time: {result.processing_time:.2f}s")
                    
                    if args.verbose:
                        print(f"\nDetailed Results:")
                        for i, ingredient in enumerate(result.analyzed_ingredients, 1):
                            print(f"  {i}. {ingredient.original_text} → {ingredient.parsed_ingredient.standardized_format}")
                else:
                    print("Failed to save analysis result")
                    return 1
            else:
                print("Recipe analysis failed")
                print(f"Errors: {'; '.join(result.errors)}")
                return 1
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())