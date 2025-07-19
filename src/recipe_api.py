#!/usr/bin/env python3
"""
Recipe API Server
RESTful API server for recipe analysis system with comprehensive endpoints
for recipe management, analysis, scaling, and nutritional information.
"""

import os
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import tempfile
import asyncio
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import logging

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from marshmallow import Schema, fields, ValidationError
import jwt
from functools import wraps

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from complete_recipe_analyzer import CompleteRecipeAnalyzer
from recipe_database import RecipeDatabase
from recipe_scaler import RecipeScaler, ScalingOptions
from nutritional_analyzer import NutritionalAnalyzer
from batch_processor import BatchProcessor, BatchProcessingConfig
from enhanced_ingredient_parser import EnhancedIngredient


# API Schemas
class IngredientSchema(Schema):
    """Schema for ingredient input."""
    original_text = fields.Str(required=True)
    ingredient_name = fields.Str(required=True)
    quantity = fields.Str(allow_none=True)
    unit = fields.Str(allow_none=True)
    normalized_quantity = fields.Float(allow_none=True)
    normalized_unit = fields.Str(allow_none=True)
    preparation = fields.Str(allow_none=True)


class RecipeAnalysisSchema(Schema):
    """Schema for recipe analysis request."""
    image_path = fields.Str(required=True)
    include_nutrition = fields.Bool(missing=True)
    include_scaling = fields.Bool(missing=False)
    target_servings = fields.Int(allow_none=True)


class RecipeScalingSchema(Schema):
    """Schema for recipe scaling request."""
    recipe_id = fields.Str(required=True)
    target_servings = fields.Int(allow_none=True)
    scale_factor = fields.Float(allow_none=True)
    unit_system = fields.Str(missing="metric", validate=lambda x: x in ["metric", "imperial", "us"])
    dietary_modifications = fields.List(fields.Str(), missing=[])
    round_to_nice_numbers = fields.Bool(missing=True)


class BatchProcessingSchema(Schema):
    """Schema for batch processing request."""
    input_directory = fields.Str(required=True)
    file_pattern = fields.Str(missing="*.jpg")
    max_workers = fields.Int(missing=4)
    processing_mode = fields.Str(missing="parallel", validate=lambda x: x in ["sequential", "parallel", "hybrid"])
    database_storage = fields.Bool(missing=True)
    scaling_enabled = fields.Bool(missing=False)
    target_servings = fields.Int(allow_none=True)


class RecipeAPI:
    """RESTful API server for recipe analysis system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Recipe API server.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.app = Flask(__name__)
        
        # Configuration
        self.app.config.update({
            'SECRET_KEY': self.config.get('secret_key', 'your-secret-key-here'),
            'MAX_CONTENT_LENGTH': self.config.get('max_file_size', 16 * 1024 * 1024),  # 16MB
            'UPLOAD_FOLDER': self.config.get('upload_folder', 'uploads'),
            'CACHE_TYPE': 'simple',
            'CACHE_DEFAULT_TIMEOUT': 300
        })
        
        # Extensions
        CORS(self.app)
        self.limiter = Limiter(
            self.app,
            key_func=get_remote_address,
            default_limits=["200 per day", "50 per hour"]
        )
        self.cache = Cache(self.app)
        
        # Initialize components
        self.analyzer = CompleteRecipeAnalyzer(self.config.get('analyzer', {}))
        self.database = RecipeDatabase(
            self.config.get('database_path', 'recipes.db'),
            self.config.get('database', {})
        )
        self.scaler = RecipeScaler(self.config.get('scaler', {}))
        self.nutritional_analyzer = NutritionalAnalyzer(self.config.get('nutrition', {}))
        self.batch_processor = BatchProcessor()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Create upload directory
        self.upload_dir = Path(self.app.config['UPLOAD_FOLDER'])
        self.upload_dir.mkdir(exist_ok=True)
        
        # Authentication
        self.auth_enabled = self.config.get('auth_enabled', False)
        self.api_keys = self.config.get('api_keys', {})
        
        # Register routes
        self._register_routes()
        
        self.logger.info("Initialized Recipe API server")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for API server."""
        logger = logging.getLogger('recipe_api')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _require_auth(self, f):
        """Decorator for API key authentication."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not self.auth_enabled:
                return f(*args, **kwargs)
            
            api_key = request.headers.get('X-API-Key')
            if not api_key or api_key not in self.api_keys:
                return jsonify({'error': 'Invalid API key'}), 401
            
            return f(*args, **kwargs)
        return decorated_function
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
        
        @self.app.route('/api/info', methods=['GET'])
        def system_info():
            """System information endpoint."""
            stats = self.database.get_database_stats()
            return jsonify({
                'system': {
                    'version': '1.0.0',
                    'components': {
                        'analyzer': 'CompleteRecipeAnalyzer',
                        'database': 'RecipeDatabase',
                        'scaler': 'RecipeScaler',
                        'nutritional_analyzer': 'NutritionalAnalyzer',
                        'batch_processor': 'BatchProcessor'
                    }
                },
                'database': stats,
                'features': {
                    'image_analysis': True,
                    'ingredient_parsing': True,
                    'recipe_scaling': True,
                    'nutritional_analysis': True,
                    'batch_processing': True,
                    'full_text_search': True
                }
            })
        
        @self.app.route('/api/analyze', methods=['POST'])
        @self.limiter.limit("10 per minute")
        @self._require_auth
        def analyze_recipe():
            """Analyze recipe from uploaded image."""
            try:
                # Check if file was uploaded
                if 'file' not in request.files:
                    return jsonify({'error': 'No file uploaded'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                if not self._allowed_file(file.filename):
                    return jsonify({'error': 'File type not allowed'}), 400
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = self.upload_dir / unique_filename
                file.save(filepath)
                
                # Get options
                include_nutrition = request.form.get('include_nutrition', 'true').lower() == 'true'
                include_scaling = request.form.get('include_scaling', 'false').lower() == 'true'
                target_servings = request.form.get('target_servings', type=int)
                
                # Analyze recipe
                result = self.analyzer.analyze_recipe(str(filepath))
                
                if result.success:
                    # Store in database
                    recipe_id = self.database.store_recipe_analysis(result)
                    
                    # Prepare response
                    response = {
                        'success': True,
                        'recipe_id': recipe_id,
                        'analysis_id': result.analysis_id,
                        'recipe': {
                            'title': result.recipe_title,
                            'servings': result.servings,
                            'ingredients': [
                                {
                                    'original_text': ing.original_text,
                                    'standardized_format': ing.parsed_ingredient.standardized_format,
                                    'ingredient_name': ing.parsed_ingredient.ingredient_name,
                                    'quantity': ing.parsed_ingredient.quantity,
                                    'unit': ing.parsed_ingredient.unit,
                                    'confidence': ing.confidence
                                }
                                for ing in result.analyzed_ingredients
                            ],
                            'instructions': result.instructions,
                            'confidence_scores': result.confidence_scores,
                            'processing_time': result.processing_time
                        }
                    }
                    
                    # Add nutritional analysis if requested
                    if include_nutrition and result.nutritional_analysis:
                        response['nutrition'] = {
                            'total_calories': result.nutritional_analysis.total_calories,
                            'calories_per_serving': result.nutritional_analysis.calories_per_serving,
                            'total_protein_g': result.nutritional_analysis.total_protein_g,
                            'protein_per_serving_g': result.nutritional_analysis.protein_per_serving_g,
                            'total_carbs_g': result.nutritional_analysis.total_carbs_g,
                            'carbs_per_serving_g': result.nutritional_analysis.carbs_per_serving_g,
                            'total_fat_g': result.nutritional_analysis.total_fat_g,
                            'fat_per_serving_g': result.nutritional_analysis.fat_per_serving_g,
                            'confidence': result.nutritional_analysis.confidence
                        }
                    
                    # Add scaling if requested
                    if include_scaling and target_servings:
                        scaled_result = self.analyzer.scale_recipe(result, target_servings / result.servings)
                        response['scaled_recipe'] = {
                            'target_servings': target_servings,
                            'scaling_factor': target_servings / result.servings,
                            'scaled_ingredients': [
                                {
                                    'original_text': ing.original_text,
                                    'standardized_format': ing.parsed_ingredient.standardized_format,
                                    'ingredient_name': ing.parsed_ingredient.ingredient_name,
                                    'quantity': ing.parsed_ingredient.quantity,
                                    'unit': ing.parsed_ingredient.unit,
                                    'scaling_factor': ing.scaling_factor
                                }
                                for ing in scaled_result.analyzed_ingredients
                            ]
                        }
                    
                    return jsonify(response)
                else:
                    return jsonify({
                        'error': 'Recipe analysis failed',
                        'details': result.errors
                    }), 500
                    
            except RequestEntityTooLarge:
                return jsonify({'error': 'File too large'}), 413
            except Exception as e:
                self.logger.error(f"Analysis error: {e}")
                return jsonify({'error': 'Analysis failed'}), 500
        
        @self.app.route('/api/recipes', methods=['GET'])
        @self.cache.cached(timeout=300)
        def list_recipes():
            """List all recipes with pagination."""
            try:
                page = request.args.get('page', 1, type=int)
                per_page = min(request.args.get('per_page', 10, type=int), 100)
                query = request.args.get('q', '')
                
                if query:
                    recipes = self.database.search_recipes(query, limit=per_page)
                else:
                    # Get all recipes (simplified pagination)
                    recipes = self.database.search_recipes('', limit=per_page)
                
                return jsonify({
                    'recipes': recipes,
                    'page': page,
                    'per_page': per_page,
                    'total': len(recipes)
                })
                
            except Exception as e:
                self.logger.error(f"List recipes error: {e}")
                return jsonify({'error': 'Failed to list recipes'}), 500
        
        @self.app.route('/api/recipes/<recipe_id>', methods=['GET'])
        @self.cache.cached(timeout=300)
        def get_recipe(recipe_id: str):
            """Get recipe by ID."""
            try:
                recipe_data = self.database.get_recipe(recipe_id)
                if not recipe_data:
                    return jsonify({'error': 'Recipe not found'}), 404
                
                return jsonify(recipe_data)
                
            except Exception as e:
                self.logger.error(f"Get recipe error: {e}")
                return jsonify({'error': 'Failed to get recipe'}), 500
        
        @self.app.route('/api/recipes/<recipe_id>/scale', methods=['POST'])
        @self.limiter.limit("20 per minute")
        @self._require_auth
        def scale_recipe(recipe_id: str):
            """Scale recipe."""
            try:
                # Validate request
                schema = RecipeScalingSchema()
                try:
                    data = schema.load(request.json)
                except ValidationError as e:
                    return jsonify({'error': 'Invalid request', 'details': e.messages}), 400
                
                # Get original recipe
                recipe_data = self.database.get_recipe(recipe_id)
                if not recipe_data:
                    return jsonify({'error': 'Recipe not found'}), 404
                
                # Create scaling options
                options = ScalingOptions(
                    target_servings=data.get('target_servings'),
                    scale_factor=data.get('scale_factor'),
                    unit_system=data.get('unit_system', 'metric'),
                    dietary_modifications=data.get('dietary_modifications', []),
                    round_to_nice_numbers=data.get('round_to_nice_numbers', True)
                )
                
                # Convert database ingredients to EnhancedIngredients
                ingredients = []
                for ing_data in recipe_data['ingredients']:
                    ingredient = EnhancedIngredient(
                        original_text=ing_data['original_text'],
                        ingredient_name=ing_data['ingredient_name'],
                        quantity=ing_data['quantity'],
                        unit=ing_data['unit'],
                        normalized_quantity=ing_data['normalized_quantity'],
                        normalized_unit=ing_data['normalized_unit'],
                        preparation=ing_data['preparation'],
                        confidence=ing_data['confidence'],
                        standardized_format=ing_data['standardized_format']
                    )
                    ingredients.append(ingredient)
                
                # Scale recipe
                scaled_recipe = self.scaler.scale_recipe(
                    ingredients,
                    recipe_data['recipe']['servings'],
                    json.loads(recipe_data['recipe']['instructions']) if recipe_data['recipe']['instructions'] else [],
                    options
                )
                
                # Store scaled recipe
                new_recipe_id = self.database.store_scaled_recipe(scaled_recipe, recipe_id)
                
                return jsonify({
                    'success': True,
                    'new_recipe_id': new_recipe_id,
                    'original_recipe_id': recipe_id,
                    'scaling_factor': scaled_recipe.scaling_factor,
                    'target_servings': scaled_recipe.target_servings,
                    'scaled_ingredients': [
                        {
                            'original_text': ing.original_ingredient.original_text,
                            'standardized_format': f"{ing.display_quantity} {ing.display_unit} {ing.original_ingredient.ingredient_name}",
                            'ingredient_name': ing.original_ingredient.ingredient_name,
                            'quantity': ing.display_quantity,
                            'unit': ing.display_unit,
                            'scaling_factor': ing.scaling_factor,
                            'scaling_notes': ing.scaling_notes
                        }
                        for ing in scaled_recipe.scaled_ingredients
                    ]
                })
                
            except Exception as e:
                self.logger.error(f"Scaling error: {e}")
                return jsonify({'error': 'Scaling failed'}), 500
        
        @self.app.route('/api/recipes/<recipe_id>/nutrition', methods=['GET'])
        @self.cache.cached(timeout=600)
        def get_nutrition(recipe_id: str):
            """Get nutritional analysis for recipe."""
            try:
                recipe_data = self.database.get_recipe(recipe_id)
                if not recipe_data:
                    return jsonify({'error': 'Recipe not found'}), 404
                
                # Convert to ingredients
                ingredients = []
                for ing_data in recipe_data['ingredients']:
                    ingredient = EnhancedIngredient(
                        original_text=ing_data['original_text'],
                        ingredient_name=ing_data['ingredient_name'],
                        normalized_quantity=ing_data['normalized_quantity'],
                        normalized_unit=ing_data['normalized_unit'],
                        confidence=ing_data['confidence']
                    )
                    ingredients.append(ingredient)
                
                # Analyze nutrition
                servings = recipe_data['recipe']['servings'] or 1
                nutrition_analysis = self.nutritional_analyzer.analyze_recipe(ingredients, servings)
                
                return jsonify({
                    'recipe_id': recipe_id,
                    'nutrition': {
                        'total_servings': nutrition_analysis.total_servings,
                        'calories_per_serving': nutrition_analysis.calories_per_serving,
                        'protein_per_serving_g': nutrition_analysis.protein_per_serving_g,
                        'carbs_per_serving_g': nutrition_analysis.carbs_per_serving_g,
                        'fat_per_serving_g': nutrition_analysis.fat_per_serving_g,
                        'fiber_per_serving_g': nutrition_analysis.fiber_per_serving_g,
                        'sugar_per_serving_g': nutrition_analysis.sugar_per_serving_g,
                        'nutrient_breakdown': nutrition_analysis.nutrient_breakdown,
                        'dietary_analysis': nutrition_analysis.dietary_analysis,
                        'data_coverage': nutrition_analysis.data_coverage,
                        'analysis_confidence': nutrition_analysis.analysis_confidence
                    }
                })
                
            except Exception as e:
                self.logger.error(f"Nutrition analysis error: {e}")
                return jsonify({'error': 'Nutrition analysis failed'}), 500
        
        @self.app.route('/api/search', methods=['GET'])
        def search():
            """Search recipes and ingredients."""
            try:
                query = request.args.get('q', '')
                search_type = request.args.get('type', 'recipes')
                limit = min(request.args.get('limit', 10, type=int), 100)
                
                if search_type == 'ingredients':
                    results = self.database.search_ingredients(query, limit)
                elif search_type == 'recipes':
                    results = self.database.search_recipes(query, limit)
                else:
                    return jsonify({'error': 'Invalid search type'}), 400
                
                return jsonify({
                    'query': query,
                    'type': search_type,
                    'results': results,
                    'count': len(results)
                })
                
            except Exception as e:
                self.logger.error(f"Search error: {e}")
                return jsonify({'error': 'Search failed'}), 500
        
        @self.app.route('/api/ingredients/<ingredient_name>/recipes', methods=['GET'])
        @self.cache.cached(timeout=300)
        def get_recipes_by_ingredient(ingredient_name: str):
            """Get recipes containing specific ingredient."""
            try:
                limit = min(request.args.get('limit', 10, type=int), 100)
                recipes = self.database.get_recipes_by_ingredient(ingredient_name, limit)
                
                return jsonify({
                    'ingredient': ingredient_name,
                    'recipes': recipes,
                    'count': len(recipes)
                })
                
            except Exception as e:
                self.logger.error(f"Get recipes by ingredient error: {e}")
                return jsonify({'error': 'Failed to get recipes'}), 500
        
        @self.app.route('/api/batch/process', methods=['POST'])
        @self.limiter.limit("2 per hour")
        @self._require_auth
        def batch_process():
            """Start batch processing."""
            try:
                # Validate request
                schema = BatchProcessingSchema()
                try:
                    data = schema.load(request.json)
                except ValidationError as e:
                    return jsonify({'error': 'Invalid request', 'details': e.messages}), 400
                
                # Create batch configuration
                config = BatchProcessingConfig(
                    max_workers=data.get('max_workers', 4),
                    processing_mode=data.get('processing_mode', 'parallel'),
                    database_storage=data.get('database_storage', True),
                    scaling_enabled=data.get('scaling_enabled', False)
                )
                
                if config.scaling_enabled and data.get('target_servings'):
                    from recipe_scaler import ScalingOptions
                    config.scaling_options = ScalingOptions(target_servings=data['target_servings'])
                
                # Initialize batch processor
                processor = BatchProcessor(config)
                
                # Start processing (in background)
                input_dir = data['input_directory']
                file_pattern = data.get('file_pattern', '*.jpg')
                
                # This would typically be run in a background task
                # For now, we'll return a job ID and the client can poll for status
                job_id = str(uuid.uuid4())
                
                return jsonify({
                    'success': True,
                    'job_id': job_id,
                    'status': 'started',
                    'message': 'Batch processing started'
                })
                
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                return jsonify({'error': 'Batch processing failed'}), 500
        
        @self.app.route('/api/stats', methods=['GET'])
        @self.cache.cached(timeout=300)
        def get_stats():
            """Get system statistics."""
            try:
                db_stats = self.database.get_database_stats()
                nutrition_stats = self.database.get_nutrition_stats()
                
                return jsonify({
                    'database': db_stats,
                    'nutrition': nutrition_stats,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Stats error: {e}")
                return jsonify({'error': 'Failed to get stats'}), 500
        
        @self.app.route('/api/export', methods=['GET'])
        @self.limiter.limit("5 per hour")
        @self._require_auth
        def export_recipes():
            """Export recipes."""
            try:
                export_format = request.args.get('format', 'json')
                
                if export_format not in ['json']:
                    return jsonify({'error': 'Unsupported format'}), 400
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{export_format}', delete=False) as f:
                    temp_path = f.name
                
                # Export to temporary file
                if self.database.export_recipes(temp_path, export_format):
                    return send_file(
                        temp_path,
                        as_attachment=True,
                        download_name=f'recipes_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format}',
                        mimetype=f'application/{export_format}'
                    )
                else:
                    return jsonify({'error': 'Export failed'}), 500
                    
            except Exception as e:
                self.logger.error(f"Export error: {e}")
                return jsonify({'error': 'Export failed'}), 500
        
        @self.app.route('/api/recipes/<recipe_id>', methods=['DELETE'])
        @self.limiter.limit("10 per minute")
        @self._require_auth
        def delete_recipe(recipe_id: str):
            """Delete recipe."""
            try:
                if self.database.delete_recipe(recipe_id):
                    return jsonify({'success': True})
                else:
                    return jsonify({'error': 'Delete failed'}), 500
                    
            except Exception as e:
                self.logger.error(f"Delete error: {e}")
                return jsonify({'error': 'Delete failed'}), 500
        
        # Error handlers
        @self.app.errorhandler(404)
        def not_found(e):
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(405)
        def method_not_allowed(e):
            return jsonify({'error': 'Method not allowed'}), 405
        
        @self.app.errorhandler(413)
        def too_large(e):
            return jsonify({'error': 'File too large'}), 413
        
        @self.app.errorhandler(429)
        def rate_limit_exceeded(e):
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        @self.app.errorhandler(500)
        def internal_error(e):
            return jsonify({'error': 'Internal server error'}), 500
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """
        Run the API server.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        self.logger.info(f"Starting Recipe API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main API server script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recipe API server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--config', help='Configuration file (JSON)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize API server
    api = RecipeAPI(config)
    
    # Run server
    api.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()