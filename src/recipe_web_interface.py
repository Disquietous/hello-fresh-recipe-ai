#!/usr/bin/env python3
"""
Recipe Web Interface
Flask web application for uploading recipe images, viewing extracted ingredients,
and managing the recipe database with a modern, responsive UI.
"""

import os
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import tempfile
import mimetypes

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import logging

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from complete_recipe_analyzer import CompleteRecipeAnalyzer
from recipe_database import RecipeDatabase
from recipe_scaler import RecipeScaler, ScalingOptions


class RecipeWebInterface:
    """Flask web interface for recipe analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize web interface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.app = Flask(__name__)
        self.app.secret_key = self.config.get('secret_key', 'your-secret-key-here')
        
        # Configuration
        self.upload_folder = Path(self.config.get('upload_folder', 'uploads'))
        self.upload_folder.mkdir(exist_ok=True)
        self.max_file_size = self.config.get('max_file_size', 16 * 1024 * 1024)  # 16MB
        self.allowed_extensions = self.config.get('allowed_extensions', {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'})
        
        # Initialize components
        self.analyzer = CompleteRecipeAnalyzer(self.config.get('analyzer', {}))
        self.database = RecipeDatabase(
            self.config.get('database_path', 'recipes.db'),
            self.config.get('database', {})
        )
        self.scaler = RecipeScaler(self.config.get('scaler', {}))
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Setup Flask configuration
        self.app.config['MAX_CONTENT_LENGTH'] = self.max_file_size
        self.app.config['UPLOAD_FOLDER'] = str(self.upload_folder)
        
        # Register routes
        self._register_routes()
        
        self.logger.info("Initialized RecipeWebInterface")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for web interface."""
        logger = logging.getLogger('recipe_web_interface')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def _register_routes(self):
        """Register Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main page."""
            return render_template('index.html')
        
        @self.app.route('/upload', methods=['GET', 'POST'])
        def upload_recipe():
            """Upload recipe image for analysis."""
            if request.method == 'GET':
                return render_template('upload.html')
            
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
                filepath = self.upload_folder / unique_filename
                file.save(filepath)
                
                # Analyze recipe
                result = self.analyzer.analyze_recipe(str(filepath))
                
                if result.success:
                    # Store in database
                    recipe_id = self.database.store_recipe_analysis(result)
                    
                    return jsonify({
                        'success': True,
                        'recipe_id': recipe_id,
                        'analysis_id': result.analysis_id,
                        'redirect_url': url_for('view_recipe', recipe_id=recipe_id)
                    })
                else:
                    return jsonify({
                        'error': 'Recipe analysis failed',
                        'details': result.errors
                    }), 500
                    
            except RequestEntityTooLarge:
                return jsonify({'error': 'File too large'}), 413
            except Exception as e:
                self.logger.error(f"Upload error: {e}")
                return jsonify({'error': 'Upload failed'}), 500
        
        @self.app.route('/recipe/<recipe_id>')
        def view_recipe(recipe_id: str):
            """View recipe details."""
            recipe_data = self.database.get_recipe(recipe_id)
            if not recipe_data:
                flash('Recipe not found', 'error')
                return redirect(url_for('index'))
            
            return render_template('recipe.html', recipe=recipe_data)
        
        @self.app.route('/api/recipe/<recipe_id>')
        def api_get_recipe(recipe_id: str):
            """API endpoint to get recipe data."""
            recipe_data = self.database.get_recipe(recipe_id)
            if not recipe_data:
                return jsonify({'error': 'Recipe not found'}), 404
            
            return jsonify(recipe_data)
        
        @self.app.route('/api/scale/<recipe_id>', methods=['POST'])
        def api_scale_recipe(recipe_id: str):
            """API endpoint to scale recipe."""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
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
                # This is a simplified conversion - in practice you'd need proper reconstruction
                from enhanced_ingredient_parser import EnhancedIngredient
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
                    'scaling_factor': scaled_recipe.scaling_factor,
                    'target_servings': scaled_recipe.target_servings
                })
                
            except Exception as e:
                self.logger.error(f"Scaling error: {e}")
                return jsonify({'error': 'Scaling failed'}), 500
        
        @self.app.route('/search')
        def search_recipes():
            """Search recipes."""
            query = request.args.get('q', '')
            if not query:
                return render_template('search.html', recipes=[], query='')
            
            results = self.database.search_recipes(query, limit=20)
            return render_template('search.html', recipes=results, query=query)
        
        @self.app.route('/api/search')
        def api_search():
            """API endpoint for search."""
            query = request.args.get('q', '')
            search_type = request.args.get('type', 'recipes')  # 'recipes' or 'ingredients'
            limit = int(request.args.get('limit', 10))
            
            if search_type == 'ingredients':
                results = self.database.search_ingredients(query, limit)
            else:
                results = self.database.search_recipes(query, limit)
            
            return jsonify(results)
        
        @self.app.route('/browse')
        def browse_recipes():
            """Browse all recipes."""
            # Get all recipes (simplified - in practice you'd want pagination)
            results = self.database.search_recipes('', limit=100)
            return render_template('browse.html', recipes=results)
        
        @self.app.route('/stats')
        def view_stats():
            """View database statistics."""
            stats = self.database.get_database_stats()
            nutrition_stats = self.database.get_nutrition_stats()
            return render_template('stats.html', stats=stats, nutrition=nutrition_stats)
        
        @self.app.route('/api/stats')
        def api_stats():
            """API endpoint for statistics."""
            stats = self.database.get_database_stats()
            nutrition_stats = self.database.get_nutrition_stats()
            return jsonify({
                'database': stats,
                'nutrition': nutrition_stats
            })
        
        @self.app.route('/ingredient/<ingredient_name>')
        def view_ingredient(ingredient_name: str):
            """View recipes containing specific ingredient."""
            recipes = self.database.get_recipes_by_ingredient(ingredient_name, limit=20)
            return render_template('ingredient.html', ingredient=ingredient_name, recipes=recipes)
        
        @self.app.route('/api/export')
        def api_export():
            """Export recipes to JSON."""
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    temp_path = f.name
                
                # Export to temporary file
                if self.database.export_recipes(temp_path, 'json'):
                    return send_file(
                        temp_path,
                        as_attachment=True,
                        download_name=f'recipes_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                        mimetype='application/json'
                    )
                else:
                    return jsonify({'error': 'Export failed'}), 500
                    
            except Exception as e:
                self.logger.error(f"Export error: {e}")
                return jsonify({'error': 'Export failed'}), 500
        
        @self.app.route('/api/delete/<recipe_id>', methods=['DELETE'])
        def api_delete_recipe(recipe_id: str):
            """Delete recipe."""
            try:
                if self.database.delete_recipe(recipe_id):
                    return jsonify({'success': True})
                else:
                    return jsonify({'error': 'Delete failed'}), 500
            except Exception as e:
                self.logger.error(f"Delete error: {e}")
                return jsonify({'error': 'Delete failed'}), 500
        
        @self.app.route('/uploads/<filename>')
        def uploaded_file(filename: str):
            """Serve uploaded files."""
            return send_from_directory(self.app.config['UPLOAD_FOLDER'], filename)
        
        @self.app.errorhandler(413)
        def too_large(e):
            """Handle file too large error."""
            return jsonify({'error': 'File too large'}), 413
        
        @self.app.errorhandler(404)
        def not_found(e):
            """Handle 404 errors."""
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Not found'}), 404
            return render_template('404.html'), 404
        
        @self.app.errorhandler(500)
        def internal_error(e):
            """Handle 500 errors."""
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Internal server error'}), 500
            return render_template('500.html'), 500
    
    def create_templates(self):
        """Create HTML templates."""
        templates_dir = Path(__file__).parent / 'templates'
        templates_dir.mkdir(exist_ok=True)
        
        # Base template
        base_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Recipe AI{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .recipe-card { transition: transform 0.2s; }
        .recipe-card:hover { transform: translateY(-5px); }
        .ingredient-list { max-height: 300px; overflow-y: auto; }
        .confidence-badge { font-size: 0.8em; }
        .upload-area { border: 2px dashed #dee2e6; padding: 2rem; text-align: center; }
        .upload-area.dragover { border-color: #0d6efd; background-color: #f8f9fa; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                <i class="fas fa-utensils"></i> Recipe AI
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="{{ url_for('index') }}">Home</a>
                <a class="nav-link" href="{{ url_for('upload_recipe') }}">Upload</a>
                <a class="nav-link" href="{{ url_for('browse_recipes') }}">Browse</a>
                <a class="nav-link" href="{{ url_for('search_recipes') }}">Search</a>
                <a class="nav-link" href="{{ url_for('view_stats') }}">Stats</a>
            </div>
        </div>
    </nav>
    
    <div class="container mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ 'danger' if category == 'error' else 'success' }} alert-dismissible fade show">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        {% block content %}{% endblock %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
'''
        
        # Index page
        index_template = '''
{% extends "base.html" %}

{% block title %}Recipe AI - Home{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8">
        <h1>Welcome to Recipe AI</h1>
        <p class="lead">Transform your recipe images into structured, searchable data with AI-powered ingredient extraction and analysis.</p>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title"><i class="fas fa-upload"></i> Upload Recipe</h5>
                        <p class="card-text">Upload recipe images and let AI extract ingredients automatically.</p>
                        <a href="{{ url_for('upload_recipe') }}" class="btn btn-primary">Upload Now</a>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title"><i class="fas fa-search"></i> Search Recipes</h5>
                        <p class="card-text">Search through your recipe collection by ingredients or recipe names.</p>
                        <a href="{{ url_for('search_recipes') }}" class="btn btn-primary">Search Now</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-chart-bar"></i> Quick Stats</h5>
            </div>
            <div class="card-body">
                <div id="quick-stats">
                    <div class="text-center">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="row mt-5">
    <div class="col-12">
        <h3>Features</h3>
        <div class="row">
            <div class="col-md-3">
                <div class="text-center">
                    <i class="fas fa-camera fa-3x text-primary mb-3"></i>
                    <h5>Image Processing</h5>
                    <p>Advanced image preprocessing for recipe cards and cookbook pages.</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="text-center">
                    <i class="fas fa-eye fa-3x text-success mb-3"></i>
                    <h5>Text Detection</h5>
                    <p>YOLOv8-based text detection with high accuracy OCR extraction.</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="text-center">
                    <i class="fas fa-brain fa-3x text-info mb-3"></i>
                    <h5>AI Parsing</h5>
                    <p>Intelligent ingredient parsing with typo correction and normalization.</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="text-center">
                    <i class="fas fa-balance-scale fa-3x text-warning mb-3"></i>
                    <h5>Recipe Scaling</h5>
                    <p>Automatic recipe scaling with unit conversion and dietary modifications.</p>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
// Load quick stats
$(document).ready(function() {
    $.get('/api/stats', function(data) {
        $('#quick-stats').html(`
            <div class="row text-center">
                <div class="col-6">
                    <h4 class="text-primary">${data.database.recipes.total}</h4>
                    <small>Recipes</small>
                </div>
                <div class="col-6">
                    <h4 class="text-success">${data.database.ingredients.unique}</h4>
                    <small>Ingredients</small>
                </div>
            </div>
        `);
    });
});
</script>
{% endblock %}
'''
        
        # Upload page
        upload_template = '''
{% extends "base.html" %}

{% block title %}Upload Recipe{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8 mx-auto">
        <h2>Upload Recipe Image</h2>
        <p class="text-muted">Upload an image of your recipe and let AI extract the ingredients automatically.</p>
        
        <div class="card">
            <div class="card-body">
                <form id="upload-form" enctype="multipart/form-data">
                    <div class="upload-area" id="upload-area">
                        <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                        <p>Drag and drop your recipe image here, or click to select</p>
                        <input type="file" id="file-input" name="file" accept="image/*" class="d-none">
                        <button type="button" class="btn btn-outline-primary" onclick="$('#file-input').click()">
                            Choose File
                        </button>
                    </div>
                    
                    <div id="file-info" class="mt-3 d-none">
                        <div class="alert alert-info">
                            <i class="fas fa-file-image"></i> <span id="file-name"></span>
                        </div>
                    </div>
                    
                    <div class="mt-3">
                        <button type="submit" class="btn btn-primary btn-lg" id="upload-btn" disabled>
                            <i class="fas fa-upload"></i> Upload and Analyze
                        </button>
                    </div>
                </form>
                
                <div id="upload-progress" class="mt-3 d-none">
                    <div class="progress">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" 
                             role="progressbar" style="width: 100%">
                            Analyzing recipe...
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
$(document).ready(function() {
    let selectedFile = null;
    
    // Drag and drop handling
    $('#upload-area').on('dragover', function(e) {
        e.preventDefault();
        $(this).addClass('dragover');
    });
    
    $('#upload-area').on('dragleave', function(e) {
        e.preventDefault();
        $(this).removeClass('dragover');
    });
    
    $('#upload-area').on('drop', function(e) {
        e.preventDefault();
        $(this).removeClass('dragover');
        
        let files = e.originalEvent.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    $('#file-input').on('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    function handleFileSelect(file) {
        selectedFile = file;
        $('#file-name').text(file.name);
        $('#file-info').removeClass('d-none');
        $('#upload-btn').prop('disabled', false);
    }
    
    // Form submission
    $('#upload-form').on('submit', function(e) {
        e.preventDefault();
        
        if (!selectedFile) {
            alert('Please select a file first');
            return;
        }
        
        let formData = new FormData();
        formData.append('file', selectedFile);
        
        $('#upload-btn').prop('disabled', true);
        $('#upload-progress').removeClass('d-none');
        
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.success) {
                    window.location.href = response.redirect_url;
                } else {
                    alert('Upload failed: ' + response.error);
                }
            },
            error: function(xhr) {
                let error = xhr.responseJSON ? xhr.responseJSON.error : 'Upload failed';
                alert(error);
            },
            complete: function() {
                $('#upload-btn').prop('disabled', false);
                $('#upload-progress').addClass('d-none');
            }
        });
    });
});
</script>
{% endblock %}
'''
        
        # Write templates
        with open(templates_dir / 'base.html', 'w') as f:
            f.write(base_template)
        
        with open(templates_dir / 'index.html', 'w') as f:
            f.write(index_template)
        
        with open(templates_dir / 'upload.html', 'w') as f:
            f.write(upload_template)
        
        self.logger.info("Created HTML templates")
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """
        Run the Flask application.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        # Create templates if they don't exist
        templates_dir = Path(__file__).parent / 'templates'
        if not templates_dir.exists():
            self.create_templates()
        
        self.logger.info(f"Starting Recipe Web Interface on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main web interface script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recipe web interface')
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
    
    # Initialize web interface
    web_interface = RecipeWebInterface(config)
    
    # Run application
    web_interface.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()