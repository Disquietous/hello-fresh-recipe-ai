#!/usr/bin/env python3
"""
Recipe Database Management System
SQLite database schema and operations for storing processed recipes,
ingredients, and analysis results with full-text search capabilities.
"""

import sqlite3
import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import hashlib

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from complete_recipe_analyzer import RecipeAnalysisResult, RecipeIngredient, RecipeNutrition
from recipe_scaler import ScaledRecipe, ScaledIngredient
from enhanced_ingredient_parser import EnhancedIngredient


@dataclass
class RecipeRecord:
    """Database record for a recipe."""
    recipe_id: str
    title: Optional[str]
    servings: Optional[int]
    instructions: List[str]
    image_path: str
    analysis_id: str
    created_at: str
    updated_at: str
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class IngredientRecord:
    """Database record for an ingredient."""
    ingredient_id: str
    recipe_id: str
    original_text: str
    standardized_format: str
    ingredient_name: str
    quantity: Optional[str]
    unit: Optional[str]
    normalized_quantity: Optional[float]
    normalized_unit: Optional[str]
    preparation: Optional[str]
    confidence: float
    database_match: Optional[str]
    nutritional_info: Optional[Dict[str, Any]]
    scaling_factor: float


@dataclass
class NutritionRecord:
    """Database record for nutrition information."""
    nutrition_id: str
    recipe_id: str
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
class AnalysisRecord:
    """Database record for analysis metadata."""
    analysis_id: str
    recipe_id: str
    image_path: str
    image_size: str
    processing_time: float
    confidence_scores: str
    processing_steps: str
    errors: str
    warnings: str
    success: bool
    created_at: str


class RecipeDatabase:
    """Recipe database management system."""
    
    def __init__(self, db_path: str = "recipes.db", config: Optional[Dict[str, Any]] = None):
        """
        Initialize recipe database.
        
        Args:
            db_path: Path to SQLite database file
            config: Configuration dictionary
        """
        self.db_path = Path(db_path)
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize database
        self._init_database()
        
        # Enable full-text search
        self.full_text_search_enabled = self.config.get('full_text_search', True)
        
        self.logger.info(f"Initialized RecipeDatabase at {self.db_path}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for database."""
        logger = logging.getLogger('recipe_database')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create recipes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recipes (
                    recipe_id TEXT PRIMARY KEY,
                    title TEXT,
                    servings INTEGER,
                    instructions TEXT,
                    image_path TEXT NOT NULL,
                    analysis_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create ingredients table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingredients (
                    ingredient_id TEXT PRIMARY KEY,
                    recipe_id TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    standardized_format TEXT NOT NULL,
                    ingredient_name TEXT NOT NULL,
                    quantity TEXT,
                    unit TEXT,
                    normalized_quantity REAL,
                    normalized_unit TEXT,
                    preparation TEXT,
                    confidence REAL NOT NULL,
                    database_match TEXT,
                    nutritional_info TEXT,
                    scaling_factor REAL DEFAULT 1.0,
                    FOREIGN KEY (recipe_id) REFERENCES recipes (recipe_id) ON DELETE CASCADE
                )
            """)
            
            # Create nutrition table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nutrition (
                    nutrition_id TEXT PRIMARY KEY,
                    recipe_id TEXT NOT NULL,
                    total_calories REAL,
                    calories_per_serving REAL,
                    total_protein_g REAL,
                    protein_per_serving_g REAL,
                    total_carbs_g REAL,
                    carbs_per_serving_g REAL,
                    total_fat_g REAL,
                    fat_per_serving_g REAL,
                    total_fiber_g REAL,
                    fiber_per_serving_g REAL,
                    total_sodium_mg REAL,
                    sodium_per_serving_mg REAL,
                    servings INTEGER,
                    calculation_method TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    FOREIGN KEY (recipe_id) REFERENCES recipes (recipe_id) ON DELETE CASCADE
                )
            """)
            
            # Create analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis (
                    analysis_id TEXT PRIMARY KEY,
                    recipe_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    image_size TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    confidence_scores TEXT NOT NULL,
                    processing_steps TEXT NOT NULL,
                    errors TEXT NOT NULL,
                    warnings TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (recipe_id) REFERENCES recipes (recipe_id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_recipes_title ON recipes(title)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_recipes_created_at ON recipes(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ingredients_recipe_id ON ingredients(recipe_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ingredients_name ON ingredients(ingredient_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nutrition_recipe_id ON nutrition(recipe_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_recipe_id ON analysis(recipe_id)")
            
            # Create full-text search tables if enabled
            if self.full_text_search_enabled:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS recipes_fts USING fts5(
                        recipe_id,
                        title,
                        instructions,
                        content='recipes',
                        content_rowid='rowid'
                    )
                """)
                
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS ingredients_fts USING fts5(
                        ingredient_id,
                        ingredient_name,
                        original_text,
                        standardized_format,
                        content='ingredients',
                        content_rowid='rowid'
                    )
                """)
            
            conn.commit()
            self.logger.info("Database schema initialized")
    
    def store_recipe_analysis(self, result: RecipeAnalysisResult) -> str:
        """
        Store complete recipe analysis result in database.
        
        Args:
            result: Recipe analysis result
            
        Returns:
            Recipe ID
        """
        recipe_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                # Store recipe
                cursor.execute("""
                    INSERT INTO recipes (
                        recipe_id, title, servings, instructions, image_path, analysis_id,
                        created_at, updated_at, confidence_score, processing_time, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    recipe_id,
                    result.recipe_title,
                    result.servings,
                    json.dumps(result.instructions),
                    result.image_path,
                    result.analysis_id,
                    now,
                    now,
                    result.confidence_scores.get('overall', 0.0),
                    result.processing_time,
                    json.dumps(asdict(result.preprocessing_result))
                ))
                
                # Store ingredients
                for ingredient in result.analyzed_ingredients:
                    ingredient_id = str(uuid.uuid4())
                    cursor.execute("""
                        INSERT INTO ingredients (
                            ingredient_id, recipe_id, original_text, standardized_format,
                            ingredient_name, quantity, unit, normalized_quantity, normalized_unit,
                            preparation, confidence, database_match, nutritional_info, scaling_factor
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ingredient_id,
                        recipe_id,
                        ingredient.original_text,
                        ingredient.parsed_ingredient.standardized_format,
                        ingredient.parsed_ingredient.ingredient_name,
                        ingredient.parsed_ingredient.quantity,
                        ingredient.parsed_ingredient.unit,
                        ingredient.parsed_ingredient.normalized_quantity,
                        ingredient.parsed_ingredient.normalized_unit,
                        ingredient.parsed_ingredient.preparation,
                        ingredient.confidence,
                        json.dumps(ingredient.parsed_ingredient.database_match) if ingredient.parsed_ingredient.database_match else None,
                        json.dumps(ingredient.nutritional_info) if ingredient.nutritional_info else None,
                        ingredient.scaling_factor
                    ))
                
                # Store nutrition
                if result.nutritional_analysis:
                    nutrition_id = str(uuid.uuid4())
                    nutrition = result.nutritional_analysis
                    cursor.execute("""
                        INSERT INTO nutrition (
                            nutrition_id, recipe_id, total_calories, calories_per_serving,
                            total_protein_g, protein_per_serving_g, total_carbs_g, carbs_per_serving_g,
                            total_fat_g, fat_per_serving_g, total_fiber_g, fiber_per_serving_g,
                            total_sodium_mg, sodium_per_serving_mg, servings, calculation_method, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        nutrition_id,
                        recipe_id,
                        nutrition.total_calories,
                        nutrition.calories_per_serving,
                        nutrition.total_protein_g,
                        nutrition.protein_per_serving_g,
                        nutrition.total_carbs_g,
                        nutrition.carbs_per_serving_g,
                        nutrition.total_fat_g,
                        nutrition.fat_per_serving_g,
                        nutrition.total_fiber_g,
                        nutrition.fiber_per_serving_g,
                        nutrition.total_sodium_mg,
                        nutrition.sodium_per_serving_mg,
                        nutrition.servings,
                        nutrition.calculation_method,
                        nutrition.confidence
                    ))
                
                # Store analysis metadata
                cursor.execute("""
                    INSERT INTO analysis (
                        analysis_id, recipe_id, image_path, image_size, processing_time,
                        confidence_scores, processing_steps, errors, warnings, success, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.analysis_id,
                    recipe_id,
                    result.image_path,
                    f"{result.image_size[0]}x{result.image_size[1]}",
                    result.processing_time,
                    json.dumps(result.confidence_scores),
                    json.dumps(result.processing_steps),
                    json.dumps(result.errors),
                    json.dumps(result.warnings),
                    result.success,
                    now
                ))
                
                # Update full-text search if enabled
                if self.full_text_search_enabled:
                    cursor.execute("""
                        INSERT INTO recipes_fts (recipe_id, title, instructions)
                        VALUES (?, ?, ?)
                    """, (
                        recipe_id,
                        result.recipe_title or "",
                        " ".join(result.instructions)
                    ))
                    
                    for ingredient in result.analyzed_ingredients:
                        cursor.execute("""
                            INSERT INTO ingredients_fts (ingredient_id, ingredient_name, original_text, standardized_format)
                            VALUES (?, ?, ?, ?)
                        """, (
                            ingredient_id,
                            ingredient.parsed_ingredient.ingredient_name,
                            ingredient.original_text,
                            ingredient.parsed_ingredient.standardized_format
                        ))
                
                conn.commit()
                self.logger.info(f"Stored recipe analysis result: {recipe_id}")
                return recipe_id
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Failed to store recipe analysis: {e}")
                raise
    
    def store_scaled_recipe(self, scaled_recipe: ScaledRecipe, original_recipe_id: str) -> str:
        """
        Store scaled recipe as a new recipe with reference to original.
        
        Args:
            scaled_recipe: Scaled recipe result
            original_recipe_id: ID of original recipe
            
        Returns:
            New recipe ID
        """
        recipe_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                # Get original recipe title
                cursor.execute("SELECT title FROM recipes WHERE recipe_id = ?", (original_recipe_id,))
                original_title = cursor.fetchone()
                original_title = original_title[0] if original_title else "Unknown Recipe"
                
                # Create scaled recipe title
                scaled_title = f"{original_title} (Scaled {scaled_recipe.scaling_factor}x)"
                
                # Store scaled recipe
                cursor.execute("""
                    INSERT INTO recipes (
                        recipe_id, title, servings, instructions, image_path, analysis_id,
                        created_at, updated_at, confidence_score, processing_time, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    recipe_id,
                    scaled_title,
                    scaled_recipe.target_servings,
                    json.dumps(scaled_recipe.scaled_instructions),
                    f"scaled_from_{original_recipe_id}",
                    f"scaled_{uuid.uuid4()}",
                    now,
                    now,
                    1.0,  # Scaled recipes inherit full confidence
                    0.0,  # No processing time for scaling
                    json.dumps({
                        "original_recipe_id": original_recipe_id,
                        "scaling_factor": scaled_recipe.scaling_factor,
                        "scaling_options": asdict(scaled_recipe.scaling_options),
                        "scaling_notes": scaled_recipe.scaling_notes
                    })
                ))
                
                # Store scaled ingredients
                for ingredient in scaled_recipe.scaled_ingredients:
                    ingredient_id = str(uuid.uuid4())
                    cursor.execute("""
                        INSERT INTO ingredients (
                            ingredient_id, recipe_id, original_text, standardized_format,
                            ingredient_name, quantity, unit, normalized_quantity, normalized_unit,
                            preparation, confidence, database_match, nutritional_info, scaling_factor
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ingredient_id,
                        recipe_id,
                        ingredient.original_ingredient.original_text,
                        f"{ingredient.display_quantity} {ingredient.display_unit} {ingredient.original_ingredient.ingredient_name}",
                        ingredient.original_ingredient.ingredient_name,
                        ingredient.display_quantity,
                        ingredient.display_unit,
                        ingredient.scaled_quantity,
                        ingredient.scaled_unit,
                        ingredient.original_ingredient.preparation,
                        ingredient.original_ingredient.confidence,
                        json.dumps(ingredient.original_ingredient.database_match) if ingredient.original_ingredient.database_match else None,
                        json.dumps(ingredient.original_ingredient.nutritional_info) if ingredient.original_ingredient.nutritional_info else None,
                        ingredient.scaling_factor
                    ))
                
                # Store scaled nutrition
                if scaled_recipe.nutritional_scaling:
                    nutrition_id = str(uuid.uuid4())
                    nutrition = scaled_recipe.nutritional_scaling
                    cursor.execute("""
                        INSERT INTO nutrition (
                            nutrition_id, recipe_id, total_calories, calories_per_serving,
                            total_protein_g, protein_per_serving_g, total_carbs_g, carbs_per_serving_g,
                            total_fat_g, fat_per_serving_g, total_fiber_g, fiber_per_serving_g,
                            total_sodium_mg, sodium_per_serving_mg, servings, calculation_method, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        nutrition_id,
                        recipe_id,
                        nutrition.get('total_calories'),
                        nutrition.get('total_calories') / scaled_recipe.target_servings if scaled_recipe.target_servings else None,
                        nutrition.get('total_protein_g'),
                        nutrition.get('total_protein_g') / scaled_recipe.target_servings if scaled_recipe.target_servings else None,
                        nutrition.get('total_carbs_g'),
                        nutrition.get('total_carbs_g') / scaled_recipe.target_servings if scaled_recipe.target_servings else None,
                        nutrition.get('total_fat_g'),
                        nutrition.get('total_fat_g') / scaled_recipe.target_servings if scaled_recipe.target_servings else None,
                        None,  # fiber not in scaling
                        None,
                        None,  # sodium not in scaling
                        None,
                        scaled_recipe.target_servings,
                        "scaled_calculation",
                        nutrition.get('nutrition_coverage', 1.0)
                    ))
                
                conn.commit()
                self.logger.info(f"Stored scaled recipe: {recipe_id}")
                return recipe_id
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Failed to store scaled recipe: {e}")
                raise
    
    def get_recipe(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete recipe data by ID.
        
        Args:
            recipe_id: Recipe ID
            
        Returns:
            Recipe data dictionary or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get recipe
            cursor.execute("SELECT * FROM recipes WHERE recipe_id = ?", (recipe_id,))
            recipe = cursor.fetchone()
            
            if not recipe:
                return None
            
            # Get ingredients
            cursor.execute("SELECT * FROM ingredients WHERE recipe_id = ?", (recipe_id,))
            ingredients = cursor.fetchall()
            
            # Get nutrition
            cursor.execute("SELECT * FROM nutrition WHERE recipe_id = ?", (recipe_id,))
            nutrition = cursor.fetchone()
            
            # Get analysis
            cursor.execute("SELECT * FROM analysis WHERE recipe_id = ?", (recipe_id,))
            analysis = cursor.fetchone()
            
            return {
                "recipe": dict(recipe),
                "ingredients": [dict(ing) for ing in ingredients],
                "nutrition": dict(nutrition) if nutrition else None,
                "analysis": dict(analysis) if analysis else None
            }
    
    def search_recipes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search recipes using full-text search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of recipe search results
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if self.full_text_search_enabled:
                # Use full-text search
                cursor.execute("""
                    SELECT r.*, rank
                    FROM recipes_fts
                    JOIN recipes r ON recipes_fts.recipe_id = r.recipe_id
                    WHERE recipes_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, limit))
            else:
                # Use basic LIKE search
                cursor.execute("""
                    SELECT *
                    FROM recipes
                    WHERE title LIKE ? OR instructions LIKE ?
                    ORDER BY title
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def search_ingredients(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search ingredients using full-text search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of ingredient search results
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if self.full_text_search_enabled:
                # Use full-text search
                cursor.execute("""
                    SELECT i.*, r.title as recipe_title
                    FROM ingredients_fts
                    JOIN ingredients i ON ingredients_fts.ingredient_id = i.ingredient_id
                    JOIN recipes r ON i.recipe_id = r.recipe_id
                    WHERE ingredients_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, limit))
            else:
                # Use basic LIKE search
                cursor.execute("""
                    SELECT i.*, r.title as recipe_title
                    FROM ingredients i
                    JOIN recipes r ON i.recipe_id = r.recipe_id
                    WHERE i.ingredient_name LIKE ? OR i.original_text LIKE ?
                    ORDER BY i.ingredient_name
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_recipes_by_ingredient(self, ingredient_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recipes containing a specific ingredient.
        
        Args:
            ingredient_name: Ingredient name to search for
            limit: Maximum number of results
            
        Returns:
            List of recipes containing the ingredient
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT r.*, i.ingredient_name, i.standardized_format
                FROM recipes r
                JOIN ingredients i ON r.recipe_id = i.recipe_id
                WHERE i.ingredient_name LIKE ?
                ORDER BY r.title
                LIMIT ?
            """, (f"%{ingredient_name}%", limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_nutrition_stats(self) -> Dict[str, Any]:
        """
        Get nutrition statistics across all recipes.
        
        Returns:
            Nutrition statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_recipes,
                    AVG(calories_per_serving) as avg_calories_per_serving,
                    AVG(protein_per_serving_g) as avg_protein_per_serving,
                    AVG(carbs_per_serving_g) as avg_carbs_per_serving,
                    AVG(fat_per_serving_g) as avg_fat_per_serving,
                    MIN(calories_per_serving) as min_calories_per_serving,
                    MAX(calories_per_serving) as max_calories_per_serving
                FROM nutrition
                WHERE calories_per_serving IS NOT NULL
            """)
            
            result = cursor.fetchone()
            
            return {
                "total_recipes": result[0],
                "avg_calories_per_serving": result[1],
                "avg_protein_per_serving": result[2],
                "avg_carbs_per_serving": result[3],
                "avg_fat_per_serving": result[4],
                "min_calories_per_serving": result[5],
                "max_calories_per_serving": result[6]
            }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.
        
        Returns:
            Database statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Recipe stats
            cursor.execute("SELECT COUNT(*) FROM recipes")
            total_recipes = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM recipes WHERE title IS NOT NULL")
            recipes_with_titles = cursor.fetchone()[0]
            
            # Ingredient stats
            cursor.execute("SELECT COUNT(*) FROM ingredients")
            total_ingredients = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT ingredient_name) FROM ingredients")
            unique_ingredients = cursor.fetchone()[0]
            
            # Nutrition stats
            cursor.execute("SELECT COUNT(*) FROM nutrition")
            recipes_with_nutrition = cursor.fetchone()[0]
            
            # Analysis stats
            cursor.execute("SELECT COUNT(*) FROM analysis WHERE success = 1")
            successful_analyses = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(processing_time) FROM analysis")
            avg_processing_time = cursor.fetchone()[0]
            
            return {
                "recipes": {
                    "total": total_recipes,
                    "with_titles": recipes_with_titles,
                    "title_coverage": recipes_with_titles / total_recipes if total_recipes > 0 else 0
                },
                "ingredients": {
                    "total": total_ingredients,
                    "unique": unique_ingredients,
                    "avg_per_recipe": total_ingredients / total_recipes if total_recipes > 0 else 0
                },
                "nutrition": {
                    "recipes_with_nutrition": recipes_with_nutrition,
                    "nutrition_coverage": recipes_with_nutrition / total_recipes if total_recipes > 0 else 0
                },
                "analysis": {
                    "successful": successful_analyses,
                    "success_rate": successful_analyses / total_recipes if total_recipes > 0 else 0,
                    "avg_processing_time": avg_processing_time or 0
                }
            }
    
    def delete_recipe(self, recipe_id: str) -> bool:
        """
        Delete recipe and all associated data.
        
        Args:
            recipe_id: Recipe ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                # Delete recipe (cascades to ingredients, nutrition, analysis)
                cursor.execute("DELETE FROM recipes WHERE recipe_id = ?", (recipe_id,))
                
                # Delete from FTS tables if enabled
                if self.full_text_search_enabled:
                    cursor.execute("DELETE FROM recipes_fts WHERE recipe_id = ?", (recipe_id,))
                    cursor.execute("DELETE FROM ingredients_fts WHERE ingredient_id IN (SELECT ingredient_id FROM ingredients WHERE recipe_id = ?)", (recipe_id,))
                
                conn.commit()
                self.logger.info(f"Deleted recipe: {recipe_id}")
                return True
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Failed to delete recipe: {e}")
                return False
    
    def export_recipes(self, output_file: str, format: str = "json") -> bool:
        """
        Export all recipes to file.
        
        Args:
            output_file: Output file path
            format: Export format (json, csv)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if format == "json":
                recipes = []
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    cursor.execute("SELECT recipe_id FROM recipes")
                    recipe_ids = [row[0] for row in cursor.fetchall()]
                    
                    for recipe_id in recipe_ids:
                        recipe_data = self.get_recipe(recipe_id)
                        if recipe_data:
                            recipes.append(recipe_data)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(recipes, f, indent=2, ensure_ascii=False, default=str)
                
                self.logger.info(f"Exported {len(recipes)} recipes to {output_file}")
                return True
                
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to export recipes: {e}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create database backup.
        
        Args:
            backup_path: Backup file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return False


def main():
    """Main database management script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recipe database management')
    parser.add_argument('--db', default='recipes.db', help='Database file path')
    parser.add_argument('--action', required=True, choices=['init', 'stats', 'search', 'export', 'backup'], help='Action to perform')
    parser.add_argument('--query', help='Search query')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', default='json', choices=['json', 'csv'], help='Export format')
    
    args = parser.parse_args()
    
    # Initialize database
    db = RecipeDatabase(args.db)
    
    if args.action == 'init':
        print("Database initialized successfully")
        
    elif args.action == 'stats':
        stats = db.get_database_stats()
        print("Database Statistics:")
        print(f"  Total recipes: {stats['recipes']['total']}")
        print(f"  Recipes with titles: {stats['recipes']['with_titles']}")
        print(f"  Total ingredients: {stats['ingredients']['total']}")
        print(f"  Unique ingredients: {stats['ingredients']['unique']}")
        print(f"  Recipes with nutrition: {stats['nutrition']['recipes_with_nutrition']}")
        print(f"  Successful analyses: {stats['analysis']['successful']}")
        print(f"  Average processing time: {stats['analysis']['avg_processing_time']:.2f}s")
        
    elif args.action == 'search':
        if not args.query:
            print("Search query required")
            return 1
        
        results = db.search_recipes(args.query)
        print(f"Found {len(results)} recipes:")
        for recipe in results:
            print(f"  - {recipe['title']} ({recipe['recipe_id']})")
        
    elif args.action == 'export':
        if not args.output:
            print("Output file required")
            return 1
        
        if db.export_recipes(args.output, args.format):
            print(f"Recipes exported to {args.output}")
        else:
            print("Export failed")
            return 1
            
    elif args.action == 'backup':
        if not args.output:
            print("Backup path required")
            return 1
        
        if db.backup_database(args.output):
            print(f"Database backed up to {args.output}")
        else:
            print("Backup failed")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())