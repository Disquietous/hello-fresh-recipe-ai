#!/usr/bin/env python3
"""
Nutritional Analysis System
Advanced nutritional analysis with multiple data sources, dietary tracking,
and comprehensive nutritional insights for recipe ingredients.
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import time
from datetime import datetime
import sqlite3
import pickle
from functools import lru_cache

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from enhanced_ingredient_parser import EnhancedIngredient


@dataclass
class NutrientInfo:
    """Individual nutrient information."""
    name: str
    amount: float
    unit: str
    daily_value_percentage: Optional[float] = None
    category: str = "other"  # macronutrient, vitamin, mineral, other


@dataclass
class NutritionalProfile:
    """Comprehensive nutritional profile."""
    ingredient_id: str
    ingredient_name: str
    serving_size: str
    serving_weight_g: float
    
    # Macronutrients
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    sugar_g: float
    
    # Detailed nutrients
    nutrients: List[NutrientInfo]
    
    # Metadata
    data_source: str
    confidence: float
    last_updated: str
    
    # Dietary flags
    dietary_flags: Dict[str, bool]  # vegetarian, vegan, gluten_free, etc.


@dataclass
class RecipeNutritionalAnalysis:
    """Complete recipe nutritional analysis."""
    recipe_id: str
    total_servings: int
    
    # Per recipe totals
    total_calories: float
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    total_fiber_g: float
    total_sugar_g: float
    
    # Per serving
    calories_per_serving: float
    protein_per_serving_g: float
    carbs_per_serving_g: float
    fat_per_serving_g: float
    fiber_per_serving_g: float
    sugar_per_serving_g: float
    
    # Detailed analysis
    ingredient_profiles: List[NutritionalProfile]
    nutrient_breakdown: Dict[str, float]
    dietary_analysis: Dict[str, Any]
    
    # Quality metrics
    data_coverage: float  # Percentage of ingredients with nutritional data
    analysis_confidence: float
    analysis_timestamp: str


class NutritionalAnalyzer:
    """Advanced nutritional analysis system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize nutritional analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # API configuration
        self.usda_api_key = self.config.get('usda_api_key')
        self.spoonacular_api_key = self.config.get('spoonacular_api_key')
        self.edamam_app_id = self.config.get('edamam_app_id')
        self.edamam_app_key = self.config.get('edamam_app_key')
        
        # Cache configuration
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_duration = self.config.get('cache_duration', 86400)  # 24 hours
        self.cache_db_path = self.config.get('cache_db_path', 'nutrition_cache.db')
        
        # Initialize cache
        if self.cache_enabled:
            self._init_cache()
        
        # Nutritional database
        self.nutrient_database = self._load_nutrient_database()
        
        # Dietary analysis rules
        self.dietary_rules = self._load_dietary_rules()
        
        # Request session for API calls
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HelloFresh-Recipe-AI/1.0'
        })
        
        self.logger.info("Initialized NutritionalAnalyzer")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for nutritional analyzer."""
        logger = logging.getLogger('nutritional_analyzer')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _init_cache(self):
        """Initialize nutritional data cache."""
        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nutrition_cache (
                    ingredient_key TEXT PRIMARY KEY,
                    nutritional_data TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    source TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_timestamp 
                ON nutrition_cache(timestamp)
            """)
            conn.commit()
    
    def _load_nutrient_database(self) -> Dict[str, Any]:
        """Load local nutrient database."""
        db_path = Path(__file__).parent / 'data' / 'nutrient_database.json'
        
        if db_path.exists():
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load nutrient database: {e}")
        
        # Return minimal database
        return {
            "common_ingredients": {},
            "unit_conversions": {},
            "nutrient_references": {}
        }
    
    def _load_dietary_rules(self) -> Dict[str, Any]:
        """Load dietary analysis rules."""
        return {
            "vegetarian": {
                "forbidden_ingredients": [
                    "meat", "beef", "pork", "chicken", "fish", "seafood", "gelatin"
                ],
                "allowed_ingredients": [],
                "confidence_threshold": 0.8
            },
            "vegan": {
                "forbidden_ingredients": [
                    "meat", "beef", "pork", "chicken", "fish", "seafood", "dairy",
                    "milk", "cheese", "butter", "eggs", "honey", "gelatin"
                ],
                "allowed_ingredients": [],
                "confidence_threshold": 0.8
            },
            "gluten_free": {
                "forbidden_ingredients": [
                    "wheat", "flour", "bread", "pasta", "barley", "rye", "oats"
                ],
                "allowed_ingredients": ["rice", "corn", "quinoa"],
                "confidence_threshold": 0.9
            },
            "keto": {
                "max_carbs_per_serving": 20,
                "min_fat_percentage": 70,
                "max_carb_percentage": 10,
                "confidence_threshold": 0.7
            },
            "low_sodium": {
                "max_sodium_per_serving": 600,  # mg
                "confidence_threshold": 0.8
            },
            "high_protein": {
                "min_protein_per_serving": 20,  # g
                "confidence_threshold": 0.8
            }
        }
    
    @lru_cache(maxsize=1000)
    def _get_cached_nutrition(self, ingredient_key: str) -> Optional[NutritionalProfile]:
        """Get cached nutritional data."""
        if not self.cache_enabled:
            return None
        
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT nutritional_data, timestamp, source 
                    FROM nutrition_cache 
                    WHERE ingredient_key = ?
                """, (ingredient_key,))
                
                result = cursor.fetchone()
                if result:
                    data, timestamp, source = result
                    
                    # Check if cache is still valid
                    if time.time() - timestamp < self.cache_duration:
                        try:
                            profile_data = json.loads(data)
                            return NutritionalProfile(**profile_data)
                        except Exception as e:
                            self.logger.warning(f"Invalid cached data: {e}")
                
                return None
                
        except Exception as e:
            self.logger.warning(f"Cache lookup failed: {e}")
            return None
    
    def _cache_nutrition(self, ingredient_key: str, profile: NutritionalProfile, source: str):
        """Cache nutritional data."""
        if not self.cache_enabled:
            return
        
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO nutrition_cache 
                    (ingredient_key, nutritional_data, timestamp, source)
                    VALUES (?, ?, ?, ?)
                """, (
                    ingredient_key,
                    json.dumps(asdict(profile), default=str),
                    time.time(),
                    source
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")
    
    def analyze_ingredient(self, ingredient: EnhancedIngredient) -> Optional[NutritionalProfile]:
        """
        Analyze nutritional content of a single ingredient.
        
        Args:
            ingredient: Enhanced ingredient to analyze
            
        Returns:
            Nutritional profile or None
        """
        # Create cache key
        ingredient_key = f"{ingredient.ingredient_name}_{ingredient.normalized_quantity}_{ingredient.normalized_unit}"
        
        # Check cache first
        cached_profile = self._get_cached_nutrition(ingredient_key)
        if cached_profile:
            return cached_profile
        
        # Try multiple data sources
        profile = None
        
        # 1. Try local database
        profile = self._get_nutrition_from_local_db(ingredient)
        if profile:
            self._cache_nutrition(ingredient_key, profile, "local")
            return profile
        
        # 2. Try USDA API
        if self.usda_api_key:
            profile = self._get_nutrition_from_usda(ingredient)
            if profile:
                self._cache_nutrition(ingredient_key, profile, "usda")
                return profile
        
        # 3. Try Spoonacular API
        if self.spoonacular_api_key:
            profile = self._get_nutrition_from_spoonacular(ingredient)
            if profile:
                self._cache_nutrition(ingredient_key, profile, "spoonacular")
                return profile
        
        # 4. Try Edamam API
        if self.edamam_app_id and self.edamam_app_key:
            profile = self._get_nutrition_from_edamam(ingredient)
            if profile:
                self._cache_nutrition(ingredient_key, profile, "edamam")
                return profile
        
        # 5. Use estimation if no data found
        profile = self._estimate_nutrition(ingredient)
        if profile:
            self._cache_nutrition(ingredient_key, profile, "estimated")
            return profile
        
        return None
    
    def _get_nutrition_from_local_db(self, ingredient: EnhancedIngredient) -> Optional[NutritionalProfile]:
        """Get nutrition from local database."""
        ingredient_name = ingredient.ingredient_name.lower()
        
        # Look for exact match
        if ingredient_name in self.nutrient_database.get("common_ingredients", {}):
            data = self.nutrient_database["common_ingredients"][ingredient_name]
            return self._create_nutritional_profile(ingredient, data, "local", 0.9)
        
        # Look for partial matches
        for db_ingredient, data in self.nutrient_database.get("common_ingredients", {}).items():
            if ingredient_name in db_ingredient or db_ingredient in ingredient_name:
                return self._create_nutritional_profile(ingredient, data, "local", 0.7)
        
        return None
    
    def _get_nutrition_from_usda(self, ingredient: EnhancedIngredient) -> Optional[NutritionalProfile]:
        """Get nutrition from USDA FoodData Central API."""
        try:
            # Search for food
            search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
            search_params = {
                "query": ingredient.ingredient_name,
                "api_key": self.usda_api_key,
                "pageSize": 1
            }
            
            response = self.session.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            
            search_data = response.json()
            
            if search_data.get("foods"):
                food = search_data["foods"][0]
                food_id = food["fdcId"]
                
                # Get detailed nutrition
                detail_url = f"https://api.nal.usda.gov/fdc/v1/food/{food_id}"
                detail_params = {"api_key": self.usda_api_key}
                
                detail_response = self.session.get(detail_url, params=detail_params, timeout=10)
                detail_response.raise_for_status()
                
                detail_data = detail_response.json()
                
                return self._parse_usda_nutrition(ingredient, detail_data)
            
        except Exception as e:
            self.logger.warning(f"USDA API error: {e}")
        
        return None
    
    def _get_nutrition_from_spoonacular(self, ingredient: EnhancedIngredient) -> Optional[NutritionalProfile]:
        """Get nutrition from Spoonacular API."""
        try:
            url = "https://api.spoonacular.com/food/ingredients/search"
            params = {
                "query": ingredient.ingredient_name,
                "apiKey": self.spoonacular_api_key,
                "number": 1
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("results"):
                ingredient_data = data["results"][0]
                ingredient_id = ingredient_data["id"]
                
                # Get detailed nutrition
                nutrition_url = f"https://api.spoonacular.com/food/ingredients/{ingredient_id}/information"
                nutrition_params = {
                    "apiKey": self.spoonacular_api_key,
                    "amount": ingredient.normalized_quantity or 1,
                    "unit": ingredient.normalized_unit or "serving"
                }
                
                nutrition_response = self.session.get(nutrition_url, params=nutrition_params, timeout=10)
                nutrition_response.raise_for_status()
                
                nutrition_data = nutrition_response.json()
                
                return self._parse_spoonacular_nutrition(ingredient, nutrition_data)
            
        except Exception as e:
            self.logger.warning(f"Spoonacular API error: {e}")
        
        return None
    
    def _get_nutrition_from_edamam(self, ingredient: EnhancedIngredient) -> Optional[NutritionalProfile]:
        """Get nutrition from Edamam API."""
        try:
            url = "https://api.edamam.com/api/nutrition-data"
            params = {
                "app_id": self.edamam_app_id,
                "app_key": self.edamam_app_key,
                "ingr": f"{ingredient.normalized_quantity or 1} {ingredient.normalized_unit or 'serving'} {ingredient.ingredient_name}"
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("calories"):
                return self._parse_edamam_nutrition(ingredient, data)
            
        except Exception as e:
            self.logger.warning(f"Edamam API error: {e}")
        
        return None
    
    def _estimate_nutrition(self, ingredient: EnhancedIngredient) -> Optional[NutritionalProfile]:
        """Estimate nutrition based on ingredient category."""
        # Basic estimation based on ingredient type
        ingredient_name = ingredient.ingredient_name.lower()
        
        # Estimation rules
        if any(word in ingredient_name for word in ["flour", "bread", "pasta", "rice"]):
            # Carbohydrate-rich foods
            calories = 350
            protein = 10
            carbs = 70
            fat = 2
            fiber = 3
        elif any(word in ingredient_name for word in ["meat", "chicken", "beef", "fish"]):
            # Protein-rich foods
            calories = 200
            protein = 25
            carbs = 0
            fat = 10
            fiber = 0
        elif any(word in ingredient_name for word in ["oil", "butter", "fat"]):
            # Fat-rich foods
            calories = 800
            protein = 0
            carbs = 0
            fat = 90
            fiber = 0
        elif any(word in ingredient_name for word in ["vegetable", "fruit", "tomato", "onion"]):
            # Vegetables and fruits
            calories = 50
            protein = 2
            carbs = 10
            fat = 0.5
            fiber = 3
        else:
            # Default estimation
            calories = 150
            protein = 5
            carbs = 20
            fat = 3
            fiber = 2
        
        # Scale by quantity
        quantity = ingredient.normalized_quantity or 1
        scale_factor = quantity * 0.01  # Assume 100g base
        
        return NutritionalProfile(
            ingredient_id=str(hash(ingredient.ingredient_name)),
            ingredient_name=ingredient.ingredient_name,
            serving_size=f"{quantity} {ingredient.normalized_unit or 'serving'}",
            serving_weight_g=quantity * 100 if ingredient.normalized_unit != "gram" else quantity,
            calories=calories * scale_factor,
            protein_g=protein * scale_factor,
            carbs_g=carbs * scale_factor,
            fat_g=fat * scale_factor,
            fiber_g=fiber * scale_factor,
            sugar_g=carbs * 0.1 * scale_factor,
            nutrients=[],
            data_source="estimated",
            confidence=0.3,
            last_updated=datetime.now().isoformat(),
            dietary_flags={}
        )
    
    def _create_nutritional_profile(self, ingredient: EnhancedIngredient, data: Dict, source: str, confidence: float) -> NutritionalProfile:
        """Create nutritional profile from data."""
        return NutritionalProfile(
            ingredient_id=str(hash(ingredient.ingredient_name)),
            ingredient_name=ingredient.ingredient_name,
            serving_size=data.get("serving_size", "100g"),
            serving_weight_g=data.get("serving_weight_g", 100),
            calories=data.get("calories", 0),
            protein_g=data.get("protein_g", 0),
            carbs_g=data.get("carbs_g", 0),
            fat_g=data.get("fat_g", 0),
            fiber_g=data.get("fiber_g", 0),
            sugar_g=data.get("sugar_g", 0),
            nutrients=[],
            data_source=source,
            confidence=confidence,
            last_updated=datetime.now().isoformat(),
            dietary_flags=data.get("dietary_flags", {})
        )
    
    def _parse_usda_nutrition(self, ingredient: EnhancedIngredient, data: Dict) -> Optional[NutritionalProfile]:
        """Parse USDA nutrition data."""
        try:
            nutrients = {}
            nutrient_list = []
            
            for nutrient in data.get("foodNutrients", []):
                name = nutrient.get("nutrient", {}).get("name", "")
                amount = nutrient.get("amount", 0)
                unit = nutrient.get("nutrient", {}).get("unitName", "")
                
                if name and amount:
                    nutrients[name.lower()] = amount
                    nutrient_list.append(NutrientInfo(
                        name=name,
                        amount=amount,
                        unit=unit
                    ))
            
            return NutritionalProfile(
                ingredient_id=str(data.get("fdcId", "")),
                ingredient_name=ingredient.ingredient_name,
                serving_size="100g",
                serving_weight_g=100,
                calories=nutrients.get("energy", 0),
                protein_g=nutrients.get("protein", 0),
                carbs_g=nutrients.get("carbohydrate, by difference", 0),
                fat_g=nutrients.get("total lipid (fat)", 0),
                fiber_g=nutrients.get("fiber, total dietary", 0),
                sugar_g=nutrients.get("sugars, total including nlea", 0),
                nutrients=nutrient_list,
                data_source="usda",
                confidence=0.9,
                last_updated=datetime.now().isoformat(),
                dietary_flags={}
            )
            
        except Exception as e:
            self.logger.warning(f"USDA parsing error: {e}")
            return None
    
    def _parse_spoonacular_nutrition(self, ingredient: EnhancedIngredient, data: Dict) -> Optional[NutritionalProfile]:
        """Parse Spoonacular nutrition data."""
        try:
            nutrition = data.get("nutrition", {})
            
            return NutritionalProfile(
                ingredient_id=str(data.get("id", "")),
                ingredient_name=ingredient.ingredient_name,
                serving_size=f"{data.get('amount', 1)} {data.get('unit', 'serving')}",
                serving_weight_g=data.get("amount", 1) * 100,  # Estimate
                calories=nutrition.get("calories", 0),
                protein_g=nutrition.get("protein", 0),
                carbs_g=nutrition.get("carbs", 0),
                fat_g=nutrition.get("fat", 0),
                fiber_g=nutrition.get("fiber", 0),
                sugar_g=nutrition.get("sugar", 0),
                nutrients=[],
                data_source="spoonacular",
                confidence=0.8,
                last_updated=datetime.now().isoformat(),
                dietary_flags={}
            )
            
        except Exception as e:
            self.logger.warning(f"Spoonacular parsing error: {e}")
            return None
    
    def _parse_edamam_nutrition(self, ingredient: EnhancedIngredient, data: Dict) -> Optional[NutritionalProfile]:
        """Parse Edamam nutrition data."""
        try:
            return NutritionalProfile(
                ingredient_id=str(hash(ingredient.ingredient_name)),
                ingredient_name=ingredient.ingredient_name,
                serving_size=data.get("servingSize", "1 serving"),
                serving_weight_g=data.get("totalWeight", 100),
                calories=data.get("calories", 0),
                protein_g=data.get("totalNutrients", {}).get("PROCNT", {}).get("quantity", 0),
                carbs_g=data.get("totalNutrients", {}).get("CHOCDF", {}).get("quantity", 0),
                fat_g=data.get("totalNutrients", {}).get("FAT", {}).get("quantity", 0),
                fiber_g=data.get("totalNutrients", {}).get("FIBTG", {}).get("quantity", 0),
                sugar_g=data.get("totalNutrients", {}).get("SUGAR", {}).get("quantity", 0),
                nutrients=[],
                data_source="edamam",
                confidence=0.8,
                last_updated=datetime.now().isoformat(),
                dietary_flags={}
            )
            
        except Exception as e:
            self.logger.warning(f"Edamam parsing error: {e}")
            return None
    
    def analyze_recipe(self, ingredients: List[EnhancedIngredient], servings: int = 1) -> RecipeNutritionalAnalysis:
        """
        Analyze complete recipe nutrition.
        
        Args:
            ingredients: List of recipe ingredients
            servings: Number of servings
            
        Returns:
            Complete recipe nutritional analysis
        """
        ingredient_profiles = []
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0
        total_fiber = 0
        total_sugar = 0
        
        analyzed_ingredients = 0
        
        # Analyze each ingredient
        for ingredient in ingredients:
            profile = self.analyze_ingredient(ingredient)
            if profile:
                ingredient_profiles.append(profile)
                analyzed_ingredients += 1
                
                # Scale by ingredient quantity
                scale_factor = 1.0
                if ingredient.normalized_quantity:
                    scale_factor = ingredient.normalized_quantity / 100  # Assume 100g base
                
                total_calories += profile.calories * scale_factor
                total_protein += profile.protein_g * scale_factor
                total_carbs += profile.carbs_g * scale_factor
                total_fat += profile.fat_g * scale_factor
                total_fiber += profile.fiber_g * scale_factor
                total_sugar += profile.sugar_g * scale_factor
        
        # Calculate per serving values
        servings = max(servings, 1)
        calories_per_serving = total_calories / servings
        protein_per_serving = total_protein / servings
        carbs_per_serving = total_carbs / servings
        fat_per_serving = total_fat / servings
        fiber_per_serving = total_fiber / servings
        sugar_per_serving = total_sugar / servings
        
        # Calculate data coverage and confidence
        data_coverage = analyzed_ingredients / len(ingredients) if ingredients else 0
        analysis_confidence = sum(p.confidence for p in ingredient_profiles) / len(ingredient_profiles) if ingredient_profiles else 0
        
        # Nutrient breakdown
        nutrient_breakdown = {
            "calories": total_calories,
            "protein": total_protein,
            "carbs": total_carbs,
            "fat": total_fat,
            "fiber": total_fiber,
            "sugar": total_sugar
        }
        
        # Dietary analysis
        dietary_analysis = self._analyze_dietary_compatibility(ingredient_profiles, nutrient_breakdown, servings)
        
        return RecipeNutritionalAnalysis(
            recipe_id=str(hash(str(ingredients))),
            total_servings=servings,
            total_calories=total_calories,
            total_protein_g=total_protein,
            total_carbs_g=total_carbs,
            total_fat_g=total_fat,
            total_fiber_g=total_fiber,
            total_sugar_g=total_sugar,
            calories_per_serving=calories_per_serving,
            protein_per_serving_g=protein_per_serving,
            carbs_per_serving_g=carbs_per_serving,
            fat_per_serving_g=fat_per_serving,
            fiber_per_serving_g=fiber_per_serving,
            sugar_per_serving_g=sugar_per_serving,
            ingredient_profiles=ingredient_profiles,
            nutrient_breakdown=nutrient_breakdown,
            dietary_analysis=dietary_analysis,
            data_coverage=data_coverage,
            analysis_confidence=analysis_confidence,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _analyze_dietary_compatibility(self, profiles: List[NutritionalProfile], nutrients: Dict[str, float], servings: int) -> Dict[str, Any]:
        """Analyze dietary compatibility and restrictions."""
        analysis = {}
        
        # Check each dietary restriction
        for diet_type, rules in self.dietary_rules.items():
            if diet_type in ["vegetarian", "vegan", "gluten_free"]:
                # Ingredient-based analysis
                forbidden = rules.get("forbidden_ingredients", [])
                compatible = True
                confidence = 1.0
                
                for profile in profiles:
                    ingredient_name = profile.ingredient_name.lower()
                    for forbidden_item in forbidden:
                        if forbidden_item in ingredient_name:
                            compatible = False
                            confidence = 0.0
                            break
                    if not compatible:
                        break
                
                analysis[diet_type] = {
                    "compatible": compatible,
                    "confidence": confidence,
                    "reason": "ingredient_analysis"
                }
            
            elif diet_type == "keto":
                # Macro-based analysis
                calories_per_serving = nutrients["calories"] / servings
                carbs_per_serving = nutrients["carbs"] / servings
                fat_per_serving = nutrients["fat"] / servings
                
                carb_percentage = (carbs_per_serving * 4) / calories_per_serving * 100 if calories_per_serving > 0 else 0
                fat_percentage = (fat_per_serving * 9) / calories_per_serving * 100 if calories_per_serving > 0 else 0
                
                compatible = (
                    carbs_per_serving <= rules["max_carbs_per_serving"] and
                    carb_percentage <= rules["max_carb_percentage"] and
                    fat_percentage >= rules["min_fat_percentage"]
                )
                
                analysis[diet_type] = {
                    "compatible": compatible,
                    "confidence": rules["confidence_threshold"],
                    "carbs_per_serving": carbs_per_serving,
                    "carb_percentage": carb_percentage,
                    "fat_percentage": fat_percentage,
                    "reason": "macro_analysis"
                }
            
            elif diet_type == "low_sodium":
                # Sodium-based analysis (simplified - would need detailed nutrient data)
                # For now, use estimation
                estimated_sodium = nutrients["calories"] * 0.5  # Rough estimate
                sodium_per_serving = estimated_sodium / servings
                
                compatible = sodium_per_serving <= rules["max_sodium_per_serving"]
                
                analysis[diet_type] = {
                    "compatible": compatible,
                    "confidence": 0.5,  # Low confidence due to estimation
                    "sodium_per_serving": sodium_per_serving,
                    "reason": "estimated_analysis"
                }
            
            elif diet_type == "high_protein":
                # Protein-based analysis
                protein_per_serving = nutrients["protein"] / servings
                compatible = protein_per_serving >= rules["min_protein_per_serving"]
                
                analysis[diet_type] = {
                    "compatible": compatible,
                    "confidence": rules["confidence_threshold"],
                    "protein_per_serving": protein_per_serving,
                    "reason": "protein_analysis"
                }
        
        return analysis
    
    def generate_nutrition_report(self, analysis: RecipeNutritionalAnalysis) -> str:
        """Generate human-readable nutrition report."""
        report = []
        
        report.append("RECIPE NUTRITIONAL ANALYSIS")
        report.append("=" * 50)
        report.append(f"Total Servings: {analysis.total_servings}")
        report.append(f"Data Coverage: {analysis.data_coverage:.1%}")
        report.append(f"Analysis Confidence: {analysis.analysis_confidence:.1%}")
        report.append("")
        
        # Per serving nutrition
        report.append("NUTRITION PER SERVING:")
        report.append("-" * 25)
        report.append(f"Calories: {analysis.calories_per_serving:.0f}")
        report.append(f"Protein: {analysis.protein_per_serving_g:.1f}g")
        report.append(f"Carbohydrates: {analysis.carbs_per_serving_g:.1f}g")
        report.append(f"Fat: {analysis.fat_per_serving_g:.1f}g")
        report.append(f"Fiber: {analysis.fiber_per_serving_g:.1f}g")
        report.append(f"Sugar: {analysis.sugar_per_serving_g:.1f}g")
        report.append("")
        
        # Macronutrient breakdown
        total_calories = analysis.calories_per_serving
        if total_calories > 0:
            protein_pct = (analysis.protein_per_serving_g * 4) / total_calories * 100
            carbs_pct = (analysis.carbs_per_serving_g * 4) / total_calories * 100
            fat_pct = (analysis.fat_per_serving_g * 9) / total_calories * 100
            
            report.append("MACRONUTRIENT BREAKDOWN:")
            report.append("-" * 25)
            report.append(f"Protein: {protein_pct:.1f}%")
            report.append(f"Carbohydrates: {carbs_pct:.1f}%")
            report.append(f"Fat: {fat_pct:.1f}%")
            report.append("")
        
        # Dietary analysis
        report.append("DIETARY ANALYSIS:")
        report.append("-" * 20)
        for diet_type, analysis_data in analysis.dietary_analysis.items():
            compatible = "✓" if analysis_data["compatible"] else "✗"
            confidence = analysis_data["confidence"]
            report.append(f"{diet_type.title()}: {compatible} (confidence: {confidence:.1%})")
        report.append("")
        
        # Ingredient breakdown
        report.append("INGREDIENT BREAKDOWN:")
        report.append("-" * 25)
        for profile in analysis.ingredient_profiles:
            report.append(f"• {profile.ingredient_name}")
            report.append(f"  Calories: {profile.calories:.0f}, Protein: {profile.protein_g:.1f}g")
            report.append(f"  Source: {profile.data_source}, Confidence: {profile.confidence:.1%}")
        
        return "\n".join(report)


def main():
    """Main nutritional analysis script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Nutritional analysis')
    parser.add_argument('--ingredient', help='Single ingredient to analyze')
    parser.add_argument('--recipe', help='Recipe JSON file to analyze')
    parser.add_argument('--servings', type=int, default=1, help='Number of servings')
    parser.add_argument('--config', help='Configuration file (JSON)')
    parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize analyzer
    analyzer = NutritionalAnalyzer(config)
    
    if args.ingredient:
        # Analyze single ingredient
        from enhanced_ingredient_parser import EnhancedIngredient
        ingredient = EnhancedIngredient(
            original_text=args.ingredient,
            ingredient_name=args.ingredient,
            confidence=1.0
        )
        
        profile = analyzer.analyze_ingredient(ingredient)
        if profile:
            print(f"Nutritional profile for {args.ingredient}:")
            print(f"Calories: {profile.calories:.0f}")
            print(f"Protein: {profile.protein_g:.1f}g")
            print(f"Carbs: {profile.carbs_g:.1f}g")
            print(f"Fat: {profile.fat_g:.1f}g")
            print(f"Source: {profile.data_source}")
            print(f"Confidence: {profile.confidence:.1%}")
        else:
            print(f"Could not analyze ingredient: {args.ingredient}")
    
    elif args.recipe:
        # Analyze recipe
        with open(args.recipe, 'r') as f:
            recipe_data = json.load(f)
        
        # Convert to ingredients (simplified)
        ingredients = []
        for ing_data in recipe_data.get("ingredients", []):
            ingredient = EnhancedIngredient(
                original_text=ing_data.get("original_text", ""),
                ingredient_name=ing_data.get("ingredient_name", ""),
                normalized_quantity=ing_data.get("normalized_quantity"),
                normalized_unit=ing_data.get("normalized_unit"),
                confidence=ing_data.get("confidence", 1.0)
            )
            ingredients.append(ingredient)
        
        analysis = analyzer.analyze_recipe(ingredients, args.servings)
        report = analyzer.generate_nutrition_report(analysis)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Nutrition report saved to {args.output}")
        else:
            print(report)
    
    else:
        print("Please specify --ingredient or --recipe")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())