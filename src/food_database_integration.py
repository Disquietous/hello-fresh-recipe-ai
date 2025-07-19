#!/usr/bin/env python3
"""
Food Database Integration Module
Integrates with USDA FoodData Central, Spoonacular, and other food databases
for ingredient recognition, nutritional information, and standardization.
"""

import os
import sys
import json
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from urllib.parse import urlencode
import hashlib

# Add src to path
sys.path.append(str(Path(__file__).parent))

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False


@dataclass
class FoodItem:
    """Standardized food item from database."""
    id: str
    name: str
    description: str
    category: str
    brand: Optional[str]
    source: str  # 'usda', 'spoonacular', 'local'
    
    # Nutritional information
    calories_per_100g: Optional[float]
    protein_g: Optional[float]
    carbs_g: Optional[float]
    fat_g: Optional[float]
    fiber_g: Optional[float]
    sugar_g: Optional[float]
    sodium_mg: Optional[float]
    
    # Additional metadata
    aliases: List[str]
    common_units: List[str]
    density_g_per_ml: Optional[float]
    confidence: float
    
    # Database-specific data
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class IngredientMatch:
    """Result of ingredient matching against database."""
    query: str
    matches: List[FoodItem]
    best_match: Optional[FoodItem]
    confidence: float
    search_method: str  # 'exact', 'fuzzy', 'semantic'
    alternatives: List[str]


@dataclass
class NutritionalInfo:
    """Nutritional information for an ingredient."""
    food_item: FoodItem
    quantity: float
    unit: str
    
    # Per serving
    calories: Optional[float]
    protein_g: Optional[float]
    carbs_g: Optional[float]
    fat_g: Optional[float]
    fiber_g: Optional[float]
    sugar_g: Optional[float]
    sodium_mg: Optional[float]
    
    # Vitamins and minerals
    vitamin_c_mg: Optional[float]
    calcium_mg: Optional[float]
    iron_mg: Optional[float]
    potassium_mg: Optional[float]
    
    # Calculation metadata
    calculation_method: str
    serving_size_g: Optional[float]
    confidence: float


class FoodDatabaseIntegration:
    """Integration with multiple food databases."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize food database integration.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # API keys and endpoints
        self.usda_api_key = self.config.get('usda_api_key', os.environ.get('USDA_API_KEY'))
        self.spoonacular_api_key = self.config.get('spoonacular_api_key', os.environ.get('SPOONACULAR_API_KEY'))
        
        # API endpoints
        self.usda_endpoint = "https://api.nal.usda.gov/fdc/v1"
        self.spoonacular_endpoint = "https://api.spoonacular.com"
        
        # Cache settings
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_ttl = self.config.get('cache_ttl', 86400)  # 24 hours
        
        # Initialize local database
        self.db_path = Path(__file__).parent / "data" / "food_database.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_local_database()
        
        # Load local food data
        self._load_local_food_data()
        
        # Request session for connection pooling
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests per second max
        
        self.logger.info("Initialized FoodDatabaseIntegration")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for database integration."""
        logger = logging.getLogger('food_database_integration')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_local_database(self):
        """Initialize local SQLite database for caching."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS food_items (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    brand TEXT,
                    source TEXT NOT NULL,
                    calories_per_100g REAL,
                    protein_g REAL,
                    carbs_g REAL,
                    fat_g REAL,
                    fiber_g REAL,
                    sugar_g REAL,
                    sodium_mg REAL,
                    aliases TEXT,
                    common_units TEXT,
                    density_g_per_ml REAL,
                    confidence REAL,
                    raw_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    results TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_food_items_name ON food_items(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_food_items_category ON food_items(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_cache_query ON search_cache(query)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize local database: {e}")
    
    def _load_local_food_data(self):
        """Load local food data into database."""
        local_data_file = Path(__file__).parent / "data" / "common_ingredients.json"
        
        if not local_data_file.exists():
            self._create_default_food_data(local_data_file)
        
        try:
            with open(local_data_file, 'r') as f:
                data = json.load(f)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for item_data in data.get('ingredients', []):
                food_item = FoodItem(
                    id=item_data.get('id', ''),
                    name=item_data.get('name', ''),
                    description=item_data.get('description', ''),
                    category=item_data.get('category', ''),
                    brand=item_data.get('brand'),
                    source='local',
                    calories_per_100g=item_data.get('calories_per_100g'),
                    protein_g=item_data.get('protein_g'),
                    carbs_g=item_data.get('carbs_g'),
                    fat_g=item_data.get('fat_g'),
                    fiber_g=item_data.get('fiber_g'),
                    sugar_g=item_data.get('sugar_g'),
                    sodium_mg=item_data.get('sodium_mg'),
                    aliases=item_data.get('aliases', []),
                    common_units=item_data.get('common_units', []),
                    density_g_per_ml=item_data.get('density_g_per_ml'),
                    confidence=item_data.get('confidence', 0.9)
                )
                
                self._store_food_item(food_item)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to load local food data: {e}")
    
    def _create_default_food_data(self, file_path: Path):
        """Create default food data file."""
        default_data = {
            "ingredients": [
                {
                    "id": "local_flour_001",
                    "name": "All-Purpose Flour",
                    "description": "Wheat flour, enriched",
                    "category": "Grain Products",
                    "calories_per_100g": 364,
                    "protein_g": 10.3,
                    "carbs_g": 76.3,
                    "fat_g": 0.98,
                    "fiber_g": 2.7,
                    "sugar_g": 0.27,
                    "sodium_mg": 2,
                    "aliases": ["flour", "wheat flour", "ap flour", "plain flour"],
                    "common_units": ["cup", "tablespoon", "teaspoon", "gram", "ounce"],
                    "density_g_per_ml": 0.5,
                    "confidence": 0.95
                },
                {
                    "id": "local_sugar_001",
                    "name": "Granulated Sugar",
                    "description": "White sugar, granulated",
                    "category": "Sweets",
                    "calories_per_100g": 387,
                    "protein_g": 0,
                    "carbs_g": 99.98,
                    "fat_g": 0,
                    "fiber_g": 0,
                    "sugar_g": 99.8,
                    "sodium_mg": 1,
                    "aliases": ["sugar", "white sugar", "caster sugar", "superfine sugar"],
                    "common_units": ["cup", "tablespoon", "teaspoon", "gram", "ounce"],
                    "density_g_per_ml": 0.85,
                    "confidence": 0.95
                },
                {
                    "id": "local_salt_001",
                    "name": "Table Salt",
                    "description": "Salt, table",
                    "category": "Spices and Herbs",
                    "calories_per_100g": 0,
                    "protein_g": 0,
                    "carbs_g": 0,
                    "fat_g": 0,
                    "fiber_g": 0,
                    "sugar_g": 0,
                    "sodium_mg": 38758,
                    "aliases": ["salt", "table salt", "fine salt", "iodized salt"],
                    "common_units": ["teaspoon", "tablespoon", "pinch", "gram", "ounce"],
                    "density_g_per_ml": 1.22,
                    "confidence": 0.95
                },
                {
                    "id": "local_butter_001",
                    "name": "Butter",
                    "description": "Butter, salted",
                    "category": "Dairy and Egg Products",
                    "calories_per_100g": 717,
                    "protein_g": 0.85,
                    "carbs_g": 0.06,
                    "fat_g": 81.11,
                    "fiber_g": 0,
                    "sugar_g": 0.06,
                    "sodium_mg": 643,
                    "aliases": ["butter", "salted butter", "unsalted butter"],
                    "common_units": ["cup", "tablespoon", "teaspoon", "stick", "gram", "ounce"],
                    "density_g_per_ml": 0.911,
                    "confidence": 0.95
                },
                {
                    "id": "local_milk_001",
                    "name": "Whole Milk",
                    "description": "Milk, whole, 3.25% milkfat",
                    "category": "Dairy and Egg Products",
                    "calories_per_100g": 61,
                    "protein_g": 3.15,
                    "carbs_g": 4.8,
                    "fat_g": 3.25,
                    "fiber_g": 0,
                    "sugar_g": 5.05,
                    "sodium_mg": 40,
                    "aliases": ["milk", "whole milk", "full-fat milk"],
                    "common_units": ["cup", "tablespoon", "teaspoon", "liter", "milliliter"],
                    "density_g_per_ml": 1.03,
                    "confidence": 0.95
                },
                {
                    "id": "local_egg_001",
                    "name": "Chicken Egg",
                    "description": "Egg, whole, raw, fresh",
                    "category": "Dairy and Egg Products",
                    "calories_per_100g": 155,
                    "protein_g": 13.0,
                    "carbs_g": 1.1,
                    "fat_g": 11.0,
                    "fiber_g": 0,
                    "sugar_g": 1.1,
                    "sodium_mg": 124,
                    "aliases": ["egg", "eggs", "chicken egg", "fresh egg"],
                    "common_units": ["piece", "large", "medium", "small"],
                    "density_g_per_ml": 1.03,
                    "confidence": 0.95
                },
                {
                    "id": "local_onion_001",
                    "name": "Onion",
                    "description": "Onions, raw",
                    "category": "Vegetables and Vegetable Products",
                    "calories_per_100g": 40,
                    "protein_g": 1.1,
                    "carbs_g": 9.34,
                    "fat_g": 0.1,
                    "fiber_g": 1.7,
                    "sugar_g": 4.24,
                    "sodium_mg": 4,
                    "aliases": ["onion", "yellow onion", "white onion", "cooking onion"],
                    "common_units": ["cup", "medium", "large", "small", "piece"],
                    "density_g_per_ml": 0.96,
                    "confidence": 0.95
                },
                {
                    "id": "local_garlic_001",
                    "name": "Garlic",
                    "description": "Garlic, raw",
                    "category": "Vegetables and Vegetable Products",
                    "calories_per_100g": 149,
                    "protein_g": 6.36,
                    "carbs_g": 33.06,
                    "fat_g": 0.5,
                    "fiber_g": 2.1,
                    "sugar_g": 1.0,
                    "sodium_mg": 17,
                    "aliases": ["garlic", "garlic clove", "fresh garlic"],
                    "common_units": ["clove", "teaspoon", "tablespoon", "head"],
                    "density_g_per_ml": 0.9,
                    "confidence": 0.95
                },
                {
                    "id": "local_chicken_001",
                    "name": "Chicken Breast",
                    "description": "Chicken, broilers or fryers, breast, meat only, raw",
                    "category": "Poultry Products",
                    "calories_per_100g": 165,
                    "protein_g": 31.0,
                    "carbs_g": 0,
                    "fat_g": 3.6,
                    "fiber_g": 0,
                    "sugar_g": 0,
                    "sodium_mg": 74,
                    "aliases": ["chicken breast", "chicken", "boneless chicken breast"],
                    "common_units": ["pound", "ounce", "gram", "piece"],
                    "density_g_per_ml": 1.05,
                    "confidence": 0.95
                },
                {
                    "id": "local_tomato_001",
                    "name": "Tomato",
                    "description": "Tomatoes, red, ripe, raw, year round average",
                    "category": "Vegetables and Vegetable Products",
                    "calories_per_100g": 18,
                    "protein_g": 0.88,
                    "carbs_g": 3.89,
                    "fat_g": 0.2,
                    "fiber_g": 1.2,
                    "sugar_g": 2.63,
                    "sodium_mg": 5,
                    "aliases": ["tomato", "fresh tomato", "red tomato"],
                    "common_units": ["cup", "medium", "large", "small", "piece"],
                    "density_g_per_ml": 0.95,
                    "confidence": 0.95
                }
            ]
        }
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(default_data, f, indent=2)
    
    def _store_food_item(self, food_item: FoodItem):
        """Store food item in local database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO food_items (
                    id, name, description, category, brand, source,
                    calories_per_100g, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg,
                    aliases, common_units, density_g_per_ml, confidence, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                food_item.id,
                food_item.name,
                food_item.description,
                food_item.category,
                food_item.brand,
                food_item.source,
                food_item.calories_per_100g,
                food_item.protein_g,
                food_item.carbs_g,
                food_item.fat_g,
                food_item.fiber_g,
                food_item.sugar_g,
                food_item.sodium_mg,
                json.dumps(food_item.aliases),
                json.dumps(food_item.common_units),
                food_item.density_g_per_ml,
                food_item.confidence,
                json.dumps(food_item.raw_data) if food_item.raw_data else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store food item: {e}")
    
    def _rate_limit(self):
        """Apply rate limiting to API requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def search_local_database(self, query: str, limit: int = 10) -> List[FoodItem]:
        """Search local database for matching food items."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Search by name (exact and partial matches)
            cursor.execute('''
                SELECT * FROM food_items 
                WHERE name LIKE ? OR description LIKE ? OR aliases LIKE ?
                ORDER BY 
                    CASE 
                        WHEN name = ? THEN 1
                        WHEN name LIKE ? THEN 2
                        WHEN description LIKE ? THEN 3
                        ELSE 4
                    END,
                    confidence DESC
                LIMIT ?
            ''', (
                f'%{query}%', f'%{query}%', f'%{query}%',
                query, f'{query}%', f'{query}%',
                limit
            ))
            
            results = cursor.fetchall()
            conn.close()
            
            # Convert to FoodItem objects
            food_items = []
            for row in results:
                food_item = FoodItem(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    category=row[3],
                    brand=row[4],
                    source=row[5],
                    calories_per_100g=row[6],
                    protein_g=row[7],
                    carbs_g=row[8],
                    fat_g=row[9],
                    fiber_g=row[10],
                    sugar_g=row[11],
                    sodium_mg=row[12],
                    aliases=json.loads(row[13]) if row[13] else [],
                    common_units=json.loads(row[14]) if row[14] else [],
                    density_g_per_ml=row[15],
                    confidence=row[16],
                    raw_data=json.loads(row[17]) if row[17] else None
                )
                food_items.append(food_item)
            
            return food_items
            
        except Exception as e:
            self.logger.error(f"Failed to search local database: {e}")
            return []
    
    def search_usda_database(self, query: str, limit: int = 10) -> List[FoodItem]:
        """Search USDA FoodData Central database."""
        if not self.usda_api_key or not REQUESTS_AVAILABLE:
            return []
        
        try:
            self._rate_limit()
            
            # Search foods
            url = f"{self.usda_endpoint}/foods/search"
            params = {
                'api_key': self.usda_api_key,
                'query': query,
                'dataType': ['Foundation', 'SR Legacy'],
                'pageSize': limit,
                'sortBy': 'dataType.keyword',
                'sortOrder': 'asc'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            food_items = []
            
            for food_data in data.get('foods', []):
                # Extract nutritional information
                nutrients = {}
                for nutrient in food_data.get('foodNutrients', []):
                    nutrient_id = nutrient.get('nutrientId')
                    value = nutrient.get('value')
                    
                    if nutrient_id == 1008:  # Energy (calories)
                        nutrients['calories_per_100g'] = value
                    elif nutrient_id == 1003:  # Protein
                        nutrients['protein_g'] = value
                    elif nutrient_id == 1005:  # Carbohydrates
                        nutrients['carbs_g'] = value
                    elif nutrient_id == 1004:  # Total lipid (fat)
                        nutrients['fat_g'] = value
                    elif nutrient_id == 1079:  # Fiber
                        nutrients['fiber_g'] = value
                    elif nutrient_id == 2000:  # Sugars
                        nutrients['sugar_g'] = value
                    elif nutrient_id == 1093:  # Sodium
                        nutrients['sodium_mg'] = value
                
                # Create food item
                food_item = FoodItem(
                    id=f"usda_{food_data.get('fdcId')}",
                    name=food_data.get('description', ''),
                    description=food_data.get('description', ''),
                    category=food_data.get('foodCategory', ''),
                    brand=food_data.get('brandOwner'),
                    source='usda',
                    calories_per_100g=nutrients.get('calories_per_100g'),
                    protein_g=nutrients.get('protein_g'),
                    carbs_g=nutrients.get('carbs_g'),
                    fat_g=nutrients.get('fat_g'),
                    fiber_g=nutrients.get('fiber_g'),
                    sugar_g=nutrients.get('sugar_g'),
                    sodium_mg=nutrients.get('sodium_mg'),
                    aliases=[],
                    common_units=['gram', 'ounce', 'pound'],
                    density_g_per_ml=None,
                    confidence=0.8,
                    raw_data=food_data
                )
                
                food_items.append(food_item)
                
                # Store in local database for caching
                self._store_food_item(food_item)
            
            return food_items
            
        except Exception as e:
            self.logger.error(f"USDA API error: {e}")
            return []
    
    def search_spoonacular_database(self, query: str, limit: int = 10) -> List[FoodItem]:
        """Search Spoonacular ingredients database."""
        if not self.spoonacular_api_key or not REQUESTS_AVAILABLE:
            return []
        
        try:
            self._rate_limit()
            
            # Search ingredients
            url = f"{self.spoonacular_endpoint}/food/ingredients/search"
            params = {
                'apiKey': self.spoonacular_api_key,
                'query': query,
                'number': limit,
                'sort': 'popularity',
                'sortDirection': 'desc'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            food_items = []
            
            for ingredient_data in data.get('results', []):
                ingredient_id = ingredient_data.get('id')
                
                # Get detailed information
                detail_url = f"{self.spoonacular_endpoint}/food/ingredients/{ingredient_id}/information"
                detail_params = {
                    'apiKey': self.spoonacular_api_key,
                    'amount': 100,
                    'unit': 'grams'
                }
                
                detail_response = self.session.get(detail_url, params=detail_params, timeout=10)
                if detail_response.status_code != 200:
                    continue
                
                detail_data = detail_response.json()
                
                # Extract nutrition
                nutrition = detail_data.get('nutrition', {})
                nutrients = nutrition.get('nutrients', [])
                
                nutrient_values = {}
                for nutrient in nutrients:
                    name = nutrient.get('name', '').lower()
                    value = nutrient.get('amount', 0)
                    
                    if 'calorie' in name:
                        nutrient_values['calories_per_100g'] = value
                    elif 'protein' in name:
                        nutrient_values['protein_g'] = value
                    elif 'carbohydrate' in name:
                        nutrient_values['carbs_g'] = value
                    elif 'fat' in name:
                        nutrient_values['fat_g'] = value
                    elif 'fiber' in name:
                        nutrient_values['fiber_g'] = value
                    elif 'sugar' in name:
                        nutrient_values['sugar_g'] = value
                    elif 'sodium' in name:
                        nutrient_values['sodium_mg'] = value
                
                # Create food item
                food_item = FoodItem(
                    id=f"spoonacular_{ingredient_id}",
                    name=ingredient_data.get('name', ''),
                    description=detail_data.get('original', ''),
                    category=detail_data.get('categoryPath', [''])[0] if detail_data.get('categoryPath') else '',
                    brand=None,
                    source='spoonacular',
                    calories_per_100g=nutrient_values.get('calories_per_100g'),
                    protein_g=nutrient_values.get('protein_g'),
                    carbs_g=nutrient_values.get('carbs_g'),
                    fat_g=nutrient_values.get('fat_g'),
                    fiber_g=nutrient_values.get('fiber_g'),
                    sugar_g=nutrient_values.get('sugar_g'),
                    sodium_mg=nutrient_values.get('sodium_mg'),
                    aliases=detail_data.get('possibleUnits', []),
                    common_units=detail_data.get('possibleUnits', []),
                    density_g_per_ml=None,
                    confidence=0.8,
                    raw_data=detail_data
                )
                
                food_items.append(food_item)
                
                # Store in local database for caching
                self._store_food_item(food_item)
            
            return food_items
            
        except Exception as e:
            self.logger.error(f"Spoonacular API error: {e}")
            return []
    
    def search_ingredient(self, query: str, limit: int = 10) -> IngredientMatch:
        """
        Search for ingredient across all databases.
        
        Args:
            query: Ingredient name to search for
            limit: Maximum number of results
            
        Returns:
            IngredientMatch with results from all sources
        """
        query_normalized = query.lower().strip()
        
        # Check cache first
        if self.cache_enabled:
            cached_result = self._get_cached_search(query_normalized)
            if cached_result:
                return cached_result
        
        all_matches = []
        
        # Search local database first
        local_matches = self.search_local_database(query_normalized, limit)
        all_matches.extend(local_matches)
        
        # Search external databases if needed
        if len(all_matches) < limit:
            # Search USDA
            usda_matches = self.search_usda_database(query_normalized, limit)
            all_matches.extend(usda_matches)
            
            # Search Spoonacular
            spoonacular_matches = self.search_spoonacular_database(query_normalized, limit)
            all_matches.extend(spoonacular_matches)
        
        # Remove duplicates and sort by confidence
        seen_names = set()
        unique_matches = []
        for match in all_matches:
            if match.name.lower() not in seen_names:
                seen_names.add(match.name.lower())
                unique_matches.append(match)
        
        # Sort by confidence and relevance
        unique_matches.sort(key=lambda x: (x.confidence, self._calculate_relevance(query_normalized, x.name)), reverse=True)
        
        # Limit results
        unique_matches = unique_matches[:limit]
        
        # Find best match
        best_match = None
        best_confidence = 0
        
        for match in unique_matches:
            relevance = self._calculate_relevance(query_normalized, match.name)
            combined_confidence = (match.confidence + relevance) / 2
            
            if combined_confidence > best_confidence:
                best_confidence = combined_confidence
                best_match = match
        
        # Generate alternatives
        alternatives = self._generate_alternatives(query_normalized, unique_matches)
        
        # Determine search method
        search_method = "fuzzy"
        if best_match and best_match.name.lower() == query_normalized:
            search_method = "exact"
        
        # Create result
        result = IngredientMatch(
            query=query,
            matches=unique_matches,
            best_match=best_match,
            confidence=best_confidence,
            search_method=search_method,
            alternatives=alternatives
        )
        
        # Cache result
        if self.cache_enabled:
            self._cache_search_result(query_normalized, result)
        
        return result
    
    def _calculate_relevance(self, query: str, name: str) -> float:
        """Calculate relevance score between query and name."""
        if not FUZZYWUZZY_AVAILABLE:
            return 1.0 if query.lower() in name.lower() else 0.5
        
        # Exact match
        if query.lower() == name.lower():
            return 1.0
        
        # Fuzzy matching
        ratio = fuzz.ratio(query.lower(), name.lower()) / 100
        partial_ratio = fuzz.partial_ratio(query.lower(), name.lower()) / 100
        token_sort_ratio = fuzz.token_sort_ratio(query.lower(), name.lower()) / 100
        
        return max(ratio, partial_ratio, token_sort_ratio)
    
    def _generate_alternatives(self, query: str, matches: List[FoodItem]) -> List[str]:
        """Generate alternative ingredient names."""
        alternatives = []
        
        # Add similar names from matches
        for match in matches[:5]:  # Top 5 matches
            if match.name.lower() != query.lower():
                alternatives.append(match.name)
            
            # Add aliases
            for alias in match.aliases:
                if alias.lower() != query.lower() and alias not in alternatives:
                    alternatives.append(alias)
        
        return alternatives[:10]  # Limit to 10 alternatives
    
    def _get_cached_search(self, query: str) -> Optional[IngredientMatch]:
        """Get cached search result."""
        try:
            query_hash = hashlib.md5(query.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT results FROM search_cache 
                WHERE query_hash = ? AND 
                datetime(created_at) > datetime('now', '-{} seconds')
            '''.format(self.cache_ttl), (query_hash,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return IngredientMatch(**json.loads(result[0]))
            
        except Exception as e:
            self.logger.error(f"Failed to get cached search: {e}")
        
        return None
    
    def _cache_search_result(self, query: str, result: IngredientMatch):
        """Cache search result."""
        try:
            query_hash = hashlib.md5(query.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO search_cache (query_hash, query, results)
                VALUES (?, ?, ?)
            ''', (query_hash, query, json.dumps(asdict(result), default=str)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to cache search result: {e}")
    
    def get_nutritional_info(self, food_item: FoodItem, quantity: float, unit: str) -> NutritionalInfo:
        """
        Calculate nutritional information for a specific quantity.
        
        Args:
            food_item: Food item from database
            quantity: Quantity amount
            unit: Unit of measurement
            
        Returns:
            Nutritional information for the specified quantity
        """
        # Convert quantity to grams
        serving_size_g = self._convert_to_grams(quantity, unit, food_item)
        
        if not serving_size_g:
            return NutritionalInfo(
                food_item=food_item,
                quantity=quantity,
                unit=unit,
                calories=None,
                protein_g=None,
                carbs_g=None,
                fat_g=None,
                fiber_g=None,
                sugar_g=None,
                sodium_mg=None,
                vitamin_c_mg=None,
                calcium_mg=None,
                iron_mg=None,
                potassium_mg=None,
                calculation_method="failed",
                serving_size_g=None,
                confidence=0.0
            )
        
        # Calculate nutrition per serving
        factor = serving_size_g / 100  # Convert from per 100g to per serving
        
        return NutritionalInfo(
            food_item=food_item,
            quantity=quantity,
            unit=unit,
            calories=food_item.calories_per_100g * factor if food_item.calories_per_100g else None,
            protein_g=food_item.protein_g * factor if food_item.protein_g else None,
            carbs_g=food_item.carbs_g * factor if food_item.carbs_g else None,
            fat_g=food_item.fat_g * factor if food_item.fat_g else None,
            fiber_g=food_item.fiber_g * factor if food_item.fiber_g else None,
            sugar_g=food_item.sugar_g * factor if food_item.sugar_g else None,
            sodium_mg=food_item.sodium_mg * factor if food_item.sodium_mg else None,
            vitamin_c_mg=None,  # Would need additional data
            calcium_mg=None,
            iron_mg=None,
            potassium_mg=None,
            calculation_method="weight_based",
            serving_size_g=serving_size_g,
            confidence=0.8
        )
    
    def _convert_to_grams(self, quantity: float, unit: str, food_item: FoodItem) -> Optional[float]:
        """Convert quantity and unit to grams."""
        unit_lower = unit.lower()
        
        # Direct weight units
        if unit_lower in ['g', 'gram', 'grams']:
            return quantity
        elif unit_lower in ['kg', 'kilogram', 'kilograms']:
            return quantity * 1000
        elif unit_lower in ['oz', 'ounce', 'ounces']:
            return quantity * 28.35
        elif unit_lower in ['lb', 'lbs', 'pound', 'pounds']:
            return quantity * 453.59
        
        # Volume units (need density)
        elif unit_lower in ['ml', 'milliliter', 'milliliters']:
            if food_item.density_g_per_ml:
                return quantity * food_item.density_g_per_ml
        elif unit_lower in ['l', 'liter', 'liters']:
            if food_item.density_g_per_ml:
                return quantity * 1000 * food_item.density_g_per_ml
        elif unit_lower in ['cup', 'cups', 'c']:
            if food_item.density_g_per_ml:
                return quantity * 240 * food_item.density_g_per_ml
        elif unit_lower in ['tbsp', 'tablespoon', 'tablespoons']:
            if food_item.density_g_per_ml:
                return quantity * 15 * food_item.density_g_per_ml
        elif unit_lower in ['tsp', 'teaspoon', 'teaspoons']:
            if food_item.density_g_per_ml:
                return quantity * 5 * food_item.density_g_per_ml
        
        # Count units (need typical weights)
        elif unit_lower in ['piece', 'pieces', 'item', 'items']:
            # Use typical weights for common items
            typical_weights = {
                'egg': 50,
                'apple': 150,
                'banana': 120,
                'orange': 130,
                'tomato': 120,
                'onion': 110,
                'potato': 150,
                'carrot': 60,
                'garlic': 3,  # per clove
                'chicken': 200,  # per breast
                'bread': 25  # per slice
            }
            
            for keyword, weight in typical_weights.items():
                if keyword in food_item.name.lower():
                    return quantity * weight
        
        # Default fallback
        return None
    
    def batch_search_ingredients(self, queries: List[str]) -> List[IngredientMatch]:
        """Search for multiple ingredients in batch."""
        results = []
        
        for query in queries:
            try:
                result = self.search_ingredient(query)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to search ingredient '{query}': {e}")
                # Create empty result
                results.append(IngredientMatch(
                    query=query,
                    matches=[],
                    best_match=None,
                    confidence=0.0,
                    search_method="failed",
                    alternatives=[]
                ))
        
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count food items by source
            cursor.execute('SELECT source, COUNT(*) FROM food_items GROUP BY source')
            source_counts = dict(cursor.fetchall())
            
            # Count total items
            cursor.execute('SELECT COUNT(*) FROM food_items')
            total_items = cursor.fetchone()[0]
            
            # Count cached searches
            cursor.execute('SELECT COUNT(*) FROM search_cache')
            cached_searches = cursor.fetchone()[0]
            
            # Count categories
            cursor.execute('SELECT COUNT(DISTINCT category) FROM food_items')
            categories = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_items': total_items,
                'source_counts': source_counts,
                'cached_searches': cached_searches,
                'categories': categories,
                'api_keys': {
                    'usda': bool(self.usda_api_key),
                    'spoonacular': bool(self.spoonacular_api_key)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}


def main():
    """Main food database integration script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Search food databases')
    parser.add_argument('--search', '-s', help='Ingredient to search for')
    parser.add_argument('--file', '-f', help='File with ingredients to search (one per line)')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--usda-key', help='USDA API key')
    parser.add_argument('--spoonacular-key', help='Spoonacular API key')
    parser.add_argument('--limit', '-l', type=int, default=10, help='Maximum results per search')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {}
    if args.usda_key:
        config['usda_api_key'] = args.usda_key
    if args.spoonacular_key:
        config['spoonacular_api_key'] = args.spoonacular_key
    
    # Initialize database integration
    db_integration = FoodDatabaseIntegration(config)
    
    try:
        if args.stats:
            # Show database statistics
            stats = db_integration.get_database_stats()
            print("Database Statistics:")
            print(f"  Total items: {stats.get('total_items', 0)}")
            print(f"  Categories: {stats.get('categories', 0)}")
            print(f"  Cached searches: {stats.get('cached_searches', 0)}")
            print(f"  Source counts: {stats.get('source_counts', {})}")
            print(f"  API keys available: {stats.get('api_keys', {})}")
            return 0
        
        # Search ingredients
        search_queries = []
        
        if args.search:
            search_queries.append(args.search)
        elif args.file:
            with open(args.file, 'r') as f:
                search_queries = [line.strip() for line in f if line.strip()]
        else:
            # Interactive mode
            print("Enter ingredient names to search (empty line to finish):")
            while True:
                query = input("> ").strip()
                if not query:
                    break
                search_queries.append(query)
        
        if not search_queries:
            print("No search queries provided.")
            return 1
        
        # Perform searches
        results = db_integration.batch_search_ingredients(search_queries)
        
        # Display results
        print(f"\nFood Database Search Results:")
        print(f"=============================")
        
        for result in results:
            print(f"\nQuery: {result.query}")
            print(f"Method: {result.search_method}")
            print(f"Confidence: {result.confidence:.3f}")
            
            if result.best_match:
                print(f"Best match: {result.best_match.name}")
                print(f"  Category: {result.best_match.category}")
                print(f"  Source: {result.best_match.source}")
                
                if result.best_match.calories_per_100g:
                    print(f"  Calories/100g: {result.best_match.calories_per_100g:.1f}")
                if result.best_match.protein_g:
                    print(f"  Protein/100g: {result.best_match.protein_g:.1f}g")
                if result.best_match.carbs_g:
                    print(f"  Carbs/100g: {result.best_match.carbs_g:.1f}g")
                if result.best_match.fat_g:
                    print(f"  Fat/100g: {result.best_match.fat_g:.1f}g")
            
            if result.alternatives:
                print(f"  Alternatives: {', '.join(result.alternatives[:5])}")
            
            print(f"  Total matches: {len(result.matches)}")
        
        # Save results if requested
        if args.output:
            output_data = {
                'search_results': [asdict(result) for result in results],
                'summary': {
                    'total_queries': len(search_queries),
                    'successful_searches': len([r for r in results if r.best_match]),
                    'average_confidence': sum(r.confidence for r in results) / len(results) if results else 0
                }
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            print(f"\nResults saved to: {args.output}")
    
    except Exception as e:
        print(f"Search failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())