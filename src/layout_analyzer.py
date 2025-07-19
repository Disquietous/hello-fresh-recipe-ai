#!/usr/bin/env python3
"""
Layout Analyzer for Recipe Text Formats
Analyzes and handles different ingredient list layouts:
- Single/multi-column layouts
- Bullet point lists
- Numbered lists
- Paragraph-style ingredients
- Table formats
- Mixed layouts
- Handwritten recipe card layouts
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
from enum import Enum
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path
sys.path.append(str(Path(__file__).parent))


class LayoutType(Enum):
    """Types of ingredient list layouts."""
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    BULLET_POINTS = "bullet_points"
    NUMBERED_LIST = "numbered_list"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    MIXED = "mixed"
    HANDWRITTEN_CARD = "handwritten_card"
    RECIPE_CARD = "recipe_card"
    BLOG_STYLE = "blog_style"


class TextRegionType(Enum):
    """Types of text regions."""
    INGREDIENT_LINE = "ingredient_line"
    INGREDIENT_BLOCK = "ingredient_block"
    QUANTITY = "quantity"
    UNIT = "unit"
    INGREDIENT_NAME = "ingredient_name"
    PREPARATION = "preparation"
    HEADER = "header"
    INSTRUCTION = "instruction"
    DECORATION = "decoration"
    NOISE = "noise"


@dataclass
class TextRegion:
    """Represents a detected text region."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    text: str
    confidence: float
    region_type: TextRegionType
    layout_role: str
    features: Dict[str, Any]


@dataclass
class LayoutAnalysis:
    """Result of layout analysis."""
    layout_type: LayoutType
    confidence: float
    text_regions: List[TextRegion]
    structure_info: Dict[str, Any]
    reading_order: List[int]
    grouping_info: Dict[str, List[int]]
    processing_recommendations: List[str]


class LayoutAnalyzer:
    """Analyzes ingredient list layouts and structures."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the layout analyzer."""
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Analysis parameters
        self.min_text_region_size = self.config.get('min_text_region_size', 100)
        self.column_threshold = self.config.get('column_threshold', 0.3)
        self.alignment_tolerance = self.config.get('alignment_tolerance', 10)
        self.clustering_eps = self.config.get('clustering_eps', 30)
        
        # Layout detection models
        self.layout_classifiers = self._initialize_layout_classifiers()
        
        self.logger.info("Layout Analyzer initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the analyzer."""
        logger = logging.getLogger('layout_analyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_layout(self, image_path: str) -> LayoutAnalysis:
        """
        Analyze the layout of a recipe image.
        
        Args:
            image_path: Path to the recipe image
            
        Returns:
            Layout analysis result
        """
        self.logger.info(f"Analyzing layout: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect text regions
        text_regions = self._detect_text_regions(image)
        
        # Classify layout type
        layout_type = self._classify_layout_type(image, text_regions)
        
        # Analyze structure
        structure_info = self._analyze_structure(image, text_regions, layout_type)
        
        # Determine reading order
        reading_order = self._determine_reading_order(text_regions, layout_type)
        
        # Group related regions
        grouping_info = self._group_related_regions(text_regions, layout_type)
        
        # Calculate confidence
        confidence = self._calculate_layout_confidence(image, text_regions, layout_type)
        
        # Generate processing recommendations
        processing_recommendations = self._generate_processing_recommendations(
            layout_type, structure_info, text_regions
        )
        
        return LayoutAnalysis(
            layout_type=layout_type,
            confidence=confidence,
            text_regions=text_regions,
            structure_info=structure_info,
            reading_order=reading_order,
            grouping_info=grouping_info,
            processing_recommendations=processing_recommendations
        )
    
    def _detect_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """Detect and classify text regions."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Find text regions using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w * h > self.min_text_region_size and w > 20 and h > 10:
                # Extract features
                features = self._extract_region_features(gray, (x, y, w, h))
                
                # Classify region type
                region_type = self._classify_region_type(gray, (x, y, w, h), features)
                
                # Create text region
                text_region = TextRegion(
                    bbox=(x, y, x + w, y + h),
                    text="",  # Will be filled by OCR
                    confidence=0.5,  # Default confidence
                    region_type=region_type,
                    layout_role="",  # Will be determined later
                    features=features
                )
                
                text_regions.append(text_region)
        
        return text_regions
    
    def _extract_region_features(self, gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Extract features from a text region."""
        x, y, w, h = bbox
        region = gray[y:y+h, x:x+w]
        
        features = {
            'width': w,
            'height': h,
            'aspect_ratio': w / h,
            'area': w * h,
            'position_x': x,
            'position_y': y,
            'center_x': x + w // 2,
            'center_y': y + h // 2,
            'brightness_mean': np.mean(region),
            'brightness_std': np.std(region),
            'edge_density': 0,
            'text_density': 0,
            'horizontal_projection': [],
            'vertical_projection': []
        }
        
        # Calculate edge density
        edges = cv2.Canny(region, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Calculate text density (approximate)
        binary_region = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        features['text_density'] = np.sum(binary_region == 0) / binary_region.size
        
        # Calculate projections
        features['horizontal_projection'] = np.sum(binary_region == 0, axis=1).tolist()
        features['vertical_projection'] = np.sum(binary_region == 0, axis=0).tolist()
        
        return features
    
    def _classify_region_type(self, gray: np.ndarray, bbox: Tuple[int, int, int, int], 
                            features: Dict[str, Any]) -> TextRegionType:
        """Classify the type of text region."""
        # Simple classification based on features
        aspect_ratio = features['aspect_ratio']
        area = features['area']
        text_density = features['text_density']
        
        # Small, square regions might be quantities
        if area < 500 and 0.5 < aspect_ratio < 2.0 and text_density > 0.1:
            return TextRegionType.QUANTITY
        
        # Long, thin regions might be ingredient lines
        elif aspect_ratio > 5.0 and text_density > 0.05:
            return TextRegionType.INGREDIENT_LINE
        
        # Large regions might be ingredient blocks
        elif area > 2000 and text_density > 0.03:
            return TextRegionType.INGREDIENT_BLOCK
        
        # Small regions with low text density might be decorations
        elif area < 200 and text_density < 0.02:
            return TextRegionType.DECORATION
        
        # Default classification
        else:
            return TextRegionType.INGREDIENT_LINE
    
    def _classify_layout_type(self, image: np.ndarray, text_regions: List[TextRegion]) -> LayoutType:
        """Classify the overall layout type."""
        if not text_regions:
            return LayoutType.MIXED
        
        # Extract region positions
        positions = [(region.bbox[0], region.bbox[1]) for region in text_regions]
        x_positions = [pos[0] for pos in positions]
        y_positions = [pos[1] for pos in positions]
        
        # Check for column structure
        if self._has_column_structure(x_positions):
            return LayoutType.MULTI_COLUMN
        
        # Check for bullet points
        if self._has_bullet_structure(image, text_regions):
            return LayoutType.BULLET_POINTS
        
        # Check for numbered list
        if self._has_numbered_structure(image, text_regions):
            return LayoutType.NUMBERED_LIST
        
        # Check for table structure
        if self._has_table_structure(text_regions):
            return LayoutType.TABLE
        
        # Check for paragraph structure
        if self._has_paragraph_structure(text_regions):
            return LayoutType.PARAGRAPH
        
        # Check for handwritten card
        if self._has_handwritten_structure(image, text_regions):
            return LayoutType.HANDWRITTEN_CARD
        
        # Check for blog style
        if self._has_blog_structure(image, text_regions):
            return LayoutType.BLOG_STYLE
        
        # Default to single column
        return LayoutType.SINGLE_COLUMN
    
    def _has_column_structure(self, x_positions: List[int]) -> bool:
        """Check if text regions form columns."""
        if len(x_positions) < 4:
            return False
        
        # Use clustering to detect columns
        x_array = np.array(x_positions).reshape(-1, 1)
        
        try:
            # Try 2-column clustering
            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(x_array)
            
            # Check cluster separation
            cluster_0_x = x_array[clusters == 0]
            cluster_1_x = x_array[clusters == 1]
            
            if len(cluster_0_x) > 1 and len(cluster_1_x) > 1:
                separation = abs(np.mean(cluster_0_x) - np.mean(cluster_1_x))
                return separation > 100
                
        except Exception:
            pass
        
        return False
    
    def _has_bullet_structure(self, image: np.ndarray, text_regions: List[TextRegion]) -> bool:
        """Check for bullet point structure."""
        # Look for small circular or square regions at the beginning of lines
        bullet_candidates = []
        
        for region in text_regions:
            if region.features['area'] < 100 and region.features['aspect_ratio'] < 1.5:
                bullet_candidates.append(region)
        
        # Check if bullet candidates align with larger text regions
        if len(bullet_candidates) > 2:
            # Check alignment with text regions
            for bullet in bullet_candidates:
                bullet_y = bullet.bbox[1]
                
                # Look for text regions at similar y-coordinate
                for text_region in text_regions:
                    if text_region != bullet and text_region.features['area'] > 200:
                        text_y = text_region.bbox[1]
                        if abs(bullet_y - text_y) < self.alignment_tolerance:
                            return True
        
        return False
    
    def _has_numbered_structure(self, image: np.ndarray, text_regions: List[TextRegion]) -> bool:
        """Check for numbered list structure."""
        # Look for small regions that might contain numbers
        number_candidates = []
        
        for region in text_regions:
            if (region.features['area'] < 200 and 
                region.features['aspect_ratio'] < 2.0 and
                region.features['text_density'] > 0.1):
                number_candidates.append(region)
        
        # Check if we have enough candidates in vertical alignment
        if len(number_candidates) > 2:
            y_positions = [region.bbox[1] for region in number_candidates]
            y_positions.sort()
            
            # Check for regular spacing
            if len(y_positions) > 2:
                spacings = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
                mean_spacing = np.mean(spacings)
                spacing_variance = np.var(spacings)
                
                # Regular spacing indicates numbered list
                return spacing_variance < mean_spacing * 0.5
        
        return False
    
    def _has_table_structure(self, text_regions: List[TextRegion]) -> bool:
        """Check for table structure."""
        if len(text_regions) < 6:
            return False
        
        # Check for grid-like arrangement
        x_positions = [region.bbox[0] for region in text_regions]
        y_positions = [region.bbox[1] for region in text_regions]
        
        # Use clustering to detect rows and columns
        try:
            # Cluster x-positions for columns
            x_array = np.array(x_positions).reshape(-1, 1)
            x_clusters = KMeans(n_clusters=min(3, len(set(x_positions))), random_state=42)
            x_labels = x_clusters.fit_predict(x_array)
            
            # Cluster y-positions for rows
            y_array = np.array(y_positions).reshape(-1, 1)
            y_clusters = KMeans(n_clusters=min(3, len(set(y_positions))), random_state=42)
            y_labels = y_clusters.fit_predict(y_array)
            
            # Check if we have a grid pattern
            unique_x_clusters = len(set(x_labels))
            unique_y_clusters = len(set(y_labels))
            
            return unique_x_clusters >= 2 and unique_y_clusters >= 3
            
        except Exception:
            return False
    
    def _has_paragraph_structure(self, text_regions: List[TextRegion]) -> bool:
        """Check for paragraph structure."""
        if len(text_regions) < 3:
            return False
        
        # Paragraph structure has similar x-positions but varying y-positions
        x_positions = [region.bbox[0] for region in text_regions]
        y_positions = [region.bbox[1] for region in text_regions]
        
        # Check x-alignment
        x_std = np.std(x_positions)
        x_mean = np.mean(x_positions)
        
        # Check y-variation
        y_std = np.std(y_positions)
        y_mean = np.mean(y_positions)
        
        # Paragraph: low x-variation, high y-variation
        return (x_std < x_mean * 0.1) and (y_std > y_mean * 0.1)
    
    def _has_handwritten_structure(self, image: np.ndarray, text_regions: List[TextRegion]) -> bool:
        """Check for handwritten card structure."""
        # Handwritten cards often have irregular spacing and alignment
        if len(text_regions) < 2:
            return False
        
        # Check for irregular alignment
        x_positions = [region.bbox[0] for region in text_regions]
        y_positions = [region.bbox[1] for region in text_regions]
        
        # Calculate alignment irregularity
        x_std = np.std(x_positions)
        y_std = np.std(y_positions)
        
        # Check for varying text region sizes
        areas = [region.features['area'] for region in text_regions]
        area_variation = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
        
        # Handwritten: high variation in positions and sizes
        return (x_std > 50 and y_std > 30 and area_variation > 0.5)
    
    def _has_blog_structure(self, image: np.ndarray, text_regions: List[TextRegion]) -> bool:
        """Check for blog-style structure."""
        if len(text_regions) < 5:
            return False
        
        # Blog style often has mixed region sizes and types
        areas = [region.features['area'] for region in text_regions]
        aspect_ratios = [region.features['aspect_ratio'] for region in text_regions]
        
        # Check for variety in sizes and shapes
        area_variation = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
        ratio_variation = np.std(aspect_ratios) / np.mean(aspect_ratios) if np.mean(aspect_ratios) > 0 else 0
        
        # Blog style: high variation in both area and aspect ratio
        return area_variation > 0.7 and ratio_variation > 0.5
    
    def _analyze_structure(self, image: np.ndarray, text_regions: List[TextRegion], 
                         layout_type: LayoutType) -> Dict[str, Any]:
        """Analyze the structure of the layout."""
        structure_info = {
            'layout_type': layout_type.value,
            'total_regions': len(text_regions),
            'region_types': {},
            'spatial_analysis': {},
            'alignment_analysis': {},
            'grouping_analysis': {}
        }
        
        # Count region types
        for region in text_regions:
            region_type = region.region_type.value
            structure_info['region_types'][region_type] = structure_info['region_types'].get(region_type, 0) + 1
        
        # Spatial analysis
        if text_regions:
            x_positions = [region.bbox[0] for region in text_regions]
            y_positions = [region.bbox[1] for region in text_regions]
            
            structure_info['spatial_analysis'] = {
                'x_range': (min(x_positions), max(x_positions)),
                'y_range': (min(y_positions), max(y_positions)),
                'x_mean': np.mean(x_positions),
                'y_mean': np.mean(y_positions),
                'x_std': np.std(x_positions),
                'y_std': np.std(y_positions)
            }
        
        # Layout-specific analysis
        if layout_type == LayoutType.MULTI_COLUMN:
            structure_info['column_analysis'] = self._analyze_columns(text_regions)
        elif layout_type == LayoutType.TABLE:
            structure_info['table_analysis'] = self._analyze_table(text_regions)
        elif layout_type == LayoutType.BULLET_POINTS:
            structure_info['bullet_analysis'] = self._analyze_bullets(text_regions)
        
        return structure_info
    
    def _analyze_columns(self, text_regions: List[TextRegion]) -> Dict[str, Any]:
        """Analyze column structure."""
        x_positions = [region.bbox[0] for region in text_regions]
        
        # Cluster x-positions to find columns
        x_array = np.array(x_positions).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(x_array)
        
        column_info = {
            'num_columns': 2,
            'column_boundaries': [],
            'column_widths': [],
            'regions_per_column': {}
        }
        
        for i in range(2):
            column_regions = [region for j, region in enumerate(text_regions) if clusters[j] == i]
            column_info['regions_per_column'][f'column_{i}'] = len(column_regions)
            
            if column_regions:
                x_positions_col = [region.bbox[0] for region in column_regions]
                widths = [region.bbox[2] - region.bbox[0] for region in column_regions]
                
                column_info['column_boundaries'].append((min(x_positions_col), max(x_positions_col)))
                column_info['column_widths'].append(np.mean(widths))
        
        return column_info
    
    def _analyze_table(self, text_regions: List[TextRegion]) -> Dict[str, Any]:
        """Analyze table structure."""
        # This is a simplified table analysis
        x_positions = [region.bbox[0] for region in text_regions]
        y_positions = [region.bbox[1] for region in text_regions]
        
        # Estimate number of rows and columns
        unique_x = len(set(x_positions))
        unique_y = len(set(y_positions))
        
        table_info = {
            'estimated_rows': unique_y,
            'estimated_columns': unique_x,
            'cell_regions': len(text_regions),
            'regularity_score': 0.5  # Simplified score
        }
        
        return table_info
    
    def _analyze_bullets(self, text_regions: List[TextRegion]) -> Dict[str, Any]:
        """Analyze bullet point structure."""
        # Find potential bullet regions
        bullet_regions = []
        text_line_regions = []
        
        for region in text_regions:
            if region.features['area'] < 100 and region.features['aspect_ratio'] < 1.5:
                bullet_regions.append(region)
            else:
                text_line_regions.append(region)
        
        bullet_info = {
            'num_bullets': len(bullet_regions),
            'num_text_lines': len(text_line_regions),
            'bullet_alignment': 'left',  # Simplified
            'spacing_regularity': 0.7  # Simplified score
        }
        
        return bullet_info
    
    def _determine_reading_order(self, text_regions: List[TextRegion], 
                               layout_type: LayoutType) -> List[int]:
        """Determine the reading order of text regions."""
        if not text_regions:
            return []
        
        # Sort by reading order based on layout type
        if layout_type == LayoutType.SINGLE_COLUMN:
            # Top to bottom
            sorted_regions = sorted(enumerate(text_regions), key=lambda x: x[1].bbox[1])
        
        elif layout_type == LayoutType.MULTI_COLUMN:
            # Left to right, then top to bottom within columns
            sorted_regions = sorted(enumerate(text_regions), 
                                  key=lambda x: (x[1].bbox[0] // 200, x[1].bbox[1]))
        
        elif layout_type == LayoutType.TABLE:
            # Row by row, left to right
            sorted_regions = sorted(enumerate(text_regions), 
                                  key=lambda x: (x[1].bbox[1] // 50, x[1].bbox[0]))
        
        elif layout_type == LayoutType.HANDWRITTEN_CARD:
            # More flexible ordering based on proximity
            sorted_regions = self._determine_handwritten_order(text_regions)
        
        else:
            # Default: top to bottom, left to right
            sorted_regions = sorted(enumerate(text_regions), 
                                  key=lambda x: (x[1].bbox[1], x[1].bbox[0]))
        
        return [idx for idx, region in sorted_regions]
    
    def _determine_handwritten_order(self, text_regions: List[TextRegion]) -> List[Tuple[int, TextRegion]]:
        """Determine reading order for handwritten cards."""
        # Use a simple distance-based approach
        if not text_regions:
            return []
        
        # Start with the topmost, leftmost region
        remaining = list(enumerate(text_regions))
        ordered = []
        
        # Find starting region
        start_idx, start_region = min(remaining, key=lambda x: (x[1].bbox[1], x[1].bbox[0]))
        ordered.append((start_idx, start_region))
        remaining.remove((start_idx, start_region))
        
        # Add regions in order of proximity
        while remaining:
            current_region = ordered[-1][1]
            current_center = (current_region.bbox[0] + current_region.bbox[2]) // 2, \
                           (current_region.bbox[1] + current_region.bbox[3]) // 2
            
            # Find closest remaining region
            distances = []
            for idx, region in remaining:
                region_center = (region.bbox[0] + region.bbox[2]) // 2, \
                               (region.bbox[1] + region.bbox[3]) // 2
                
                distance = np.sqrt((current_center[0] - region_center[0])**2 + 
                                 (current_center[1] - region_center[1])**2)
                distances.append((distance, idx, region))
            
            # Add closest region
            _, next_idx, next_region = min(distances)
            ordered.append((next_idx, next_region))
            remaining.remove((next_idx, next_region))
        
        return ordered
    
    def _group_related_regions(self, text_regions: List[TextRegion], 
                             layout_type: LayoutType) -> Dict[str, List[int]]:
        """Group related text regions."""
        grouping_info = {
            'ingredient_groups': [],
            'column_groups': [],
            'row_groups': [],
            'semantic_groups': []
        }
        
        if not text_regions:
            return grouping_info
        
        # Group by layout type
        if layout_type == LayoutType.MULTI_COLUMN:
            # Group by columns
            x_positions = [region.bbox[0] for region in text_regions]
            x_array = np.array(x_positions).reshape(-1, 1)
            
            try:
                kmeans = KMeans(n_clusters=2, random_state=42)
                clusters = kmeans.fit_predict(x_array)
                
                for i in range(2):
                    column_indices = [j for j, cluster in enumerate(clusters) if cluster == i]
                    grouping_info['column_groups'].append(column_indices)
            except:
                pass
        
        elif layout_type == LayoutType.TABLE:
            # Group by rows and columns
            self._group_table_regions(text_regions, grouping_info)
        
        elif layout_type == LayoutType.BULLET_POINTS:
            # Group bullets with their text
            self._group_bullet_regions(text_regions, grouping_info)
        
        # Semantic grouping (ingredient lines)
        self._group_semantic_regions(text_regions, grouping_info)
        
        return grouping_info
    
    def _group_table_regions(self, text_regions: List[TextRegion], 
                           grouping_info: Dict[str, List[int]]):
        """Group table regions by rows and columns."""
        # Simple row grouping by y-position
        y_positions = [region.bbox[1] for region in text_regions]
        
        # Use clustering to group rows
        try:
            y_array = np.array(y_positions).reshape(-1, 1)
            dbscan = DBSCAN(eps=self.clustering_eps, min_samples=1)
            row_clusters = dbscan.fit_predict(y_array)
            
            for cluster_id in set(row_clusters):
                if cluster_id >= 0:  # Ignore noise
                    row_indices = [i for i, cluster in enumerate(row_clusters) if cluster == cluster_id]
                    grouping_info['row_groups'].append(row_indices)
        except:
            pass
    
    def _group_bullet_regions(self, text_regions: List[TextRegion], 
                            grouping_info: Dict[str, List[int]]):
        """Group bullet regions with their associated text."""
        # Find bullet regions
        bullet_indices = []
        text_indices = []
        
        for i, region in enumerate(text_regions):
            if region.features['area'] < 100 and region.features['aspect_ratio'] < 1.5:
                bullet_indices.append(i)
            else:
                text_indices.append(i)
        
        # Match bullets with text regions
        for bullet_idx in bullet_indices:
            bullet_region = text_regions[bullet_idx]
            bullet_y = bullet_region.bbox[1]
            
            # Find text region at similar y-coordinate
            for text_idx in text_indices:
                text_region = text_regions[text_idx]
                text_y = text_region.bbox[1]
                
                if abs(bullet_y - text_y) < self.alignment_tolerance:
                    grouping_info['ingredient_groups'].append([bullet_idx, text_idx])
                    break
    
    def _group_semantic_regions(self, text_regions: List[TextRegion], 
                              grouping_info: Dict[str, List[int]]):
        """Group regions by semantic meaning."""
        # Simple grouping based on region types
        ingredient_regions = []
        quantity_regions = []
        
        for i, region in enumerate(text_regions):
            if region.region_type in [TextRegionType.INGREDIENT_LINE, TextRegionType.INGREDIENT_BLOCK]:
                ingredient_regions.append(i)
            elif region.region_type == TextRegionType.QUANTITY:
                quantity_regions.append(i)
        
        if ingredient_regions:
            grouping_info['semantic_groups'].append(ingredient_regions)
        if quantity_regions:
            grouping_info['semantic_groups'].append(quantity_regions)
    
    def _calculate_layout_confidence(self, image: np.ndarray, text_regions: List[TextRegion], 
                                   layout_type: LayoutType) -> float:
        """Calculate confidence score for layout classification."""
        if not text_regions:
            return 0.0
        
        # Base confidence on layout-specific features
        if layout_type == LayoutType.SINGLE_COLUMN:
            return self._calculate_single_column_confidence(text_regions)
        elif layout_type == LayoutType.MULTI_COLUMN:
            return self._calculate_multi_column_confidence(text_regions)
        elif layout_type == LayoutType.TABLE:
            return self._calculate_table_confidence(text_regions)
        elif layout_type == LayoutType.BULLET_POINTS:
            return self._calculate_bullet_confidence(text_regions)
        elif layout_type == LayoutType.HANDWRITTEN_CARD:
            return self._calculate_handwritten_confidence(text_regions)
        else:
            return 0.5  # Default confidence
    
    def _calculate_single_column_confidence(self, text_regions: List[TextRegion]) -> float:
        """Calculate confidence for single column layout."""
        if len(text_regions) < 2:
            return 0.3
        
        # Check x-alignment
        x_positions = [region.bbox[0] for region in text_regions]
        x_std = np.std(x_positions)
        x_mean = np.mean(x_positions)
        
        # Good alignment indicates high confidence
        alignment_score = 1.0 - min(x_std / x_mean, 1.0) if x_mean > 0 else 0.0
        
        return max(0.1, min(0.9, alignment_score))
    
    def _calculate_multi_column_confidence(self, text_regions: List[TextRegion]) -> float:
        """Calculate confidence for multi-column layout."""
        if len(text_regions) < 4:
            return 0.2
        
        # Use column detection confidence
        x_positions = [region.bbox[0] for region in text_regions]
        
        try:
            x_array = np.array(x_positions).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(x_array)
            
            # Check cluster separation
            cluster_0_x = x_array[clusters == 0]
            cluster_1_x = x_array[clusters == 1]
            
            if len(cluster_0_x) > 1 and len(cluster_1_x) > 1:
                separation = abs(np.mean(cluster_0_x) - np.mean(cluster_1_x))
                return min(separation / 200, 0.9)
            else:
                return 0.2
        except:
            return 0.2
    
    def _calculate_table_confidence(self, text_regions: List[TextRegion]) -> float:
        """Calculate confidence for table layout."""
        if len(text_regions) < 6:
            return 0.1
        
        # Check regularity of positioning
        x_positions = [region.bbox[0] for region in text_regions]
        y_positions = [region.bbox[1] for region in text_regions]
        
        # Count unique positions
        unique_x = len(set(x_positions))
        unique_y = len(set(y_positions))
        
        # Table should have multiple rows and columns
        if unique_x >= 2 and unique_y >= 3:
            regularity_score = min((unique_x * unique_y) / len(text_regions), 1.0)
            return max(0.3, min(0.8, regularity_score))
        else:
            return 0.1
    
    def _calculate_bullet_confidence(self, text_regions: List[TextRegion]) -> float:
        """Calculate confidence for bullet point layout."""
        # Count potential bullet regions
        bullet_candidates = sum(1 for region in text_regions 
                              if region.features['area'] < 100 and region.features['aspect_ratio'] < 1.5)
        
        # High confidence if we have multiple bullet candidates
        if bullet_candidates > 2:
            return min(0.7 + bullet_candidates * 0.1, 0.9)
        else:
            return 0.2
    
    def _calculate_handwritten_confidence(self, text_regions: List[TextRegion]) -> float:
        """Calculate confidence for handwritten card layout."""
        if len(text_regions) < 2:
            return 0.1
        
        # Check irregularity
        x_positions = [region.bbox[0] for region in text_regions]
        y_positions = [region.bbox[1] for region in text_regions]
        areas = [region.features['area'] for region in text_regions]
        
        x_std = np.std(x_positions)
        y_std = np.std(y_positions)
        area_variation = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
        
        # High variation indicates handwritten
        irregularity_score = min((x_std + y_std) / 200 + area_variation, 1.0)
        
        return max(0.1, min(0.8, irregularity_score))
    
    def _generate_processing_recommendations(self, layout_type: LayoutType, 
                                           structure_info: Dict[str, Any],
                                           text_regions: List[TextRegion]) -> List[str]:
        """Generate processing recommendations based on layout analysis."""
        recommendations = []
        
        # General recommendations
        if len(text_regions) > 20:
            recommendations.append("high_density_processing")
        
        # Layout-specific recommendations
        if layout_type == LayoutType.SINGLE_COLUMN:
            recommendations.extend([
                "sequential_processing",
                "top_to_bottom_reading_order"
            ])
        
        elif layout_type == LayoutType.MULTI_COLUMN:
            recommendations.extend([
                "column_aware_processing",
                "left_to_right_column_order",
                "column_boundary_detection"
            ])
        
        elif layout_type == LayoutType.TABLE:
            recommendations.extend([
                "table_structure_parsing",
                "row_column_alignment",
                "cell_content_extraction"
            ])
        
        elif layout_type == LayoutType.BULLET_POINTS:
            recommendations.extend([
                "bullet_point_detection",
                "bullet_text_association",
                "list_structure_parsing"
            ])
        
        elif layout_type == LayoutType.HANDWRITTEN_CARD:
            recommendations.extend([
                "handwritten_text_optimization",
                "flexible_alignment_tolerance",
                "stroke_width_normalization"
            ])
        
        elif layout_type == LayoutType.PARAGRAPH:
            recommendations.extend([
                "paragraph_segmentation",
                "line_grouping",
                "text_flow_analysis"
            ])
        
        elif layout_type == LayoutType.BLOG_STYLE:
            recommendations.extend([
                "mixed_content_handling",
                "section_detection",
                "content_prioritization"
            ])
        
        return recommendations
    
    def _initialize_layout_classifiers(self) -> Dict[str, Any]:
        """Initialize layout classification models."""
        # In a real implementation, these would be trained classifiers
        return {
            'layout_classifier': None,
            'region_classifier': None,
            'structure_analyzer': None
        }
    
    def visualize_layout_analysis(self, image_path: str, analysis: LayoutAnalysis, 
                                output_path: Optional[str] = None):
        """Visualize the layout analysis results."""
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Color map for different region types
        color_map = {
            TextRegionType.INGREDIENT_LINE: 'red',
            TextRegionType.INGREDIENT_BLOCK: 'blue',
            TextRegionType.QUANTITY: 'green',
            TextRegionType.UNIT: 'orange',
            TextRegionType.INGREDIENT_NAME: 'purple',
            TextRegionType.HEADER: 'brown',
            TextRegionType.DECORATION: 'gray',
            TextRegionType.NOISE: 'black'
        }
        
        # Draw bounding boxes
        for i, region in enumerate(analysis.text_regions):
            x1, y1, x2, y2 = region.bbox
            width = x2 - x1
            height = y2 - y1
            
            color = color_map.get(region.region_type, 'cyan')
            
            # Draw rectangle
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color, 
                                   facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Add region number
            ax.text(x1, y1-5, str(i), fontsize=10, color=color, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add title with layout information
        ax.set_title(f'Layout Analysis: {analysis.layout_type.value} '
                    f'(Confidence: {analysis.confidence:.3f})', 
                    fontsize=14, fontweight='bold')
        
        # Add legend
        legend_elements = [patches.Patch(color=color, label=region_type.value) 
                          for region_type, color in color_map.items()]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Remove axes
        ax.axis('off')
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    """Main function for layout analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recipe Layout Analyzer')
    parser.add_argument('input_image', help='Input recipe image path')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualization')
    parser.add_argument('--config', '-c', help='Configuration file (JSON)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize analyzer
    analyzer = LayoutAnalyzer(config)
    
    try:
        # Analyze layout
        analysis = analyzer.analyze_layout(args.input_image)
        
        # Print results
        print(f"Layout Analysis Results:")
        print(f"========================")
        print(f"Layout Type: {analysis.layout_type.value}")
        print(f"Confidence: {analysis.confidence:.3f}")
        print(f"Text Regions: {len(analysis.text_regions)}")
        print(f"Reading Order: {analysis.reading_order}")
        
        if args.verbose:
            print(f"\nStructure Information:")
            for key, value in analysis.structure_info.items():
                print(f"  {key}: {value}")
            
            print(f"\nProcessing Recommendations:")
            for rec in analysis.processing_recommendations:
                print(f"  - {rec}")
            
            print(f"\nText Regions:")
            for i, region in enumerate(analysis.text_regions):
                print(f"  Region {i}: {region.region_type.value} at {region.bbox}")
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save analysis results
            result_file = output_path / 'layout_analysis.json'
            with open(result_file, 'w') as f:
                json.dump(asdict(analysis), f, indent=2, default=str)
            
            print(f"Results saved to: {result_file}")
        
        # Generate visualization
        if args.visualize:
            if args.output:
                viz_path = Path(args.output) / 'layout_visualization.png'
            else:
                viz_path = None
            
            analyzer.visualize_layout_analysis(args.input_image, analysis, str(viz_path) if viz_path else None)
            
            if viz_path:
                print(f"Visualization saved to: {viz_path}")
        
    except Exception as e:
        print(f"Layout analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())