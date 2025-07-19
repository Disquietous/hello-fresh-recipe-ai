"""
Annotation utilities for recipe text detection datasets.
Handles conversion between different annotation formats and validation.
"""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class TextAnnotation:
    """Text annotation data structure."""
    id: int
    category: str
    bbox: List[int]  # [x, y, width, height]
    text: str
    confidence: float = 1.0
    recipe_type: str = "unknown"
    language: str = "en"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'category': self.category,
            'bbox': self.bbox,
            'text': self.text,
            'confidence': self.confidence,
            'recipe_type': self.recipe_type,
            'language': self.language
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TextAnnotation':
        """Create from dictionary."""
        return cls(**data)


class AnnotationConverter:
    """Convert between different annotation formats."""
    
    # Class mapping for different annotation schemes
    CLASS_MAPPING = {
        'ingredient_line': 0,
        'ingredient_block': 1,
        'instruction_text': 2,
        'recipe_title': 3,
        'metadata_text': 4
    }
    
    REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
    
    def __init__(self):
        self.class_mapping = self.CLASS_MAPPING
        self.reverse_class_mapping = self.REVERSE_CLASS_MAPPING
    
    def coco_to_yolo(self, coco_annotation: Dict, image_width: int, image_height: int) -> str:
        """
        Convert COCO format annotation to YOLO format.
        
        Args:
            coco_annotation: COCO format annotation
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            YOLO format string
        """
        bbox = coco_annotation['bbox']  # [x, y, width, height]
        category = coco_annotation.get('category', 'ingredient_line')
        
        # Convert to YOLO format (normalized coordinates)
        x_center = (bbox[0] + bbox[2] / 2) / image_width
        y_center = (bbox[1] + bbox[3] / 2) / image_height
        width = bbox[2] / image_width
        height = bbox[3] / image_height
        
        # Get class ID
        class_id = self.class_mapping.get(category, 0)
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def yolo_to_coco(self, yolo_line: str, image_width: int, image_height: int) -> Dict:
        """
        Convert YOLO format to COCO format.
        
        Args:
            yolo_line: YOLO format line
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            COCO format annotation
        """
        parts = yolo_line.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO format: {yolo_line}")
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Convert to pixel coordinates
        x = (x_center - width / 2) * image_width
        y = (y_center - height / 2) * image_height
        w = width * image_width
        h = height * image_height
        
        category = self.reverse_class_mapping.get(class_id, 'ingredient_line')
        
        return {
            'bbox': [int(x), int(y), int(w), int(h)],
            'category': category,
            'confidence': 1.0
        }
    
    def icdar_to_yolo(self, icdar_annotation: str, image_width: int, image_height: int) -> str:
        """
        Convert ICDAR format to YOLO format.
        ICDAR format: "x1,y1,x2,y2,x3,y3,x4,y4,text"
        
        Args:
            icdar_annotation: ICDAR format annotation line
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            YOLO format string
        """
        parts = icdar_annotation.strip().split(',')
        if len(parts) < 9:
            raise ValueError(f"Invalid ICDAR format: {icdar_annotation}")
        
        # Extract coordinates (quad format)
        coords = [int(x) for x in parts[:8]]
        text = ','.join(parts[8:])  # Rejoin text in case it contains commas
        
        # Convert quad to bounding box
        x_coords = coords[::2]  # x1, x2, x3, x4
        y_coords = coords[1::2]  # y1, y2, y3, y4
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Determine category based on text content
        category = self._classify_text(text)
        
        # Convert to YOLO format
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        coco_annotation = {
            'bbox': bbox,
            'category': category,
            'text': text
        }
        
        return self.coco_to_yolo(coco_annotation, image_width, image_height)
    
    def _classify_text(self, text: str) -> str:
        """
        Automatically classify text into categories.
        
        Args:
            text: Text content
            
        Returns:
            Category name
        """
        text_lower = text.lower().strip()
        
        # Ingredient patterns
        ingredient_indicators = [
            'cup', 'cups', 'tbsp', 'tsp', 'teaspoon', 'tablespoon',
            'oz', 'lb', 'pound', 'gram', 'kg', 'ml', 'liter',
            'clove', 'cloves', 'piece', 'pieces'
        ]
        
        # Check for measurements (likely ingredients)
        if any(indicator in text_lower for indicator in ingredient_indicators):
            # Check if it's a single line or multi-line
            if '\n' in text or len(text) > 100:
                return 'ingredient_block'
            else:
                return 'ingredient_line'
        
        # Check for ingredient list headers
        if any(header in text_lower for header in ['ingredient', 'you will need', 'materials']):
            return 'ingredient_block'
        
        # Check for recipe titles
        if len(text.split()) <= 6 and text.istitle():
            return 'recipe_title'
        
        # Check for instructions
        instruction_indicators = ['mix', 'stir', 'bake', 'cook', 'heat', 'add', 'combine', 'preheat']
        if any(indicator in text_lower for indicator in instruction_indicators):
            return 'instruction_text'
        
        # Check for metadata
        metadata_indicators = ['serves', 'prep time', 'cook time', 'total time', 'difficulty', 'calories']
        if any(indicator in text_lower for indicator in metadata_indicators):
            return 'metadata_text'
        
        # Default to ingredient line for measurement-like text
        import re
        if re.search(r'\d+', text):
            return 'ingredient_line'
        
        return 'instruction_text'  # Default category


class AnnotationValidator:
    """Validate annotation quality and consistency."""
    
    def __init__(self):
        self.required_fields = ['bbox', 'category']
        self.valid_categories = set(AnnotationConverter.CLASS_MAPPING.keys())
    
    def validate_annotation(self, annotation: Dict, image_width: int, image_height: int) -> Dict:
        """
        Validate a single annotation.
        
        Args:
            annotation: Annotation dictionary
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in annotation:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Validate bounding box
        bbox = annotation['bbox']
        if len(bbox) != 4:
            errors.append("Bounding box must have 4 values [x, y, width, height]")
        else:
            x, y, w, h = bbox
            
            # Check bounds
            if x < 0 or y < 0:
                errors.append("Bounding box coordinates cannot be negative")
            
            if x + w > image_width or y + h > image_height:
                errors.append("Bounding box extends beyond image boundaries")
            
            if w <= 0 or h <= 0:
                errors.append("Bounding box must have positive width and height")
            
            # Check aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 20 or aspect_ratio < 0.05:
                warnings.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
            
            # Check size
            area = w * h
            total_area = image_width * image_height
            area_ratio = area / total_area
            
            if area_ratio < 0.0001:
                warnings.append("Very small text region")
            elif area_ratio > 0.5:
                warnings.append("Very large text region")
        
        # Validate category
        category = annotation.get('category', '')
        if category not in self.valid_categories:
            errors.append(f"Invalid category: {category}")
        
        # Validate text content if present
        text = annotation.get('text', '')
        if text:
            if len(text.strip()) == 0:
                warnings.append("Empty text content")
            elif len(text) > 1000:
                warnings.append("Very long text content")
        
        # Validate confidence
        confidence = annotation.get('confidence', 1.0)
        if not 0 <= confidence <= 1:
            errors.append("Confidence must be between 0 and 1")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'quality_score': max(0, 1.0 - len(warnings) * 0.1)
        }
    
    def validate_dataset(self, annotations_file: str, images_dir: str) -> Dict:
        """
        Validate entire dataset.
        
        Args:
            annotations_file: Path to annotations file
            images_dir: Path to images directory
            
        Returns:
            Dataset validation results
        """
        with open(annotations_file, 'r') as f:
            dataset = json.load(f)
        
        total_annotations = 0
        valid_annotations = 0
        all_errors = []
        all_warnings = []
        category_counts = {}
        
        for item in dataset:
            image_path = Path(images_dir) / item['image_id']
            if not image_path.exists():
                all_errors.append(f"Image not found: {item['image_id']}")
                continue
            
            # Get image dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                all_errors.append(f"Cannot read image: {item['image_id']}")
                continue
            
            height, width = image.shape[:2]
            
            # Validate each annotation
            for ann in item.get('annotations', []):
                total_annotations += 1
                validation = self.validate_annotation(ann, width, height)
                
                if validation['valid']:
                    valid_annotations += 1
                
                all_errors.extend(validation['errors'])
                all_warnings.extend(validation['warnings'])
                
                # Count categories
                category = ann.get('category', 'unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total_annotations': total_annotations,
            'valid_annotations': valid_annotations,
            'validation_rate': valid_annotations / total_annotations if total_annotations > 0 else 0,
            'errors': all_errors,
            'warnings': all_warnings,
            'category_distribution': category_counts,
            'quality_summary': {
                'excellent': sum(1 for e in all_errors if not e) if not all_errors else 0,
                'good': len(all_warnings),
                'poor': len(all_errors)
            }
        }


class DatasetConverter:
    """Convert external datasets to recipe text detection format."""
    
    def __init__(self):
        self.converter = AnnotationConverter()
        self.validator = AnnotationValidator()
    
    def convert_icdar_dataset(self, icdar_dir: str, output_dir: str, dataset_name: str = "icdar"):
        """
        Convert ICDAR dataset to YOLO format.
        
        Args:
            icdar_dir: Path to ICDAR dataset directory
            output_dir: Output directory for converted dataset
            dataset_name: Name of the dataset
        """
        icdar_path = Path(icdar_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create output subdirectories
        (output_path / 'images').mkdir(exist_ok=True)
        (output_path / 'labels').mkdir(exist_ok=True)
        
        # Find annotation files
        gt_files = list(icdar_path.glob('**/gt_*.txt'))
        
        converted_count = 0
        for gt_file in gt_files:
            # Get corresponding image file
            image_name = gt_file.name.replace('gt_', '').replace('.txt', '.jpg')
            image_path = gt_file.parent / image_name
            
            if not image_path.exists():
                # Try different extensions
                for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                    alt_path = gt_file.parent / (gt_file.stem.replace('gt_', '') + ext)
                    if alt_path.exists():
                        image_path = alt_path
                        break
            
            if not image_path.exists():
                print(f"Warning: Image not found for {gt_file}")
                continue
            
            # Read image to get dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Cannot read image {image_path}")
                continue
            
            height, width = image.shape[:2]
            
            # Convert annotations
            yolo_lines = []
            with open(gt_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        yolo_line = self.converter.icdar_to_yolo(line, width, height)
                        yolo_lines.append(yolo_line)
                    except Exception as e:
                        print(f"Warning: Failed to convert line {line_num} in {gt_file}: {e}")
            
            if yolo_lines:
                # Copy image
                output_image_path = output_path / 'images' / f"{dataset_name}_{converted_count:06d}.jpg"
                import shutil
                shutil.copy2(image_path, output_image_path)
                
                # Save YOLO annotations
                output_label_path = output_path / 'labels' / f"{dataset_name}_{converted_count:06d}.txt"
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                
                converted_count += 1
        
        print(f"Converted {converted_count} images from {dataset_name}")
        
        # Create dataset yaml
        yaml_content = f"""# {dataset_name.upper()} Dataset converted for recipe text detection
path: {output_path.absolute()}
train: images
val: images  # Use same for now, split later

nc: {len(AnnotationConverter.CLASS_MAPPING)}
names:
"""
        for name, idx in AnnotationConverter.CLASS_MAPPING.items():
            yaml_content += f"  {idx}: {name}\n"
        
        with open(output_path / f'{dataset_name}_data.yaml', 'w') as f:
            f.write(yaml_content)
    
    def create_recipe_annotation_template(self, image_path: str, output_path: str):
        """
        Create annotation template for a recipe image.
        
        Args:
            image_path: Path to recipe image
            output_path: Path to save annotation template
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        height, width = image.shape[:2]
        
        template = {
            "image_id": Path(image_path).name,
            "image_width": width,
            "image_height": height,
            "dataset_info": {
                "source": "manual_annotation",
                "annotator": "user",
                "annotation_date": "",
                "recipe_type": "unknown",  # handwritten, printed, digital
                "language": "en"
            },
            "annotations": [
                {
                    "id": 1,
                    "category": "ingredient_line",  # ingredient_line, ingredient_block, instruction_text, recipe_title, metadata_text
                    "bbox": [0, 0, 100, 30],  # [x, y, width, height]
                    "text": "Example: 2 cups all-purpose flour",
                    "confidence": 1.0,
                    "recipe_type": "unknown"
                }
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Created annotation template: {output_path}")
        print("Edit the template to add your annotations, then convert to YOLO format")


def main():
    """Example usage of annotation utilities."""
    
    # Create converter and validator
    converter = AnnotationConverter()
    validator = AnnotationValidator()
    dataset_converter = DatasetConverter()
    
    # Example: Create annotation template
    print("Creating annotation template...")
    # dataset_converter.create_recipe_annotation_template(
    #     "example_recipe.jpg", 
    #     "example_annotation.json"
    # )
    
    # Example: Convert ICDAR dataset
    print("To convert ICDAR dataset:")
    print("dataset_converter.convert_icdar_dataset('path/to/icdar', 'data/external_datasets/icdar2015')")
    
    # Example: Validate annotations
    print("To validate annotations:")
    print("validation_results = validator.validate_dataset('annotations.json', 'images/')")
    

if __name__ == "__main__":
    main()