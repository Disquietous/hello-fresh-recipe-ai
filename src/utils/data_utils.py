"""
Data utilities for HelloFresh Recipe AI project.
Functions for text detection data preprocessing, augmentation, and validation.
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import yaml
import json
from typing import List, Dict, Tuple, Optional
import random
from collections import defaultdict

from annotation_utils import AnnotationConverter, AnnotationValidator
from text_augmentation import TextAugmentationPipeline, RecipeFormatAugmenter
from recipe_format_handler import RecipeFormatClassifier, RecipeFormatProcessor


def validate_dataset_structure(data_dir, dataset_type='yolo'):
    """
    Validate dataset directory structure for text detection.
    
    Args:
        data_dir (str): Path to dataset directory
        dataset_type (str): Type of dataset structure ('yolo', 'coco', 'raw')
        
    Returns:
        bool: True if structure is valid
    """
    data_path = Path(data_dir)
    
    if dataset_type == 'yolo':
        required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
        
        for req_dir in required_dirs:
            dir_path = data_path / req_dir
            if not dir_path.exists():
                print(f"Missing required directory: {dir_path}")
                return False
        
        # Check for corresponding images and labels
        train_images = list((data_path / 'train/images').glob('*'))
        train_labels = list((data_path / 'train/labels').glob('*'))
        
        if len(train_images) == 0:
            print("No training images found")
            return False
        
        if len(train_labels) == 0:
            print("No training labels found")
            return False
        
        print(f"YOLO dataset structure validation passed!")
        print(f"  Training images: {len(train_images)}")
        print(f"  Training labels: {len(train_labels)}")
        
    elif dataset_type == 'recipe_text':
        required_dirs = ['recipe_cards', 'external_datasets', 'processed']
        
        for req_dir in required_dirs:
            dir_path = data_path / req_dir
            if not dir_path.exists():
                print(f"Missing required directory: {dir_path}")
                return False
        
        print("Recipe text dataset structure validation passed!")
    
    return True


def prepare_text_detection_dataset(raw_data_dir: str, output_dir: str, 
                                  split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                                  augment_data: bool = True,
                                  augmentations_per_image: int = 3) -> Dict[str, int]:
    """
    Prepare complete text detection dataset from raw recipe images.
    
    Args:
        raw_data_dir: Directory with raw recipe images and annotations
        output_dir: Output directory for processed dataset
        split_ratios: Train/val/test split ratios
        augment_data: Whether to apply data augmentation
        augmentations_per_image: Number of augmentations per original image
        
    Returns:
        Dictionary with dataset statistics
    """
    raw_path = Path(raw_data_dir)
    output_path = Path(output_dir)
    
    print(f"Preparing text detection dataset from {raw_data_dir}")
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    format_classifier = RecipeFormatClassifier()
    format_processor = RecipeFormatProcessor()
    augmenter = TextAugmentationPipeline()
    format_augmenter = RecipeFormatAugmenter()
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(raw_path.glob(f'**/*{ext}')))
        image_files.extend(list(raw_path.glob(f'**/*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle for random split
    random.shuffle(image_files)
    
    # Calculate split indices
    total = len(image_files)
    train_end = int(total * split_ratios[0])
    val_end = train_end + int(total * split_ratios[1])
    
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    stats = {'train': 0, 'val': 0, 'test': 0, 'augmented': 0}
    
    for split_name, files in splits.items():
        print(f"Processing {split_name} split: {len(files)} images")
        
        for img_idx, img_file in enumerate(files):
            try:
                # Load image
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                # Find corresponding annotation file
                annotation_file = _find_annotation_file(img_file)
                boxes, class_labels = [], []
                
                if annotation_file and annotation_file.exists():
                    boxes, class_labels = _load_annotations(annotation_file, img_file)
                
                # Classify and process recipe format
                metadata = format_classifier.classify_recipe_format(image)
                processed_image, _ = format_processor.process_recipe_image(image, metadata)
                
                # Save original processed image
                output_image_path = output_path / split_name / 'images' / f"{split_name}_{img_idx:06d}.jpg"
                cv2.imwrite(str(output_image_path), processed_image)
                
                # Save labels if available
                if boxes:
                    output_label_path = output_path / split_name / 'labels' / f"{split_name}_{img_idx:06d}.txt"
                    _save_yolo_labels(output_label_path, boxes, class_labels)
                
                stats[split_name] += 1
                
                # Apply augmentation for training data
                if augment_data and split_name == 'train':
                    for aug_idx in range(augmentations_per_image):
                        # Apply format-specific augmentation
                        aug_image = format_augmenter.augment_for_format(
                            processed_image, metadata.format_type.value
                        )
                        
                        # Apply general text augmentations
                        if boxes:
                            aug_image, aug_boxes, aug_labels = augmenter.augment_image_with_boxes(
                                aug_image, boxes, class_labels
                            )
                        else:
                            aug_image = augmenter.apply_text_specific_augmentations(aug_image)
                            aug_boxes, aug_labels = [], []
                        
                        # Save augmented image
                        aug_image_path = output_path / split_name / 'images' / f"{split_name}_{img_idx:06d}_aug_{aug_idx:03d}.jpg"
                        cv2.imwrite(str(aug_image_path), aug_image)
                        
                        # Save augmented labels
                        if aug_boxes:
                            aug_label_path = output_path / split_name / 'labels' / f"{split_name}_{img_idx:06d}_aug_{aug_idx:03d}.txt"
                            _save_yolo_labels(aug_label_path, aug_boxes, aug_labels)
                        
                        stats['augmented'] += 1
                
                if (img_idx + 1) % 100 == 0:
                    print(f"  Processed {img_idx + 1}/{len(files)} images")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
    
    # Create dataset configuration
    _create_dataset_config(output_path, stats)
    
    print(f"Dataset preparation complete!")
    print(f"  Train: {stats['train']} images")
    print(f"  Val: {stats['val']} images") 
    print(f"  Test: {stats['test']} images")
    print(f"  Augmented: {stats['augmented']} images")
    
    return stats


def _find_annotation_file(image_file: Path) -> Optional[Path]:
    """Find corresponding annotation file for an image."""
    # Look for various annotation formats
    base_name = image_file.stem
    annotation_dir = image_file.parent / 'annotations'
    
    # Try different annotation file formats
    for ext in ['.txt', '.json', '.xml']:
        annotation_file = annotation_dir / f"{base_name}{ext}"
        if annotation_file.exists():
            return annotation_file
    
    # Try in same directory
    for ext in ['.txt', '.json', '.xml']:
        annotation_file = image_file.parent / f"{base_name}{ext}"
        if annotation_file.exists():
            return annotation_file
    
    return None


def _load_annotations(annotation_file: Path, image_file: Path) -> Tuple[List[List[float]], List[int]]:
    """Load annotations from file."""
    boxes, class_labels = [], []
    
    try:
        if annotation_file.suffix == '.txt':
            # Assume YOLO format
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        boxes.append([float(x) for x in parts[1:]])
        
        elif annotation_file.suffix == '.json':
            # Load JSON annotations
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            # Get image dimensions
            image = cv2.imread(str(image_file))
            if image is not None:
                height, width = image.shape[:2]
                
                converter = AnnotationConverter()
                
                for ann in data.get('annotations', []):
                    # Convert to YOLO format
                    yolo_line = converter.coco_to_yolo(ann, width, height)
                    parts = yolo_line.split()
                    
                    class_labels.append(int(parts[0]))
                    boxes.append([float(x) for x in parts[1:]])
    
    except Exception as e:
        print(f"Error loading annotations from {annotation_file}: {e}")
    
    return boxes, class_labels


def _save_yolo_labels(label_file: Path, boxes: List[List[float]], class_labels: List[int]):
    """Save labels in YOLO format."""
    label_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(label_file, 'w') as f:
        for box, label in zip(boxes, class_labels):
            f.write(f"{label} {' '.join(map(str, box))}\n")


def _create_dataset_config(dataset_dir: Path, stats: Dict[str, int]):
    """Create dataset configuration files."""
    
    # Create YOLO dataset config
    config = {
        'path': str(dataset_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(AnnotationConverter.CLASS_MAPPING),
        'names': list(AnnotationConverter.CLASS_MAPPING.keys())
    }
    
    with open(dataset_dir / 'data.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create dataset info
    info = {
        'dataset_name': 'recipe_text_detection',
        'description': 'Text detection dataset for recipe ingredients',
        'format': 'YOLO',
        'classes': list(AnnotationConverter.CLASS_MAPPING.keys()),
        'statistics': stats,
        'creation_date': None,  # Add timestamp
        'preprocessing': {
            'format_classification': True,
            'adaptive_preprocessing': True,
            'data_augmentation': True
        }
    }
    
    with open(dataset_dir / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)


def analyze_dataset_distribution(dataset_dir: str) -> Dict[str, any]:
    """
    Analyze dataset distribution and characteristics.
    
    Args:
        dataset_dir: Path to dataset directory
        
    Returns:
        Dictionary with analysis results
    """
    dataset_path = Path(dataset_dir)
    
    analysis = {
        'splits': {},
        'class_distribution': defaultdict(int),
        'image_statistics': {},
        'annotation_quality': {}
    }
    
    # Analyze each split
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
        
        images = list((split_dir / 'images').glob('*'))
        labels = list((split_dir / 'labels').glob('*'))
        
        split_analysis = {
            'image_count': len(images),
            'label_count': len(labels),
            'coverage': len(labels) / len(images) if images else 0
        }
        
        # Analyze image dimensions
        if images:
            sample_images = random.sample(images, min(100, len(images)))
            widths, heights = [], []
            
            for img_file in sample_images:
                try:
                    image = cv2.imread(str(img_file))
                    if image is not None:
                        h, w = image.shape[:2]
                        widths.append(w)
                        heights.append(h)
                except Exception:
                    continue
            
            if widths and heights:
                split_analysis['image_stats'] = {
                    'mean_width': np.mean(widths),
                    'mean_height': np.mean(heights),
                    'width_std': np.std(widths),
                    'height_std': np.std(heights),
                    'min_width': min(widths),
                    'max_width': max(widths),
                    'min_height': min(heights),
                    'max_height': max(heights)
                }
        
        # Analyze class distribution
        split_classes = defaultdict(int)
        for label_file in labels:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            split_classes[class_id] += 1
                            analysis['class_distribution'][class_id] += 1
            except Exception:
                continue
        
        split_analysis['class_distribution'] = dict(split_classes)
        analysis['splits'][split] = split_analysis
    
    # Calculate overall statistics
    total_images = sum(split_data['image_count'] for split_data in analysis['splits'].values())
    total_annotations = sum(analysis['class_distribution'].values())
    
    analysis['overall'] = {
        'total_images': total_images,
        'total_annotations': total_annotations,
        'annotations_per_image': total_annotations / total_images if total_images > 0 else 0,
        'class_balance': _calculate_class_balance(analysis['class_distribution'])
    }
    
    return analysis


def _calculate_class_balance(class_distribution: Dict[int, int]) -> Dict[str, float]:
    """Calculate class balance metrics."""
    if not class_distribution:
        return {}
    
    total = sum(class_distribution.values())
    proportions = [count / total for count in class_distribution.values()]
    
    # Calculate balance metrics
    entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
    max_entropy = np.log2(len(class_distribution))
    balance_score = entropy / max_entropy if max_entropy > 0 else 0
    
    return {
        'entropy': entropy,
        'max_entropy': max_entropy,
        'balance_score': balance_score,
        'most_common_class_ratio': max(proportions),
        'least_common_class_ratio': min(proportions)
    }


def split_dataset(images_dir, labels_dir, output_dir, split_ratios=(0.7, 0.2, 0.1)):
    """
    Split dataset into train/val/test sets.
    
    Args:
        images_dir (str): Directory containing images
        labels_dir (str): Directory containing YOLO labels
        output_dir (str): Output directory for split dataset
        split_ratios (tuple): (train, val, test) ratios
    """
    import random
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    
    # Get all image files
    image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
    random.shuffle(image_files)
    
    # Calculate split indices
    total = len(image_files)
    train_end = int(total * split_ratios[0])
    val_end = train_end + int(total * split_ratios[1])
    
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    # Create directories and copy files
    for split_name, files in splits.items():
        img_dir = output_path / split_name / 'images'
        lbl_dir = output_path / split_name / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in files:
            # Copy image
            shutil.copy2(img_file, img_dir / img_file.name)
            
            # Copy corresponding label file
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, lbl_dir / label_file.name)
        
        print(f"{split_name}: {len(files)} images")


def resize_images(input_dir, output_dir, target_size=(640, 640)):
    """
    Resize images to target size for training.
    
    Args:
        input_dir (str): Input directory containing images
        output_dir (str): Output directory for resized images
        target_size (tuple): Target (width, height)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for img_file in input_path.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            img = cv2.imread(str(img_file))
            if img is not None:
                resized = cv2.resize(img, target_size)
                output_file = output_path / img_file.name
                cv2.imwrite(str(output_file), resized)


def convert_to_yolo_format(annotations, img_width, img_height):
    """
    Convert bounding box annotations to YOLO format.
    
    Args:
        annotations (list): List of [x1, y1, x2, y2, class_id] annotations
        img_width (int): Image width
        img_height (int): Image height
        
    Returns:
        list: YOLO format annotations [class_id, x_center, y_center, width, height]
    """
    yolo_annotations = []
    
    for ann in annotations:
        x1, y1, x2, y2, class_id = ann
        
        # Convert to YOLO format (normalized coordinates)
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        yolo_annotations.append([class_id, x_center, y_center, width, height])
    
    return yolo_annotations


def count_classes(labels_dir):
    """
    Count instances of each class in dataset.
    
    Args:
        labels_dir (str): Directory containing YOLO label files
        
    Returns:
        dict: Class counts
    """
    labels_path = Path(labels_dir)
    class_counts = {}
    
    for label_file in labels_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    return class_counts


def visualize_annotations(image_path, label_path, class_names, output_path=None):
    """
    Visualize YOLO annotations on image.
    
    Args:
        image_path (str): Path to image file
        label_path (str): Path to YOLO label file
        class_names (list): List of class names
        output_path (str): Path to save visualization
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    h, w = img.shape[:2]
    
    # Load annotations
    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    # Convert from YOLO format to pixel coordinates
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw class label
                    label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, img)
    else:
        cv2.imshow('Annotations', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()