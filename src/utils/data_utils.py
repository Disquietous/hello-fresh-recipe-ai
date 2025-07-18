"""
Data utilities for HelloFresh Recipe AI project.
Functions for data preprocessing, augmentation, and validation.
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import yaml


def validate_dataset_structure(data_dir):
    """
    Validate YOLO dataset directory structure.
    
    Args:
        data_dir (str): Path to dataset directory
        
    Returns:
        bool: True if structure is valid
    """
    data_path = Path(data_dir)
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    
    for req_dir in required_dirs:
        dir_path = data_path / req_dir
        if not dir_path.exists():
            print(f"Missing required directory: {dir_path}")
            return False
    
    print("Dataset structure validation passed!")
    return True


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