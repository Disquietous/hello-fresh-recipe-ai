#!/usr/bin/env python3
"""
YOLOv8-based text detection module for recipe ingredient extraction.
Detects text regions in recipe images and extracts bounding boxes for OCR processing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None


@dataclass
class TextRegion:
    """Detected text region with metadata."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    area: int
    text_content: Optional[str] = None


class TextDetector:
    """YOLOv8-based text detection for recipe images."""
    
    # Class mapping for text detection
    CLASS_NAMES = {
        0: 'ingredient_line',
        1: 'ingredient_block', 
        2: 'instruction_text',
        3: 'recipe_title',
        4: 'metadata_text'
    }
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.25, 
                 device: str = "auto"):
        """
        Initialize text detector.
        
        Args:
            model_path: Path to YOLOv8 model file
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = self._setup_device(device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = None
        self._load_model()
        
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load YOLOv8 model."""
        if YOLO is None:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
        
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.logger.info(f"Loaded YOLOv8 model: {self.model_path} on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_path}: {e}")
            raise
    
    def detect_text_regions(self, image: np.ndarray, 
                           target_classes: Optional[List[int]] = None) -> List[TextRegion]:
        """
        Detect text regions in image.
        
        Args:
            image: Input image as numpy array
            target_classes: List of class IDs to detect (None for all)
            
        Returns:
            List of detected text regions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Run inference
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        text_regions = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            # Extract detection data
            xyxy = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            conf = boxes.conf.cpu().numpy()  # confidence scores
            cls = boxes.cls.cpu().numpy()    # class IDs
            
            for i in range(len(xyxy)):
                class_id = int(cls[i])
                confidence = float(conf[i])
                
                # Filter by target classes if specified
                if target_classes and class_id not in target_classes:
                    continue
                
                # Extract bounding box
                x1, y1, x2, y2 = map(int, xyxy[i])
                
                # Calculate area
                area = (x2 - x1) * (y2 - y1)
                
                # Get class name
                class_name = self.CLASS_NAMES.get(class_id, f"class_{class_id}")
                
                text_region = TextRegion(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    area=area
                )
                
                text_regions.append(text_region)
        
        # Sort by confidence (highest first)
        text_regions.sort(key=lambda x: x.confidence, reverse=True)
        
        self.logger.info(f"Detected {len(text_regions)} text regions")
        return text_regions
    
    def extract_text_region_image(self, image: np.ndarray, 
                                 text_region: TextRegion,
                                 padding: int = 5) -> np.ndarray:
        """
        Extract text region from image as separate image.
        
        Args:
            image: Source image
            text_region: Text region to extract
            padding: Padding around bounding box
            
        Returns:
            Extracted region image
        """
        x1, y1, x2, y2 = text_region.bbox
        h, w = image.shape[:2]
        
        # Add padding and ensure within image bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Extract region
        region_image = image[y1:y2, x1:x2]
        
        return region_image
    
    def detect_ingredient_regions(self, image: np.ndarray) -> List[TextRegion]:
        """
        Detect only ingredient-related text regions.
        
        Args:
            image: Input image
            
        Returns:
            List of ingredient text regions
        """
        # Focus on ingredient classes
        ingredient_classes = [0, 1]  # ingredient_line, ingredient_block
        return self.detect_text_regions(image, target_classes=ingredient_classes)
    
    def visualize_detections(self, image: np.ndarray, text_regions: List[TextRegion],
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detected text regions on image.
        
        Args:
            image: Input image
            text_regions: Detected text regions
            save_path: Optional path to save visualization
            
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        # Define colors for different classes
        colors = {
            0: (0, 255, 0),    # ingredient_line - green
            1: (0, 255, 255),  # ingredient_block - yellow
            2: (255, 0, 0),    # instruction_text - blue
            3: (255, 0, 255),  # recipe_title - magenta
            4: (0, 165, 255),  # metadata_text - orange
        }
        
        for region in text_regions:
            x1, y1, x2, y2 = region.bbox
            color = colors.get(region.class_id, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{region.class_name}: {region.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for label
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            self.logger.info(f"Saved visualization to {save_path}")
        
        return vis_image
    
    def filter_regions_by_size(self, text_regions: List[TextRegion],
                              min_area: int = 100, max_area: Optional[int] = None) -> List[TextRegion]:
        """
        Filter text regions by size.
        
        Args:
            text_regions: List of text regions
            min_area: Minimum area threshold
            max_area: Maximum area threshold (None for no limit)
            
        Returns:
            Filtered text regions
        """
        filtered = []
        
        for region in text_regions:
            if region.area < min_area:
                continue
            if max_area and region.area > max_area:
                continue
            filtered.append(region)
        
        self.logger.info(f"Filtered {len(text_regions)} -> {len(filtered)} regions by size")
        return filtered
    
    def filter_regions_by_confidence(self, text_regions: List[TextRegion],
                                   min_confidence: float = 0.5) -> List[TextRegion]:
        """
        Filter text regions by confidence.
        
        Args:
            text_regions: List of text regions
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered text regions
        """
        filtered = [r for r in text_regions if r.confidence >= min_confidence]
        self.logger.info(f"Filtered {len(text_regions)} -> {len(filtered)} regions by confidence")
        return filtered
    
    def merge_overlapping_regions(self, text_regions: List[TextRegion],
                                 iou_threshold: float = 0.5) -> List[TextRegion]:
        """
        Merge overlapping text regions.
        
        Args:
            text_regions: List of text regions
            iou_threshold: IoU threshold for merging
            
        Returns:
            Merged text regions
        """
        if len(text_regions) <= 1:
            return text_regions
        
        merged = []
        used = set()
        
        for i, region1 in enumerate(text_regions):
            if i in used:
                continue
                
            candidates = [region1]
            used.add(i)
            
            for j, region2 in enumerate(text_regions[i+1:], i+1):
                if j in used:
                    continue
                
                iou = self._calculate_iou(region1.bbox, region2.bbox)
                if iou > iou_threshold and region1.class_id == region2.class_id:
                    candidates.append(region2)
                    used.add(j)
            
            # Merge candidates
            if len(candidates) == 1:
                merged.append(candidates[0])
            else:
                merged_region = self._merge_regions(candidates)
                merged.append(merged_region)
        
        self.logger.info(f"Merged {len(text_regions)} -> {len(merged)} regions")
        return merged
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0
        
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_regions(self, regions: List[TextRegion]) -> TextRegion:
        """Merge multiple regions into one."""
        # Calculate merged bounding box
        x1_min = min(r.bbox[0] for r in regions)
        y1_min = min(r.bbox[1] for r in regions)
        x2_max = max(r.bbox[2] for r in regions)
        y2_max = max(r.bbox[3] for r in regions)
        
        # Use highest confidence and most common class
        best_region = max(regions, key=lambda r: r.confidence)
        merged_area = (x2_max - x1_min) * (y2_max - y1_min)
        
        return TextRegion(
            bbox=(x1_min, y1_min, x2_max, y2_max),
            confidence=best_region.confidence,
            class_id=best_region.class_id,
            class_name=best_region.class_name,
            area=merged_area
        )


def main():
    """Example usage of text detector."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize detector
    detector = TextDetector(model_path="yolov8n.pt", confidence_threshold=0.25)
    
    print("Text Detection System")
    print("Usage examples:")
    print("1. Detect all text regions:")
    print("   regions = detector.detect_text_regions(image)")
    print("2. Detect only ingredients:")
    print("   ingredients = detector.detect_ingredient_regions(image)")
    print("3. Extract text region:")
    print("   region_img = detector.extract_text_region_image(image, region)")
    print("4. Visualize detections:")
    print("   vis_img = detector.visualize_detections(image, regions, 'output.jpg')")


if __name__ == "__main__":
    main()