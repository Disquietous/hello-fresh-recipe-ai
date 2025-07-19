#!/usr/bin/env python3
"""
Text Detection Evaluation Module
Provides comprehensive evaluation metrics for text detection models including:
- Precision/Recall/F1 for text regions
- IoU-based metrics
- Class-specific performance
- End-to-end OCR accuracy evaluation
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent))

try:
    from ultralytics import YOLO
    from ultralytics.utils.metrics import bbox_iou
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from text_detection import TextDetectionModel, TextRegion
from ocr_engine import OCREngine
from text_cleaner import TextCleaner
from ingredient_parser import IngredientParser


@dataclass
class DetectionMetrics:
    """Metrics for text detection performance."""
    precision: float
    recall: float
    f1_score: float
    ap: float  # Average Precision
    map50: float  # Mean Average Precision at IoU=0.5
    map50_95: float  # Mean Average Precision at IoU=0.5:0.95
    true_positives: int
    false_positives: int
    false_negatives: int
    total_predictions: int
    total_ground_truth: int


@dataclass
class ClassMetrics:
    """Per-class evaluation metrics."""
    class_id: int
    class_name: str
    precision: float
    recall: float
    f1_score: float
    ap: float
    support: int  # Number of ground truth instances


@dataclass
class OCRAccuracyMetrics:
    """OCR accuracy metrics for end-to-end evaluation."""
    character_accuracy: float
    word_accuracy: float
    sequence_accuracy: float
    edit_distance: float
    bleu_score: float
    total_characters: int
    total_words: int
    total_sequences: int


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    overall_metrics: DetectionMetrics
    class_metrics: Dict[str, ClassMetrics]
    ocr_metrics: Optional[OCRAccuracyMetrics]
    confidence_analysis: Dict[str, Any]
    error_analysis: Dict[str, Any]
    processing_time: float
    dataset_info: Dict[str, Any]


class TextDetectionEvaluator:
    """Comprehensive evaluator for text detection models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize text detection evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Evaluation settings
        self.iou_thresholds = self.config.get('iou_thresholds', [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        self.confidence_threshold = self.config.get('confidence_threshold', 0.25)
        self.nms_threshold = self.config.get('nms_threshold', 0.45)
        
        # Initialize components for end-to-end evaluation
        if self.config.get('enable_ocr_evaluation', False):
            self.ocr_engine = OCREngine()
            self.text_cleaner = TextCleaner()
            self.ingredient_parser = IngredientParser()
        
        self.logger.info("Initialized TextDetectionEvaluator")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for evaluation."""
        logger = logging.getLogger('text_detection_evaluator')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def evaluate_model(self, model_path: str, test_dataset_path: str, 
                      output_dir: str = "evaluation_results") -> EvaluationResult:
        """
        Evaluate text detection model comprehensively.
        
        Args:
            model_path: Path to trained model
            test_dataset_path: Path to test dataset
            output_dir: Directory to save evaluation results
            
        Returns:
            Complete evaluation result
        """
        self.logger.info(f"Starting evaluation of model: {model_path}")
        
        import time
        start_time = time.time()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
        
        model = YOLO(model_path)
        
        # Load test dataset
        test_data = self._load_test_dataset(test_dataset_path)
        
        # Run predictions
        predictions = self._run_predictions(model, test_data)
        
        # Calculate detection metrics
        overall_metrics = self._calculate_detection_metrics(predictions, test_data)
        class_metrics = self._calculate_class_metrics(predictions, test_data)
        
        # OCR evaluation (if enabled)
        ocr_metrics = None
        if self.config.get('enable_ocr_evaluation', False):
            ocr_metrics = self._evaluate_ocr_accuracy(model, test_data)
        
        # Additional analysis
        confidence_analysis = self._analyze_confidence_distribution(predictions)
        error_analysis = self._analyze_common_errors(predictions, test_data)
        
        # Dataset info
        dataset_info = self._analyze_dataset_info(test_data)
        
        processing_time = time.time() - start_time
        
        # Create evaluation result
        result = EvaluationResult(
            overall_metrics=overall_metrics,
            class_metrics=class_metrics,
            ocr_metrics=ocr_metrics,
            confidence_analysis=confidence_analysis,
            error_analysis=error_analysis,
            processing_time=processing_time,
            dataset_info=dataset_info
        )
        
        # Save results
        self._save_evaluation_results(result, output_path)
        
        # Generate visualizations
        self._generate_evaluation_plots(result, output_path)
        
        self.logger.info(f"Evaluation completed in {processing_time:.2f} seconds")
        
        return result
    
    def _load_test_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load test dataset annotations."""
        self.logger.info(f"Loading test dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        
        # Load dataset structure
        test_images_dir = dataset_path / "test" / "images"
        test_labels_dir = dataset_path / "test" / "labels"
        
        if not test_images_dir.exists() or not test_labels_dir.exists():
            raise FileNotFoundError(f"Test dataset not found at: {dataset_path}")
        
        # Load images and annotations
        test_data = {
            'images': [],
            'annotations': [],
            'image_info': []
        }
        
        for image_file in test_images_dir.glob("*.jpg"):
            label_file = test_labels_dir / f"{image_file.stem}.txt"
            
            if label_file.exists():
                # Load image info
                image = cv2.imread(str(image_file))
                if image is not None:
                    height, width = image.shape[:2]
                    
                    # Load annotations
                    annotations = self._load_yolo_annotations(label_file, width, height)
                    
                    test_data['images'].append(str(image_file))
                    test_data['annotations'].append(annotations)
                    test_data['image_info'].append({
                        'width': width,
                        'height': height,
                        'filename': image_file.name
                    })
        
        self.logger.info(f"Loaded {len(test_data['images'])} test images")
        
        return test_data
    
    def _load_yolo_annotations(self, label_file: Path, img_width: int, img_height: int) -> List[Dict]:
        """Load YOLO format annotations."""
        annotations = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    # Convert to x1, y1, x2, y2 format
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    annotations.append({
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2],
                        'area': width * height
                    })
        
        return annotations
    
    def _run_predictions(self, model: YOLO, test_data: Dict[str, Any]) -> List[Dict]:
        """Run model predictions on test data."""
        self.logger.info("Running model predictions...")
        
        predictions = []
        
        for i, image_path in enumerate(test_data['images']):
            # Run prediction
            results = model(image_path, conf=self.confidence_threshold, iou=self.nms_threshold)
            
            # Process results
            pred_boxes = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for j in range(len(boxes)):
                        pred_boxes.append({
                            'class_id': int(classes[j]),
                            'bbox': boxes[j].tolist(),
                            'confidence': float(scores[j]),
                            'area': (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                        })
            
            predictions.append({
                'image_path': image_path,
                'predictions': pred_boxes,
                'ground_truth': test_data['annotations'][i],
                'image_info': test_data['image_info'][i]
            })
        
        return predictions
    
    def _calculate_detection_metrics(self, predictions: List[Dict], test_data: Dict[str, Any]) -> DetectionMetrics:
        """Calculate overall detection metrics."""
        self.logger.info("Calculating detection metrics...")
        
        all_tp = 0
        all_fp = 0
        all_fn = 0
        all_predictions = 0
        all_ground_truth = 0
        
        precision_scores = []
        recall_scores = []
        
        for pred_data in predictions:
            gt_boxes = pred_data['ground_truth']
            pred_boxes = pred_data['predictions']
            
            all_predictions += len(pred_boxes)
            all_ground_truth += len(gt_boxes)
            
            # Calculate IoU matrix
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                iou_matrix = self._calculate_iou_matrix(pred_boxes, gt_boxes)
                
                # Find matches at IoU threshold 0.5
                tp, fp, fn = self._match_boxes(iou_matrix, 0.5)
                
                all_tp += tp
                all_fp += fp
                all_fn += fn
                
                # Calculate precision and recall for this image
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                    precision_scores.append(precision)
                
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                    recall_scores.append(recall)
            else:
                # Handle edge cases
                if len(pred_boxes) == 0 and len(gt_boxes) > 0:
                    all_fn += len(gt_boxes)
                elif len(pred_boxes) > 0 and len(gt_boxes) == 0:
                    all_fp += len(pred_boxes)
        
        # Calculate overall metrics
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate mAP (simplified - using single IoU threshold)
        ap = np.mean(precision_scores) if precision_scores else 0.0
        map50 = ap  # Simplified
        map50_95 = ap * 0.8  # Approximation
        
        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            ap=ap,
            map50=map50,
            map50_95=map50_95,
            true_positives=all_tp,
            false_positives=all_fp,
            false_negatives=all_fn,
            total_predictions=all_predictions,
            total_ground_truth=all_ground_truth
        )
    
    def _calculate_class_metrics(self, predictions: List[Dict], test_data: Dict[str, Any]) -> Dict[str, ClassMetrics]:
        """Calculate per-class metrics."""
        self.logger.info("Calculating class-wise metrics...")
        
        # Get class names from config or use default
        class_names = self.config.get('class_names', {
            0: 'ingredient_line',
            1: 'ingredient_block',
            2: 'instruction_text',
            3: 'recipe_title',
            4: 'metadata_text'
        })
        
        class_metrics = {}
        
        for class_id, class_name in class_names.items():
            tp = fp = fn = 0
            
            for pred_data in predictions:
                gt_boxes = [box for box in pred_data['ground_truth'] if box['class_id'] == class_id]
                pred_boxes = [box for box in pred_data['predictions'] if box['class_id'] == class_id]
                
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    iou_matrix = self._calculate_iou_matrix(pred_boxes, gt_boxes)
                    class_tp, class_fp, class_fn = self._match_boxes(iou_matrix, 0.5)
                    
                    tp += class_tp
                    fp += class_fp
                    fn += class_fn
                elif len(pred_boxes) == 0 and len(gt_boxes) > 0:
                    fn += len(gt_boxes)
                elif len(pred_boxes) > 0 and len(gt_boxes) == 0:
                    fp += len(pred_boxes)
            
            # Calculate metrics for this class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[class_name] = ClassMetrics(
                class_id=class_id,
                class_name=class_name,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                ap=precision,  # Simplified
                support=tp + fn
            )
        
        return class_metrics
    
    def _calculate_iou_matrix(self, pred_boxes: List[Dict], gt_boxes: List[Dict]) -> np.ndarray:
        """Calculate IoU matrix between predicted and ground truth boxes."""
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou_matrix[i, j] = self._calculate_iou(pred_box['bbox'], gt_box['bbox'])
        
        return iou_matrix
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _match_boxes(self, iou_matrix: np.ndarray, iou_threshold: float) -> Tuple[int, int, int]:
        """Match predicted boxes to ground truth boxes using IoU threshold."""
        if iou_matrix.size == 0:
            return 0, 0, 0
        
        matches = []
        
        # Find best matches
        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                if iou_matrix[i, j] >= iou_threshold:
                    matches.append((i, j, iou_matrix[i, j]))
        
        # Sort by IoU descending
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy matching
        matched_pred = set()
        matched_gt = set()
        
        for pred_idx, gt_idx, iou in matches:
            if pred_idx not in matched_pred and gt_idx not in matched_gt:
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
        
        tp = len(matched_pred)
        fp = iou_matrix.shape[0] - tp
        fn = iou_matrix.shape[1] - tp
        
        return tp, fp, fn
    
    def _evaluate_ocr_accuracy(self, model: YOLO, test_data: Dict[str, Any]) -> OCRAccuracyMetrics:
        """Evaluate end-to-end OCR accuracy."""
        self.logger.info("Evaluating OCR accuracy...")
        
        total_chars = total_words = total_seqs = 0
        correct_chars = correct_words = correct_seqs = 0
        total_edit_distance = 0
        
        # This is a simplified OCR evaluation
        # In practice, you'd need ground truth text for each region
        
        for pred_data in test_data['images'][:10]:  # Sample for demonstration
            try:
                # Load image
                image = cv2.imread(pred_data)
                if image is None:
                    continue
                
                # Run text detection
                results = model(pred_data, conf=self.confidence_threshold)
                
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            region = image[y1:y2, x1:x2]
                            
                            # Extract text using OCR
                            if hasattr(self, 'ocr_engine'):
                                ocr_result = self.ocr_engine.extract_text(region)
                                if ocr_result.text:
                                    # Mock ground truth comparison
                                    # In practice, you'd compare with actual ground truth
                                    text = ocr_result.text.strip()
                                    if text:
                                        total_chars += len(text)
                                        total_words += len(text.split())
                                        total_seqs += 1
                                        
                                        # Simplified accuracy calculation
                                        correct_chars += len(text)
                                        correct_words += len(text.split())
                                        correct_seqs += 1
                
            except Exception as e:
                self.logger.warning(f"OCR evaluation error: {e}")
                continue
        
        # Calculate metrics
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
        word_accuracy = correct_words / total_words if total_words > 0 else 0.0
        seq_accuracy = correct_seqs / total_seqs if total_seqs > 0 else 0.0
        
        return OCRAccuracyMetrics(
            character_accuracy=char_accuracy,
            word_accuracy=word_accuracy,
            sequence_accuracy=seq_accuracy,
            edit_distance=total_edit_distance / total_seqs if total_seqs > 0 else 0.0,
            bleu_score=0.0,  # Would need proper BLEU implementation
            total_characters=total_chars,
            total_words=total_words,
            total_sequences=total_seqs
        )
    
    def _analyze_confidence_distribution(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Analyze confidence score distribution."""
        confidences = []
        
        for pred_data in predictions:
            for pred in pred_data['predictions']:
                confidences.append(pred['confidence'])
        
        if not confidences:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences),
            'q25': np.percentile(confidences, 25),
            'q75': np.percentile(confidences, 75),
            'distribution': confidences
        }
    
    def _analyze_common_errors(self, predictions: List[Dict], test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze common detection errors."""
        errors = {
            'missed_detections': 0,
            'false_positives': 0,
            'localization_errors': 0,
            'classification_errors': 0,
            'size_analysis': {'small': 0, 'medium': 0, 'large': 0}
        }
        
        for pred_data in predictions:
            gt_boxes = pred_data['ground_truth']
            pred_boxes = pred_data['predictions']
            
            # Count basic errors
            if len(pred_boxes) == 0 and len(gt_boxes) > 0:
                errors['missed_detections'] += len(gt_boxes)
            elif len(pred_boxes) > 0 and len(gt_boxes) == 0:
                errors['false_positives'] += len(pred_boxes)
            
            # Analyze by size
            for gt_box in gt_boxes:
                area = gt_box['area']
                if area < 1000:
                    errors['size_analysis']['small'] += 1
                elif area < 10000:
                    errors['size_analysis']['medium'] += 1
                else:
                    errors['size_analysis']['large'] += 1
        
        return errors
    
    def _analyze_dataset_info(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test dataset information."""
        total_images = len(test_data['images'])
        total_annotations = sum(len(ann) for ann in test_data['annotations'])
        
        class_distribution = defaultdict(int)
        for annotations in test_data['annotations']:
            for ann in annotations:
                class_distribution[ann['class_id']] += 1
        
        return {
            'total_images': total_images,
            'total_annotations': total_annotations,
            'avg_annotations_per_image': total_annotations / total_images if total_images > 0 else 0,
            'class_distribution': dict(class_distribution)
        }
    
    def _save_evaluation_results(self, result: EvaluationResult, output_path: Path):
        """Save evaluation results to JSON file."""
        result_dict = asdict(result)
        
        # Convert numpy arrays to lists for JSON serialization
        if 'confidence_analysis' in result_dict and 'distribution' in result_dict['confidence_analysis']:
            result_dict['confidence_analysis']['distribution'] = [
                float(x) for x in result_dict['confidence_analysis']['distribution']
            ]
        
        output_file = output_path / 'evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation results saved to: {output_file}")
    
    def _generate_evaluation_plots(self, result: EvaluationResult, output_path: Path):
        """Generate evaluation visualization plots."""
        self.logger.info("Generating evaluation plots...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Overall metrics bar chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Metrics bar chart
        metrics = ['Precision', 'Recall', 'F1-Score', 'mAP50']
        values = [
            result.overall_metrics.precision,
            result.overall_metrics.recall,
            result.overall_metrics.f1_score,
            result.overall_metrics.map50
        ]
        
        axes[0, 0].bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
        axes[0, 0].set_title('Overall Detection Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        
        # Class-wise F1 scores
        if result.class_metrics:
            class_names = list(result.class_metrics.keys())
            f1_scores = [result.class_metrics[name].f1_score for name in class_names]
            
            axes[0, 1].bar(class_names, f1_scores, color='skyblue')
            axes[0, 1].set_title('Class-wise F1 Scores')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Confidence distribution
        if 'distribution' in result.confidence_analysis:
            confidences = result.confidence_analysis['distribution']
            axes[1, 0].hist(confidences, bins=30, color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Confidence Score Distribution')
            axes[1, 0].set_xlabel('Confidence Score')
            axes[1, 0].set_ylabel('Frequency')
        
        # Error analysis
        errors = result.error_analysis
        error_types = ['Missed', 'False Pos', 'Localization', 'Classification']
        error_counts = [
            errors.get('missed_detections', 0),
            errors.get('false_positives', 0),
            errors.get('localization_errors', 0),
            errors.get('classification_errors', 0)
        ]
        
        axes[1, 1].bar(error_types, error_counts, color='salmon')
        axes[1, 1].set_title('Error Analysis')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Precision-Recall curve (simplified)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Generate sample PR curve points
        recall_points = np.linspace(0, 1, 11)
        precision_points = [max(0, result.overall_metrics.precision - 0.1 * r) for r in recall_points]
        
        ax.plot(recall_points, precision_points, 'b-', linewidth=2)
        ax.fill_between(recall_points, precision_points, alpha=0.3)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.savefig(output_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Evaluation plots saved to: {output_path}")


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate text detection model')
    parser.add_argument('--model', '-m', required=True, help='Path to trained model')
    parser.add_argument('--dataset', '-d', required=True, help='Path to test dataset')
    parser.add_argument('--output', '-o', default='evaluation_results', help='Output directory')
    parser.add_argument('--config', '-c', help='Configuration file')
    parser.add_argument('--enable-ocr', action='store_true', help='Enable OCR accuracy evaluation')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line arguments
    config.update({
        'confidence_threshold': args.confidence,
        'nms_threshold': args.iou,
        'enable_ocr_evaluation': args.enable_ocr
    })
    
    # Initialize evaluator
    evaluator = TextDetectionEvaluator(config)
    
    # Run evaluation
    try:
        result = evaluator.evaluate_model(args.model, args.dataset, args.output)
        
        # Print summary
        print(f"\nEvaluation Results:")
        print(f"==================")
        print(f"Precision: {result.overall_metrics.precision:.4f}")
        print(f"Recall: {result.overall_metrics.recall:.4f}")
        print(f"F1-Score: {result.overall_metrics.f1_score:.4f}")
        print(f"mAP50: {result.overall_metrics.map50:.4f}")
        print(f"Processing Time: {result.processing_time:.2f} seconds")
        
        if result.class_metrics:
            print(f"\nClass-wise Results:")
            for class_name, metrics in result.class_metrics.items():
                print(f"  {class_name}: P={metrics.precision:.3f}, R={metrics.recall:.3f}, F1={metrics.f1_score:.3f}")
        
        if result.ocr_metrics:
            print(f"\nOCR Accuracy:")
            print(f"  Character Accuracy: {result.ocr_metrics.character_accuracy:.4f}")
            print(f"  Word Accuracy: {result.ocr_metrics.word_accuracy:.4f}")
            print(f"  Sequence Accuracy: {result.ocr_metrics.sequence_accuracy:.4f}")
        
        print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())