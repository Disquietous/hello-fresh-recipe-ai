#!/usr/bin/env python3
"""
Training script for YOLOv8 text detection on recipe images.
Optimized for detecting ingredient text regions in various recipe formats.
"""

import os
import sys
import yaml
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent))

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import torch
    import cv2
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import seaborn as sns
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

# Import our utilities
from utils.data_utils import validate_dataset_structure, analyze_dataset_distribution
from utils.text_augmentation import TextAugmentationPipeline
from utils.annotation_utils import AnnotationValidator


class TextDetectionTrainer:
    """YOLOv8 trainer optimized for text detection."""
    
    def __init__(self, config_path: str, experiment_name: Optional[str] = None):
        """
        Initialize text detection trainer.
        
        Args:
            config_path: Path to training configuration YAML file
            experiment_name: Optional experiment name override
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Override experiment name if provided
        if experiment_name:
            self.config['name'] = experiment_name
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize training components
        self.model = None
        self.training_results = {}
        self.validation_results = {}
        
        # Text-specific components
        self.text_augmenter = TextAugmentationPipeline()
        self.annotation_validator = AnnotationValidator()
        
        self.logger.info(f"Initialized TextDetectionTrainer with config: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load config from {self.config_path}: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for training."""
        logger = logging.getLogger('text_detection_trainer')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.config.get('log_file'):
            file_handler = logging.FileHandler(self.config['log_file'])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def validate_dataset(self) -> bool:
        """Validate dataset structure and quality."""
        self.logger.info("Validating dataset structure...")
        
        dataset_path = Path(self.config['path'])
        
        # Check if dataset exists
        if not dataset_path.exists():
            self.logger.error(f"Dataset path does not exist: {dataset_path}")
            return False
        
        # Validate YOLO dataset structure
        if not validate_dataset_structure(str(dataset_path), 'yolo'):
            self.logger.error("Dataset structure validation failed")
            return False
        
        # Analyze dataset distribution
        self.logger.info("Analyzing dataset distribution...")
        analysis = analyze_dataset_distribution(str(dataset_path))
        
        # Log dataset statistics
        self.logger.info(f"Dataset Analysis:")
        self.logger.info(f"  Total images: {analysis['overall']['total_images']}")
        self.logger.info(f"  Total annotations: {analysis['overall']['total_annotations']}")
        self.logger.info(f"  Annotations per image: {analysis['overall']['annotations_per_image']:.2f}")
        
        # Check splits
        for split, data in analysis['splits'].items():
            self.logger.info(f"  {split}: {data['image_count']} images, {data['label_count']} labels")
        
        # Check class distribution
        self.logger.info("Class distribution:")
        for class_id, count in analysis['class_distribution'].items():
            class_name = self.config['names'].get(class_id, f"class_{class_id}")
            percentage = count / analysis['overall']['total_annotations'] * 100
            self.logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Check for potential issues
        issues = []
        
        # Low annotation density
        if analysis['overall']['annotations_per_image'] < 2:
            issues.append("Low annotation density (< 2 annotations per image)")
        
        # Class imbalance
        if analysis['overall']['class_balance']['balance_score'] < 0.5:
            issues.append("Significant class imbalance detected")
        
        # Small dataset
        if analysis['overall']['total_images'] < 100:
            issues.append("Very small dataset (< 100 images)")
        
        if issues:
            self.logger.warning("Dataset issues detected:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        
        return True
    
    def setup_model(self) -> YOLO:
        """Setup YOLOv8 model for text detection."""
        self.logger.info("Setting up YOLOv8 model...")
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
        
        # Load base model
        model_path = self.config.get('model', 'yolov8n.pt')
        self.model = YOLO(model_path)
        
        # Modify model for text detection if needed
        if hasattr(self.model, 'model'):
            # Update number of classes
            nc = self.config['nc']
            if hasattr(self.model.model, 'nc'):
                self.model.model.nc = nc
            
            # Update class names
            self.model.names = self.config['names']
        
        self.logger.info(f"Loaded model: {model_path}")
        self.logger.info(f"Classes: {self.config['nc']}")
        self.logger.info(f"Class names: {list(self.config['names'].values())}")
        
        return self.model
    
    def prepare_text_augmentations(self) -> Dict[str, Any]:
        """Prepare text-specific augmentation settings."""
        self.logger.info("Preparing text-specific augmentations...")
        
        # Text detection optimized augmentations
        augmentations = {
            # Reduced geometric transformations for text
            'degrees': self.config.get('degrees', 5.0),  # Minimal rotation
            'translate': self.config.get('translate', 0.1),  # Minimal translation
            'scale': self.config.get('scale', 0.2),  # Minimal scaling
            'shear': self.config.get('shear', 2.0),  # Minimal shear
            'perspective': self.config.get('perspective', 0.0001),  # Minimal perspective
            'flipud': 0.0,  # No vertical flip for text
            'fliplr': 0.0,  # No horizontal flip for text
            
            # Photometric augmentations
            'hsv_h': self.config.get('hsv_h', 0.015),
            'hsv_s': self.config.get('hsv_s', 0.7),
            'hsv_v': self.config.get('hsv_v', 0.4),
            
            # Mosaic and mixup
            'mosaic': self.config.get('mosaic', 0.8),
            'mixup': self.config.get('mixup', 0.1),
            'copy_paste': self.config.get('copy_paste', 0.0),
        }
        
        # Add text-specific augmentations
        text_config = self.config.get('text_detection', {})
        if text_config.get('rotation_augment', True):
            rotation_range = text_config.get('rotation_range', [-10, 10])
            augmentations['degrees'] = max(abs(rotation_range[0]), abs(rotation_range[1]))
        
        self.logger.info(f"Augmentation settings: {augmentations}")
        
        return augmentations
    
    def train(self, resume: bool = False) -> Dict[str, Any]:
        """
        Train YOLOv8 model for text detection.
        
        Args:
            resume: Whether to resume from last checkpoint
            
        Returns:
            Training results dictionary
        """
        self.logger.info("Starting text detection training...")
        
        # Validate dataset
        if not self.validate_dataset():
            raise ValueError("Dataset validation failed")
        
        # Setup model
        self.setup_model()
        
        # Prepare augmentations
        augmentations = self.prepare_text_augmentations()
        
        # Training parameters
        train_params = {
            'data': str(self.config_path),
            'epochs': self.config.get('epochs', 200),
            'batch': self.config.get('batch', 16),
            'imgsz': self.config.get('imgsz', 640),
            'device': self.config.get('device', ''),
            'workers': self.config.get('workers', 8),
            'project': self.config.get('project', 'runs/text_detection'),
            'name': self.config.get('name', 'recipe_text_v1'),
            'exist_ok': True,
            'pretrained': self.config.get('pretrained', True),
            'optimizer': self.config.get('optimizer', 'AdamW'),
            'verbose': self.config.get('verbose', True),
            'seed': self.config.get('seed', 42),
            'deterministic': self.config.get('deterministic', False),
            'single_cls': self.config.get('single_cls', False),
            'rect': self.config.get('rect', False),
            'cos_lr': self.config.get('cos_lr', True),
            'close_mosaic': self.config.get('close_mosaic', 10),
            'resume': resume,
            'amp': self.config.get('amp', True),
            'fraction': self.config.get('fraction', 1.0),
            'profile': self.config.get('profile', False),
            'plots': self.config.get('plots', True),
            'save': self.config.get('save', True),
            'save_period': self.config.get('save_period', 10),
            'cache': self.config.get('cache', True),
            'val': self.config.get('val', True),
            'patience': self.config.get('patience', 50),
            'freeze': self.config.get('freeze', None),
            'lr0': self.config.get('lr0', 0.001),
            'lrf': self.config.get('lrf', 0.1),
            'momentum': self.config.get('momentum', 0.937),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'warmup_epochs': self.config.get('warmup_epochs', 3),
            'warmup_momentum': self.config.get('warmup_momentum', 0.8),
            'warmup_bias_lr': self.config.get('warmup_bias_lr', 0.1),
            'box': self.config.get('box', 7.5),
            'cls': self.config.get('cls', 0.5),
            'dfl': self.config.get('dfl', 1.5),
            'pose': self.config.get('pose', 12.0),
            'kobj': self.config.get('kobj', 1.0),
            'label_smoothing': self.config.get('label_smoothing', 0.0),
            'nbs': self.config.get('nbs', 64),
            'overlap_mask': self.config.get('overlap_mask', True),
            'mask_ratio': self.config.get('mask_ratio', 4),
            'dropout': self.config.get('dropout', 0.0),
            'val_conf': self.config.get('conf', 0.001),
            'val_iou': self.config.get('iou', 0.6),
            'max_det': self.config.get('max_det', 300),
        }
        
        # Update with augmentation parameters
        train_params.update(augmentations)
        
        # Log training parameters
        self.logger.info("Training parameters:")
        for key, value in train_params.items():
            self.logger.info(f"  {key}: {value}")
        
        # Start training
        start_time = time.time()
        
        try:
            results = self.model.train(**train_params)
            
            training_time = time.time() - start_time
            
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Store results
            self.training_results = {
                'training_time': training_time,
                'best_fitness': results.best_fitness if hasattr(results, 'best_fitness') else None,
                'best_epoch': results.best_epoch if hasattr(results, 'best_epoch') else None,
                'model_path': results.save_dir if hasattr(results, 'save_dir') else None,
                'config_used': train_params
            }
            
            return self.training_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def validate_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate trained model on test dataset.
        
        Args:
            model_path: Path to trained model (uses best from training if None)
            
        Returns:
            Validation results dictionary
        """
        self.logger.info("Validating trained model...")
        
        # Load model
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available for validation")
        
        # Validation parameters
        val_params = {
            'data': str(self.config_path),
            'imgsz': self.config.get('imgsz', 640),
            'batch': self.config.get('batch', 16),
            'device': self.config.get('device', ''),
            'workers': self.config.get('workers', 8),
            'verbose': self.config.get('verbose', True),
            'save_json': self.config.get('save_json', True),
            'save_hybrid': self.config.get('save_hybrid', False),
            'conf': self.config.get('conf', 0.001),
            'iou': self.config.get('iou', 0.6),
            'max_det': self.config.get('max_det', 300),
            'half': self.config.get('half', False),
            'dnn': self.config.get('dnn', False),
            'plots': self.config.get('plots', True),
            'rect': self.config.get('rect', False),
            'split': 'val'
        }
        
        try:
            # Run validation
            results = model.val(**val_params)
            
            # Extract metrics
            metrics = {
                'mAP50': results.box.map50 if hasattr(results, 'box') else None,
                'mAP50_95': results.box.map if hasattr(results, 'box') else None,
                'precision': results.box.mp if hasattr(results, 'box') else None,
                'recall': results.box.mr if hasattr(results, 'box') else None,
                'f1': results.box.f1 if hasattr(results, 'box') else None,
            }
            
            # Class-wise metrics
            if hasattr(results, 'box') and hasattr(results.box, 'maps'):
                class_metrics = {}
                for i, (class_id, class_name) in enumerate(self.config['names'].items()):
                    if i < len(results.box.maps):
                        class_metrics[class_name] = {
                            'mAP50': results.box.maps[i],
                            'ap50': results.box.ap50[i] if hasattr(results.box, 'ap50') else None,
                            'ap': results.box.ap[i] if hasattr(results.box, 'ap') else None
                        }
                
                metrics['class_metrics'] = class_metrics
            
            self.validation_results = metrics
            
            # Log results
            self.logger.info("Validation Results:")
            self.logger.info(f"  mAP50: {metrics['mAP50']:.4f}")
            self.logger.info(f"  mAP50-95: {metrics['mAP50_95']:.4f}")
            self.logger.info(f"  Precision: {metrics['precision']:.4f}")
            self.logger.info(f"  Recall: {metrics['recall']:.4f}")
            self.logger.info(f"  F1: {metrics['f1']:.4f}")
            
            if 'class_metrics' in metrics:
                self.logger.info("Class-wise metrics:")
                for class_name, class_metrics in metrics['class_metrics'].items():
                    self.logger.info(f"  {class_name}: mAP50={class_metrics['mAP50']:.4f}")
            
            return self.validation_results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise
    
    def export_model(self, model_path: Optional[str] = None, formats: List[str] = None) -> Dict[str, str]:
        """
        Export trained model to various formats.
        
        Args:
            model_path: Path to trained model
            formats: List of export formats
            
        Returns:
            Dictionary mapping format to export path
        """
        self.logger.info("Exporting trained model...")
        
        # Default export formats
        if formats is None:
            formats = self.config.get('export', {}).get('format', ['pt', 'onnx'])
        
        # Load model
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available for export")
        
        export_paths = {}
        
        for format_name in formats:
            try:
                self.logger.info(f"Exporting to {format_name}...")
                
                export_params = {
                    'format': format_name,
                    'imgsz': self.config.get('imgsz', 640),
                    'half': self.config.get('export', {}).get('half', False),
                    'dynamic': self.config.get('export', {}).get('dynamic', False),
                    'simplify': self.config.get('export', {}).get('simplify', True),
                    'opset': self.config.get('export', {}).get('opset', 17),
                    'workspace': self.config.get('export', {}).get('workspace', 4),
                    'nms': self.config.get('export', {}).get('nms', True),
                    'optimize': self.config.get('export', {}).get('optimize', True),
                }
                
                exported_path = model.export(**export_params)
                export_paths[format_name] = str(exported_path)
                
                self.logger.info(f"Exported {format_name} to: {exported_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to export {format_name}: {e}")
                continue
        
        return export_paths
    
    def save_training_summary(self, output_dir: str):
        """Save comprehensive training summary."""
        self.logger.info("Saving training summary...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Compile summary
        summary = {
            'experiment_name': self.config.get('name', 'recipe_text_v1'),
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'training_results': self.training_results,
            'validation_results': self.validation_results,
            'dataset_info': {
                'path': str(self.config['path']),
                'classes': self.config['names'],
                'num_classes': self.config['nc']
            }
        }
        
        # Save as JSON
        summary_path = output_path / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save config copy
        config_path = output_path / 'config_used.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        self.logger.info(f"Training summary saved to: {summary_path}")
        self.logger.info(f"Config saved to: {config_path}")
    
    def create_training_report(self, output_dir: str):
        """Create comprehensive training report with visualizations."""
        self.logger.info("Creating training report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate report
        report_lines = [
            "# Text Detection Training Report",
            f"**Experiment:** {self.config.get('name', 'recipe_text_v1')}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Configuration",
            f"- Model: {self.config.get('model', 'yolov8n.pt')}",
            f"- Epochs: {self.config.get('epochs', 200)}",
            f"- Batch Size: {self.config.get('batch', 16)}",
            f"- Image Size: {self.config.get('imgsz', 640)}",
            f"- Classes: {self.config.get('nc', 5)}",
            "",
            "## Classes",
        ]
        
        for class_id, class_name in self.config['names'].items():
            report_lines.append(f"- {class_id}: {class_name}")
        
        report_lines.extend([
            "",
            "## Training Results",
        ])
        
        if self.training_results:
            report_lines.extend([
                f"- Training Time: {self.training_results.get('training_time', 'N/A'):.2f} seconds",
                f"- Best Fitness: {self.training_results.get('best_fitness', 'N/A')}",
                f"- Best Epoch: {self.training_results.get('best_epoch', 'N/A')}",
            ])
        
        report_lines.extend([
            "",
            "## Validation Results",
        ])
        
        if self.validation_results:
            report_lines.extend([
                f"- mAP50: {self.validation_results.get('mAP50', 'N/A'):.4f}",
                f"- mAP50-95: {self.validation_results.get('mAP50_95', 'N/A'):.4f}",
                f"- Precision: {self.validation_results.get('precision', 'N/A'):.4f}",
                f"- Recall: {self.validation_results.get('recall', 'N/A'):.4f}",
                f"- F1: {self.validation_results.get('f1', 'N/A'):.4f}",
            ])
            
            if 'class_metrics' in self.validation_results:
                report_lines.extend([
                    "",
                    "## Class-wise Performance",
                ])
                
                for class_name, metrics in self.validation_results['class_metrics'].items():
                    report_lines.append(f"- {class_name}: mAP50={metrics.get('mAP50', 'N/A'):.4f}")
        
        # Save report
        report_path = output_path / 'training_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Training report saved to: {report_path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 for text detection')
    parser.add_argument('--config', '-c', required=True, help='Training configuration YAML file')
    parser.add_argument('--name', '-n', help='Experiment name')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--validate-only', '-v', action='store_true', help='Only run validation')
    parser.add_argument('--export', '-e', action='store_true', help='Export model after training')
    parser.add_argument('--model-path', '-m', help='Path to model for validation/export')
    parser.add_argument('--output-dir', '-o', default='training_output', help='Output directory')
    parser.add_argument('--export-formats', nargs='+', default=['pt', 'onnx'], 
                        help='Export formats')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    try:
        # Initialize trainer
        trainer = TextDetectionTrainer(args.config, args.name)
        
        if args.validate_only:
            # Validation only
            trainer.validate_model(args.model_path)
        else:
            # Full training
            trainer.train(resume=args.resume)
            
            # Validation
            trainer.validate_model()
        
        # Export model
        if args.export:
            trainer.export_model(args.model_path, args.export_formats)
        
        # Save results
        trainer.save_training_summary(args.output_dir)
        trainer.create_training_report(args.output_dir)
        
        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())