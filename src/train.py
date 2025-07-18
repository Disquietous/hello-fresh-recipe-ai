#!/usr/bin/env python3
"""
HelloFresh Recipe AI - Training Script
Train custom YOLOv8 models for food detection and recipe analysis.
"""

import argparse
import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


class FoodModelTrainer:
    """Custom model trainer for food detection."""
    
    def __init__(self, model_size="n", pretrained=True):
        """
        Initialize the trainer.
        
        Args:
            model_size (str): YOLOv8 model size (n, s, m, l, x)
            pretrained (bool): Use pretrained weights
        """
        self.model_size = model_size
        if pretrained:
            self.model = YOLO(f"yolov8{model_size}.pt")
        else:
            self.model = YOLO(f"yolov8{model_size}.yaml")
    
    def train(self, data_config, epochs=100, batch_size=16, img_size=640, 
              learning_rate=0.01, save_dir="models/custom", **kwargs):
        """
        Train the model.
        
        Args:
            data_config (str): Path to data configuration YAML file
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            img_size (int): Image size for training
            learning_rate (float): Learning rate
            save_dir (str): Directory to save trained model
            **kwargs: Additional training parameters
        """
        
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        train_params = {
            'data': data_config,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'lr0': learning_rate,
            'project': save_dir,
            'name': f'food_model_{self.model_size}',
            'save_period': 10,  # Save checkpoint every 10 epochs
            'patience': 20,     # Early stopping patience
            'cache': True,      # Cache images for faster training
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            **kwargs
        }
        
        print(f"Starting training with parameters:")
        for key, value in train_params.items():
            print(f"  {key}: {value}")
        
        # Start training
        results = self.model.train(**train_params)
        
        print(f"\nTraining completed!")
        print(f"Best model saved at: {results.save_dir}/weights/best.pt")
        
        return results
    
    def validate(self, data_config, model_path=None):
        """
        Validate the model.
        
        Args:
            data_config (str): Path to data configuration YAML file
            model_path (str): Path to trained model (if None, uses current model)
        """
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
            
        results = model.val(data=data_config)
        
        print(f"Validation Results:")
        print(f"  mAP50: {results.box.map50:.3f}")
        print(f"  mAP50-95: {results.box.map:.3f}")
        print(f"  Precision: {results.box.mp:.3f}")
        print(f"  Recall: {results.box.mr:.3f}")
        
        return results
    
    def export_model(self, model_path, export_format="onnx"):
        """
        Export trained model to different formats.
        
        Args:
            model_path (str): Path to trained model
            export_format (str): Export format (onnx, tensorrt, coreml, etc.)
        """
        model = YOLO(model_path)
        exported_path = model.export(format=export_format)
        print(f"Model exported to: {exported_path}")
        return exported_path


def create_data_config(data_dir, config_path="configs/food_data.yaml", 
                      class_names=None):
    """
    Create YOLO data configuration file.
    
    Args:
        data_dir (str): Path to data directory
        config_path (str): Path to save configuration file
        class_names (list): List of class names
    """
    
    if class_names is None:
        # Default food classes - customize as needed
        class_names = [
            'apple', 'banana', 'orange', 'carrot', 'broccoli',
            'tomato', 'potato', 'onion', 'bell_pepper', 'cucumber',
            'bread', 'egg', 'milk', 'cheese', 'chicken',
            'beef', 'fish', 'pasta', 'rice', 'salad'
        ]
    
    data_config = {
        'path': str(Path(data_dir).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    # Create config directory
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(config_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Data configuration saved to: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description='HelloFresh Recipe AI - Model Training')
    parser.add_argument('--data', required=True, help='Path to data directory or config file')
    parser.add_argument('--model-size', default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--save-dir', default='models/custom', help='Save directory')
    parser.add_argument('--validate-only', action='store_true', 
                       help='Only run validation')
    parser.add_argument('--model-path', help='Path to trained model for validation/export')
    parser.add_argument('--export', choices=['onnx', 'tensorrt', 'coreml', 'saved_model'],
                       help='Export model format')
    parser.add_argument('--create-config', action='store_true',
                       help='Create data configuration file')
    parser.add_argument('--classes', nargs='+', help='Custom class names')
    
    args = parser.parse_args()
    
    # Create data configuration if requested
    if args.create_config:
        if not Path(args.data).exists():
            print(f"Error: Data directory {args.data} does not exist")
            return
        
        config_path = create_data_config(
            args.data, 
            "configs/food_data.yaml",
            args.classes
        )
        data_config = config_path
    else:
        data_config = args.data
    
    # Initialize trainer
    trainer = FoodModelTrainer(args.model_size)
    
    # Export model if requested
    if args.export:
        if not args.model_path:
            print("Error: --model-path required for export")
            return
        trainer.export_model(args.model_path, args.export)
        return
    
    # Validation only
    if args.validate_only:
        trainer.validate(data_config, args.model_path)
        return
    
    # Training
    print(f"Training YOLOv8{args.model_size} model...")
    print(f"Data config: {data_config}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU (training will be slower)")
    
    # Start training
    results = trainer.train(
        data_config=data_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
    
    # Run validation on best model
    print("\nRunning validation on best model...")
    best_model_path = Path(args.save_dir) / f'food_model_{args.model_size}' / 'weights' / 'best.pt'
    if best_model_path.exists():
        trainer.validate(data_config, str(best_model_path))


if __name__ == "__main__":
    main()