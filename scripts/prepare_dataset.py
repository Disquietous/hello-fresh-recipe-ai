#!/usr/bin/env python3
"""
Dataset preparation script for recipe text detection.
Handles downloading, conversion, and preparation of training datasets.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.dataset_downloader import DatasetDownloader
from utils.data_utils import prepare_text_detection_dataset, analyze_dataset_distribution, validate_dataset_structure
from utils.annotation_utils import DatasetConverter
import json


def download_external_datasets(data_dir: str, datasets: list = None, exclude: list = None):
    """Download external text detection datasets."""
    print("🔽 Downloading external datasets...")
    
    downloader = DatasetDownloader(data_dir)
    
    if datasets:
        # Download specific datasets
        for dataset_name in datasets:
            print(f"Downloading {dataset_name}...")
            success = downloader.download_dataset(dataset_name)
            print(f"{'✅' if success else '❌'} {dataset_name}")
    else:
        # Download all datasets (excluding large ones by default)
        exclude = exclude or ['synthtext']
        results = downloader.download_all(exclude=exclude)
        
        print("Download Results:")
        for dataset, result in results.items():
            status = '✅' if result else '❌' if result != 'skipped' else '⏭️'
            print(f"{status} {dataset}: {result}")


def prepare_recipe_dataset(raw_data_dir: str, output_dir: str, config: dict = None):
    """Prepare recipe text detection dataset from raw images."""
    print("📋 Preparing recipe text detection dataset...")
    
    config = config or {
        'split_ratios': (0.7, 0.2, 0.1),
        'augment_data': True,
        'augmentations_per_image': 3
    }
    
    print(f"Input directory: {raw_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Configuration: {config}")
    
    # Prepare dataset
    stats = prepare_text_detection_dataset(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        split_ratios=config['split_ratios'],
        augment_data=config['augment_data'],
        augmentations_per_image=config['augmentations_per_image']
    )
    
    print("📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return stats


def convert_external_dataset(dataset_path: str, dataset_type: str, output_dir: str):
    """Convert external dataset to YOLO format."""
    print(f"🔄 Converting {dataset_type} dataset...")
    
    converter = DatasetConverter()
    
    if dataset_type.lower() == 'icdar':
        converter.convert_icdar_dataset(dataset_path, output_dir, 'icdar_converted')
    else:
        print(f"❌ Unsupported dataset type: {dataset_type}")
        return False
    
    print(f"✅ Dataset converted to: {output_dir}")
    return True


def validate_dataset(dataset_dir: str, dataset_type: str = 'yolo'):
    """Validate dataset structure and quality."""
    print(f"🔍 Validating {dataset_type} dataset at {dataset_dir}...")
    
    # Structure validation
    is_valid = validate_dataset_structure(dataset_dir, dataset_type)
    
    if not is_valid:
        print("❌ Dataset structure validation failed")
        return False
    
    # Detailed analysis
    if dataset_type == 'yolo':
        print("📊 Analyzing dataset distribution...")
        analysis = analyze_dataset_distribution(dataset_dir)
        
        print("Dataset Analysis Results:")
        print(f"  Total images: {analysis['overall']['total_images']}")
        print(f"  Total annotations: {analysis['overall']['total_annotations']}")
        print(f"  Annotations per image: {analysis['overall']['annotations_per_image']:.2f}")
        
        print("Split distribution:")
        for split, data in analysis['splits'].items():
            print(f"  {split}: {data['image_count']} images, {data['label_count']} labels")
        
        print("Class distribution:")
        for class_id, count in analysis['class_distribution'].items():
            print(f"  Class {class_id}: {count} annotations")
        
        # Check for issues
        issues = []
        if analysis['overall']['annotations_per_image'] < 1:
            issues.append("Low annotation density")
        
        balance = analysis['overall']['class_balance']
        if balance['balance_score'] < 0.5:
            issues.append("Imbalanced class distribution")
        
        if issues:
            print(f"⚠️  Issues found: {', '.join(issues)}")
        else:
            print("✅ Dataset quality looks good")
    
    return True


def create_annotation_template(image_path: str, output_path: str):
    """Create annotation template for a recipe image."""
    print(f"📝 Creating annotation template for {image_path}...")
    
    converter = DatasetConverter()
    converter.create_recipe_annotation_template(image_path, output_path)
    
    print(f"✅ Template created: {output_path}")
    print("Edit the template to add your annotations, then use prepare_dataset.py to convert to YOLO format")


def merge_datasets(dataset_dirs: list, output_dir: str, weights: list = None):
    """Merge multiple datasets into one."""
    print(f"🔗 Merging {len(dataset_dirs)} datasets into {output_dir}...")
    
    if weights and len(weights) != len(dataset_dirs):
        print("❌ Number of weights must match number of datasets")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    merged_stats = {'train': 0, 'val': 0, 'test': 0}
    
    for i, dataset_dir in enumerate(dataset_dirs):
        dataset_path = Path(dataset_dir)
        weight = weights[i] if weights else 1.0
        
        print(f"  Merging dataset {i+1}: {dataset_dir} (weight: {weight})")
        
        if not validate_dataset_structure(str(dataset_path), 'yolo'):
            print(f"    ⚠️  Skipping invalid dataset: {dataset_dir}")
            continue
        
        # Copy files from each split
        for split in ['train', 'val', 'test']:
            split_dir = dataset_path / split
            if not split_dir.exists():
                continue
            
            images = list((split_dir / 'images').glob('*'))
            labels = list((split_dir / 'labels').glob('*'))
            
            # Sample based on weight
            if weight < 1.0:
                import random
                sample_size = int(len(images) * weight)
                images = random.sample(images, min(sample_size, len(images)))
                # Get corresponding labels
                image_stems = {img.stem for img in images}
                labels = [lbl for lbl in labels if lbl.stem in image_stems]
            
            # Copy files
            for img_file in images:
                new_name = f"dataset_{i}_{img_file.name}"
                import shutil
                shutil.copy2(img_file, output_path / split / 'images' / new_name)
                merged_stats[split] += 1
            
            for lbl_file in labels:
                new_name = f"dataset_{i}_{lbl_file.name}"
                import shutil
                shutil.copy2(lbl_file, output_path / split / 'labels' / new_name)
    
    print("✅ Dataset merging complete:")
    for split, count in merged_stats.items():
        print(f"  {split}: {count} images")
    
    return True


def main():
    """Main function for dataset preparation."""
    parser = argparse.ArgumentParser(description='Recipe Text Detection Dataset Preparation')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download external datasets')
    download_parser.add_argument('--data-dir', default='data/external_datasets',
                                help='Directory to store datasets')
    download_parser.add_argument('--datasets', nargs='+', 
                                choices=['icdar2015', 'icdar2017', 'textocr', 'synthtext'],
                                help='Specific datasets to download')
    download_parser.add_argument('--exclude', nargs='+', default=['synthtext'],
                                help='Datasets to exclude')
    
    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare recipe dataset')
    prepare_parser.add_argument('input_dir', help='Directory with raw recipe images')
    prepare_parser.add_argument('output_dir', help='Output directory for processed dataset')
    prepare_parser.add_argument('--config', help='Configuration JSON file')
    prepare_parser.add_argument('--no-augment', action='store_true', 
                               help='Disable data augmentation')
    prepare_parser.add_argument('--augmentations', type=int, default=3,
                               help='Number of augmentations per image')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert external dataset')
    convert_parser.add_argument('dataset_path', help='Path to external dataset')
    convert_parser.add_argument('dataset_type', choices=['icdar'], 
                               help='Type of external dataset')
    convert_parser.add_argument('output_dir', help='Output directory')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('dataset_dir', help='Dataset directory to validate')
    validate_parser.add_argument('--type', default='yolo', choices=['yolo', 'coco', 'recipe_text'],
                                help='Dataset type')
    
    # Template command
    template_parser = subparsers.add_parser('template', help='Create annotation template')
    template_parser.add_argument('image_path', help='Path to recipe image')
    template_parser.add_argument('output_path', help='Output path for template')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple datasets')
    merge_parser.add_argument('dataset_dirs', nargs='+', help='Dataset directories to merge')
    merge_parser.add_argument('output_dir', help='Output directory for merged dataset')
    merge_parser.add_argument('--weights', type=float, nargs='+',
                             help='Sampling weights for each dataset')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'download':
            download_external_datasets(
                data_dir=args.data_dir,
                datasets=args.datasets,
                exclude=args.exclude
            )
        
        elif args.command == 'prepare':
            config = {}
            if args.config:
                with open(args.config, 'r') as f:
                    config = json.load(f)
            
            if args.no_augment:
                config['augment_data'] = False
            if args.augmentations:
                config['augmentations_per_image'] = args.augmentations
            
            prepare_recipe_dataset(args.input_dir, args.output_dir, config)
        
        elif args.command == 'convert':
            convert_external_dataset(args.dataset_path, args.dataset_type, args.output_dir)
        
        elif args.command == 'validate':
            validate_dataset(args.dataset_dir, args.type)
        
        elif args.command == 'template':
            create_annotation_template(args.image_path, args.output_path)
        
        elif args.command == 'merge':
            merge_datasets(args.dataset_dirs, args.output_dir, args.weights)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    print("✅ Task completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())