#!/usr/bin/env python3
"""
Dataset downloader for public text detection datasets.
Downloads and prepares ICDAR, TextOCR, and other text detection datasets.
"""

import os
import zipfile
import tarfile
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
import logging
from urllib.parse import urlparse
from tqdm import tqdm
import hashlib

from annotation_utils import DatasetConverter


class DatasetDownloader:
    """Download and prepare public text detection datasets."""
    
    def __init__(self, data_dir: str = "data/external_datasets"):
        """
        Initialize dataset downloader.
        
        Args:
            data_dir: Base directory for storing datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.converter = DatasetConverter()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Dataset configurations
        self.datasets = {
            'icdar2015': {
                'name': 'ICDAR 2015 Incidental Scene Text',
                'urls': {
                    'train_images': 'https://rrc.cvc.uab.es/downloads/ch4_training_images.zip',
                    'train_gt': 'https://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip',
                    'test_images': 'https://rrc.cvc.uab.es/downloads/ch4_test_images.zip'
                },
                'description': 'Focused on incidental scene text detection and recognition',
                'format': 'icdar',
                'license': 'Research use only'
            },
            'icdar2017': {
                'name': 'ICDAR 2017 Robust Reading',
                'urls': {
                    'train_images': 'https://datasets.cvc.uab.es/rrc/ch8_training_images_1.zip',
                    'train_gt': 'https://datasets.cvc.uab.es/rrc/ch8_training_gt_1.zip'
                },
                'description': 'Multi-lingual scene text detection',
                'format': 'icdar',
                'license': 'Research use only'
            },
            'synthtext': {
                'name': 'SynthText Dataset',
                'urls': {
                    'data': 'https://thor.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip'
                },
                'description': 'Synthetic text in natural images',
                'format': 'matlab',
                'license': 'Academic use only',
                'size_gb': 41
            },
            'textocr': {
                'name': 'TextOCR Dataset',
                'urls': {
                    'train_images': 'https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip',
                    'annotations': 'https://dl.fbaipublicfiles.com/textocr/data/textocr_train_annotations.json'
                },
                'description': 'Large-scale text reading dataset',
                'format': 'coco',
                'license': 'CC BY 4.0'
            },
            'coco_text': {
                'name': 'COCO-Text Dataset',
                'urls': {
                    'train_images': 'http://images.cocodataset.org/zips/train2014.zip',
                    'val_images': 'http://images.cocodataset.org/zips/val2014.zip',
                    'annotations': 'https://github.com/andreasveit/coco-text/raw/master/COCO_Text.json'
                },
                'description': 'Text detection and recognition on COCO images',
                'format': 'coco',
                'license': 'CC BY 4.0'
            }
        }
    
    def download_file(self, url: str, output_path: Path, expected_size: Optional[int] = None) -> bool:
        """
        Download file with progress bar and validation.
        
        Args:
            url: URL to download
            output_path: Local path to save file
            expected_size: Expected file size in bytes
            
        Returns:
            True if download successful
        """
        if output_path.exists():
            self.logger.info(f"File already exists: {output_path}")
            return True
        
        try:
            self.logger.info(f"Downloading {url}")
            
            # Get file size
            response = requests.head(url, allow_redirects=True)
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f, tqdm(
                desc=output_path.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Validate file size
            actual_size = output_path.stat().st_size
            if expected_size and abs(actual_size - expected_size) > 1024:
                self.logger.warning(f"File size mismatch: expected {expected_size}, got {actual_size}")
            
            self.logger.info(f"Downloaded: {output_path} ({actual_size:,} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def extract_archive(self, archive_path: Path, extract_dir: Path) -> bool:
        """
        Extract archive file.
        
        Args:
            archive_path: Path to archive file
            extract_dir: Directory to extract to
            
        Returns:
            True if extraction successful
        """
        try:
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Extracting {archive_path} to {extract_dir}")
            
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                self.logger.error(f"Unsupported archive format: {archive_path}")
                return False
            
            self.logger.info(f"Extracted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to extract {archive_path}: {e}")
            return False
    
    def download_icdar2015(self) -> bool:
        """Download and prepare ICDAR 2015 dataset."""
        dataset_info = self.datasets['icdar2015']
        dataset_dir = self.data_dir / 'icdar2015'
        
        self.logger.info(f"Downloading {dataset_info['name']}")
        
        # Download files
        downloads = []
        for name, url in dataset_info['urls'].items():
            filename = f"{name}.zip"
            file_path = dataset_dir / 'downloads' / filename
            
            if self.download_file(url, file_path):
                downloads.append((name, file_path))
            else:
                return False
        
        # Extract files
        for name, file_path in downloads:
            extract_dir = dataset_dir / name
            if not self.extract_archive(file_path, extract_dir):
                return False
        
        # Convert to YOLO format
        self.logger.info("Converting ICDAR 2015 to YOLO format")
        
        # Find the actual directory structure after extraction
        train_images_dir = self._find_images_directory(dataset_dir / 'train_images')
        train_gt_dir = self._find_gt_directory(dataset_dir / 'train_gt')
        
        if train_images_dir and train_gt_dir:
            output_dir = dataset_dir / 'yolo_format'
            self.converter.convert_icdar_dataset(
                str(train_gt_dir.parent),
                str(output_dir),
                'icdar2015'
            )
        
        # Create dataset info
        self._create_dataset_info(dataset_dir, dataset_info)
        
        return True
    
    def download_textocr(self) -> bool:
        """Download and prepare TextOCR dataset."""
        dataset_info = self.datasets['textocr']
        dataset_dir = self.data_dir / 'textocr'
        
        self.logger.info(f"Downloading {dataset_info['name']}")
        
        # Download annotations first (smaller file)
        annotations_path = dataset_dir / 'annotations' / 'textocr_train_annotations.json'
        if not self.download_file(dataset_info['urls']['annotations'], annotations_path):
            return False
        
        # Download images (large file)
        images_archive = dataset_dir / 'downloads' / 'train_val_images.zip'
        if not self.download_file(dataset_info['urls']['train_images'], images_archive):
            return False
        
        # Extract images
        images_dir = dataset_dir / 'images'
        if not self.extract_archive(images_archive, images_dir):
            return False
        
        # Convert annotations to YOLO format
        self._convert_textocr_annotations(annotations_path, images_dir, dataset_dir / 'yolo_format')
        
        # Create dataset info
        self._create_dataset_info(dataset_dir, dataset_info)
        
        return True
    
    def download_synthtext(self) -> bool:
        """Download and prepare SynthText dataset."""
        dataset_info = self.datasets['synthtext']
        dataset_dir = self.data_dir / 'synthtext'
        
        # Warn about size
        size_gb = dataset_info.get('size_gb', 0)
        self.logger.warning(f"SynthText dataset is {size_gb}GB. Continue? This may take hours to download.")
        
        # For now, just create the structure and instructions
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        instructions = f"""
# SynthText Dataset Download Instructions

The SynthText dataset is very large ({size_gb}GB) and requires manual download.

1. Visit: https://thor.robots.ox.ac.uk/~vgg/data/scenetext/
2. Download SynthText.zip
3. Extract to: {dataset_dir}/raw/
4. Run conversion script to convert to YOLO format

The dataset contains synthetic text images that can be useful for
data augmentation and pre-training text detection models.
"""
        
        with open(dataset_dir / 'download_instructions.txt', 'w') as f:
            f.write(instructions)
        
        self.logger.info(f"Created download instructions at {dataset_dir}")
        return True
    
    def _find_images_directory(self, base_dir: Path) -> Optional[Path]:
        """Find the directory containing images after extraction."""
        if not base_dir.exists():
            return None
        
        # Look for common image directories
        for subdir in base_dir.rglob('*'):
            if subdir.is_dir():
                image_files = list(subdir.glob('*.jpg')) + list(subdir.glob('*.png'))
                if len(image_files) > 10:  # Likely an images directory
                    return subdir
        
        return base_dir if any(base_dir.glob('*.jpg')) else None
    
    def _find_gt_directory(self, base_dir: Path) -> Optional[Path]:
        """Find the directory containing ground truth files after extraction."""
        if not base_dir.exists():
            return None
        
        # Look for ground truth files
        for subdir in base_dir.rglob('*'):
            if subdir.is_dir():
                gt_files = list(subdir.glob('gt_*.txt'))
                if len(gt_files) > 5:  # Likely a GT directory
                    return subdir
        
        return base_dir if any(base_dir.glob('gt_*.txt')) else None
    
    def _convert_textocr_annotations(self, annotations_path: Path, images_dir: Path, output_dir: Path):
        """Convert TextOCR annotations to YOLO format."""
        self.logger.info("Converting TextOCR annotations to YOLO format")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'images').mkdir(exist_ok=True)
        (output_dir / 'labels').mkdir(exist_ok=True)
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        # Group annotations by image
        image_annotations = {}
        for ann in data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
        
        # Convert each image
        converted_count = 0
        for image_info in data.get('images', []):
            image_id = image_info['id']
            filename = image_info['file_name']
            
            # Find image file
            image_path = None
            for img_file in images_dir.rglob(filename):
                image_path = img_file
                break
            
            if not image_path or not image_path.exists():
                continue
            
            # Get annotations for this image
            annotations = image_annotations.get(image_id, [])
            if not annotations:
                continue
            
            # Convert annotations to YOLO format
            yolo_lines = []
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                text = ann.get('utf8_string', '')
                
                # Classify text for recipe context
                category = self.converter.converter._classify_text(text)
                
                # Convert to YOLO format
                coco_ann = {
                    'bbox': bbox,
                    'category': category,
                    'text': text
                }
                
                try:
                    yolo_line = self.converter.converter.coco_to_yolo(
                        coco_ann, image_info['width'], image_info['height']
                    )
                    yolo_lines.append(yolo_line)
                except Exception as e:
                    self.logger.warning(f"Failed to convert annotation: {e}")
            
            if yolo_lines:
                # Copy image
                output_image_path = output_dir / 'images' / f"textocr_{converted_count:06d}.jpg"
                import shutil
                shutil.copy2(image_path, output_image_path)
                
                # Save labels
                output_label_path = output_dir / 'labels' / f"textocr_{converted_count:06d}.txt"
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    self.logger.info(f"Converted {converted_count} images")
        
        self.logger.info(f"Converted {converted_count} TextOCR images")
    
    def _create_dataset_info(self, dataset_dir: Path, dataset_info: Dict):
        """Create dataset information file."""
        info = {
            'name': dataset_info['name'],
            'description': dataset_info['description'],
            'format': dataset_info['format'],
            'license': dataset_info['license'],
            'download_date': None,  # Add timestamp
            'conversion_info': {
                'converted_to_yolo': True,
                'classes': list(DatasetConverter.CLASS_MAPPING.keys()),
                'total_images': 0,  # Count after conversion
                'total_annotations': 0
            }
        }
        
        # Count converted files if available
        yolo_dir = dataset_dir / 'yolo_format'
        if yolo_dir.exists():
            images = list((yolo_dir / 'images').glob('*.jpg'))
            labels = list((yolo_dir / 'labels').glob('*.txt'))
            
            info['conversion_info']['total_images'] = len(images)
            
            # Count annotations
            total_annotations = 0
            for label_file in labels:
                with open(label_file, 'r') as f:
                    total_annotations += len([l for l in f if l.strip()])
            
            info['conversion_info']['total_annotations'] = total_annotations
        
        # Save info
        with open(dataset_dir / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
    
    def list_available_datasets(self) -> Dict:
        """List all available datasets with information."""
        return self.datasets
    
    def download_dataset(self, dataset_name: str) -> bool:
        """
        Download a specific dataset.
        
        Args:
            dataset_name: Name of dataset to download
            
        Returns:
            True if download successful
        """
        if dataset_name not in self.datasets:
            self.logger.error(f"Unknown dataset: {dataset_name}")
            self.logger.info(f"Available datasets: {list(self.datasets.keys())}")
            return False
        
        if dataset_name == 'icdar2015':
            return self.download_icdar2015()
        elif dataset_name == 'textocr':
            return self.download_textocr()
        elif dataset_name == 'synthtext':
            return self.download_synthtext()
        else:
            self.logger.error(f"Download method not implemented for {dataset_name}")
            return False
    
    def download_all(self, exclude: List[str] = None) -> Dict[str, bool]:
        """
        Download all available datasets.
        
        Args:
            exclude: List of dataset names to exclude
            
        Returns:
            Dictionary with download results
        """
        exclude = exclude or ['synthtext']  # Exclude large datasets by default
        results = {}
        
        for dataset_name in self.datasets:
            if dataset_name in exclude:
                self.logger.info(f"Skipping {dataset_name} (excluded)")
                results[dataset_name] = 'skipped'
                continue
            
            self.logger.info(f"Downloading {dataset_name}...")
            results[dataset_name] = self.download_dataset(dataset_name)
        
        return results


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download text detection datasets')
    parser.add_argument('--dataset', choices=['icdar2015', 'icdar2017', 'textocr', 'synthtext', 'all'],
                       default='all', help='Dataset to download')
    parser.add_argument('--data-dir', default='data/external_datasets',
                       help='Directory to store datasets')
    parser.add_argument('--exclude', nargs='+', default=['synthtext'],
                       help='Datasets to exclude when downloading all')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.list:
        print("Available datasets:")
        for name, info in downloader.list_available_datasets().items():
            print(f"  {name}: {info['description']}")
            print(f"    Format: {info['format']}, License: {info['license']}")
            if 'size_gb' in info:
                print(f"    Size: {info['size_gb']}GB")
        return
    
    if args.dataset == 'all':
        results = downloader.download_all(exclude=args.exclude)
        print("\nDownload Results:")
        for dataset, result in results.items():
            print(f"  {dataset}: {result}")
    else:
        success = downloader.download_dataset(args.dataset)
        print(f"Download {'successful' if success else 'failed'}")


if __name__ == "__main__":
    main()