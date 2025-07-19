"""
Text-specific data augmentation for recipe text detection.
Specialized augmentations for text images including rotation, blur, lighting, and synthetic variations.
"""

import cv2
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
import json


class TextAugmentationPipeline:
    """Text-specific augmentation pipeline for recipe images."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize augmentation pipeline.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config or self._get_default_config()
        self.augmentations = self._create_augmentation_pipeline()
    
    def _get_default_config(self) -> Dict:
        """Get default augmentation configuration."""
        return {
            'geometric': {
                'rotation_range': (-15, 15),  # degrees
                'perspective_scale': (0.02, 0.05),
                'elastic_alpha': (1, 3),
                'elastic_sigma': (50, 120),
                'shear_range': (-5, 5),  # degrees
                'scale_range': (0.8, 1.2)
            },
            'photometric': {
                'brightness_range': (-0.2, 0.2),
                'contrast_range': (0.8, 1.2),
                'saturation_range': (0.8, 1.2),
                'hue_range': (-0.1, 0.1),
                'gamma_range': (0.8, 1.2)
            },
            'noise_and_blur': {
                'gaussian_noise_var': (10, 50),
                'motion_blur_kernel': (3, 7),
                'gaussian_blur_sigma': (0.5, 2.0),
                'jpeg_quality': (70, 95)
            },
            'text_specific': {
                'shadow_intensity': (0.1, 0.4),
                'highlight_intensity': (0.1, 0.3),
                'ink_bleed_probability': 0.2,
                'paper_texture_probability': 0.3,
                'handwriting_variation': 0.2
            },
            'probabilities': {
                'geometric': 0.7,
                'photometric': 0.8,
                'noise_blur': 0.5,
                'text_specific': 0.6
            }
        }
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create albumentations augmentation pipeline."""
        
        # Geometric transformations
        geometric_transforms = [
            A.Rotate(
                limit=self.config['geometric']['rotation_range'],
                border_mode=cv2.BORDER_CONSTANT,
                value=255,  # White background
                p=0.7
            ),
            A.Perspective(
                scale=self.config['geometric']['perspective_scale'],
                p=0.3
            ),
            A.ElasticTransform(
                alpha=self.config['geometric']['elastic_alpha'][1],
                sigma=self.config['geometric']['elastic_sigma'][1],
                alpha_affine=30,
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                p=0.2
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                p=0.4
            )
        ]
        
        # Photometric transformations
        photometric_transforms = [
            A.RandomBrightnessContrast(
                brightness_limit=self.config['photometric']['brightness_range'],
                contrast_limit=self.config['photometric']['contrast_range'],
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(self.config['photometric']['hue_range'][1] * 180),
                sat_shift_limit=int((self.config['photometric']['saturation_range'][1] - 1) * 100),
                val_shift_limit=20,
                p=0.6
            ),
            A.RandomGamma(
                gamma_limit=self.config['photometric']['gamma_range'],
                p=0.5
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3)
        ]
        
        # Noise and blur
        noise_blur_transforms = [
            A.GaussNoise(
                var_limit=self.config['noise_and_blur']['gaussian_noise_var'],
                p=0.4
            ),
            A.MotionBlur(
                blur_limit=self.config['noise_and_blur']['motion_blur_kernel'],
                p=0.3
            ),
            A.GaussianBlur(
                blur_limit=(3, 7),
                sigma_limit=self.config['noise_and_blur']['gaussian_blur_sigma'],
                p=0.3
            ),
            A.ImageCompression(
                quality_lower=self.config['noise_and_blur']['jpeg_quality'][0],
                quality_upper=self.config['noise_and_blur']['jpeg_quality'][1],
                p=0.3
            )
        ]
        
        # Combine all transforms
        all_transforms = []
        
        # Add geometric with probability
        if self.config['probabilities']['geometric'] > 0:
            all_transforms.append(
                A.OneOf(geometric_transforms, p=self.config['probabilities']['geometric'])
            )
        
        # Add photometric with probability
        if self.config['probabilities']['photometric'] > 0:
            all_transforms.append(
                A.OneOf(photometric_transforms, p=self.config['probabilities']['photometric'])
            )
        
        # Add noise/blur with probability
        if self.config['probabilities']['noise_blur'] > 0:
            all_transforms.append(
                A.OneOf(noise_blur_transforms, p=self.config['probabilities']['noise_blur'])
            )
        
        return A.Compose(
            all_transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_area=0,
                min_visibility=0.3
            )
        )
    
    def augment_image_with_boxes(self, image: np.ndarray, boxes: List[List[float]], 
                                class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Augment image with bounding boxes.
        
        Args:
            image: Input image
            boxes: Bounding boxes in YOLO format [x_center, y_center, width, height]
            class_labels: Class labels for each box
            
        Returns:
            Tuple of (augmented_image, augmented_boxes, augmented_labels)
        """
        if len(boxes) == 0:
            # No boxes, just augment image
            transformed = self.augmentations(image=image)
            return transformed['image'], [], []
        
        try:
            transformed = self.augmentations(
                image=image,
                bboxes=boxes,
                class_labels=class_labels
            )
            
            return (
                transformed['image'],
                transformed['bboxes'],
                transformed['class_labels']
            )
            
        except Exception as e:
            print(f"Augmentation failed: {e}")
            return image, boxes, class_labels
    
    def apply_text_specific_augmentations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply text-specific augmentations.
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        # Apply text-specific effects with probability
        if random.random() < self.config['text_specific']['shadow_intensity']:
            augmented = self._add_text_shadow(augmented)
        
        if random.random() < self.config['text_specific']['highlight_intensity']:
            augmented = self._add_text_highlight(augmented)
        
        if random.random() < self.config['text_specific']['ink_bleed_probability']:
            augmented = self._simulate_ink_bleed(augmented)
        
        if random.random() < self.config['text_specific']['paper_texture_probability']:
            augmented = self._add_paper_texture(augmented)
        
        return augmented
    
    def _add_text_shadow(self, image: np.ndarray) -> np.ndarray:
        """Add shadow effect to text."""
        shadow_intensity = random.uniform(*self.config['text_specific']['shadow_intensity'])
        
        # Create shadow by shifting image slightly
        rows, cols = image.shape[:2]
        shadow_offset = random.randint(1, 3)
        
        # Create transformation matrix
        M = np.float32([[1, 0, shadow_offset], [0, 1, shadow_offset]])
        shadow = cv2.warpAffine(image, M, (cols, rows), borderValue=255)
        
        # Darken shadow
        shadow = cv2.addWeighted(shadow, 1 - shadow_intensity, 
                               np.full_like(shadow, 128), shadow_intensity, 0)
        
        # Combine with original
        result = np.minimum(image, shadow)
        return result
    
    def _add_text_highlight(self, image: np.ndarray) -> np.ndarray:
        """Add highlight effect to text."""
        highlight_intensity = random.uniform(*self.config['text_specific']['highlight_intensity'])
        
        # Find text regions (dark areas)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Create highlight mask for text areas
        text_mask = gray < 200  # Assume text is darker than background
        
        # Create highlight
        highlight = np.full_like(image, 255)
        if len(image.shape) == 3:
            highlight[:, :, 1] = 200  # Slight yellow tint
        
        # Apply highlight to text areas
        result = image.copy()
        if len(image.shape) == 3:
            for c in range(3):
                result[:, :, c] = np.where(
                    text_mask,
                    cv2.addWeighted(image[:, :, c], 1 - highlight_intensity,
                                  highlight[:, :, c], highlight_intensity, 0),
                    image[:, :, c]
                )
        else:
            result = np.where(
                text_mask,
                cv2.addWeighted(image, 1 - highlight_intensity,
                              highlight, highlight_intensity, 0),
                image
            )
        
        return result
    
    def _simulate_ink_bleed(self, image: np.ndarray) -> np.ndarray:
        """Simulate ink bleeding effect."""
        # Apply slight dilation to text areas
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find text areas
        text_mask = gray < 200
        
        # Apply dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(text_mask.astype(np.uint8), kernel, iterations=1)
        
        # Create bleeding effect
        result = image.copy()
        bleeding_color = random.randint(50, 150)  # Gray bleeding
        
        if len(image.shape) == 3:
            for c in range(3):
                result[:, :, c] = np.where(
                    dilated_mask & ~text_mask,
                    bleeding_color,
                    image[:, :, c]
                )
        else:
            result = np.where(
                dilated_mask & ~text_mask,
                bleeding_color,
                image
            )
        
        return result
    
    def _add_paper_texture(self, image: np.ndarray) -> np.ndarray:
        """Add paper texture to background."""
        rows, cols = image.shape[:2]
        
        # Create paper texture noise
        texture = np.random.normal(255, 10, (rows, cols)).astype(np.uint8)
        texture = cv2.GaussianBlur(texture, (5, 5), 0)
        
        # Apply only to background areas
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        background_mask = gray > 200  # Assume background is light
        
        result = image.copy()
        if len(image.shape) == 3:
            for c in range(3):
                result[:, :, c] = np.where(
                    background_mask,
                    cv2.addWeighted(image[:, :, c], 0.9, texture, 0.1, 0),
                    image[:, :, c]
                )
        else:
            result = np.where(
                background_mask,
                cv2.addWeighted(image, 0.9, texture, 0.1, 0),
                image
            )
        
        return result


class RecipeFormatAugmenter:
    """Augmenter for different recipe formats (handwritten, printed, digital)."""
    
    def __init__(self):
        self.format_configs = {
            'handwritten': {
                'rotation_range': (-20, 20),
                'perspective_strength': 0.1,
                'ink_variation': True,
                'paper_texture': True,
                'lighting_variation': 0.3
            },
            'printed': {
                'rotation_range': (-5, 5),
                'perspective_strength': 0.02,
                'print_quality_variation': True,
                'scan_artifacts': True,
                'lighting_variation': 0.2
            },
            'digital': {
                'rotation_range': (-2, 2),
                'screen_glare': True,
                'compression_artifacts': True,
                'font_antialiasing': True,
                'lighting_variation': 0.1
            }
        }
    
    def augment_for_format(self, image: np.ndarray, recipe_format: str) -> np.ndarray:
        """
        Apply format-specific augmentations.
        
        Args:
            image: Input image
            recipe_format: Format type ('handwritten', 'printed', 'digital')
            
        Returns:
            Augmented image
        """
        if recipe_format not in self.format_configs:
            return image
        
        config = self.format_configs[recipe_format]
        augmented = image.copy()
        
        if recipe_format == 'handwritten':
            augmented = self._apply_handwritten_effects(augmented, config)
        elif recipe_format == 'printed':
            augmented = self._apply_printed_effects(augmented, config)
        elif recipe_format == 'digital':
            augmented = self._apply_digital_effects(augmented, config)
        
        return augmented
    
    def _apply_handwritten_effects(self, image: np.ndarray, config: Dict) -> np.ndarray:
        """Apply handwritten-specific effects."""
        # Stronger perspective and rotation
        if random.random() < 0.7:
            angle = random.uniform(*config['rotation_range'])
            center = (image.shape[1] // 2, image.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), 
                                 borderValue=255)
        
        # Ink variation
        if config.get('ink_variation') and random.random() < 0.5:
            image = self._add_ink_variation(image)
        
        # Paper texture
        if config.get('paper_texture') and random.random() < 0.6:
            image = self._add_aged_paper_texture(image)
        
        return image
    
    def _apply_printed_effects(self, image: np.ndarray, config: Dict) -> np.ndarray:
        """Apply printed material effects."""
        # Print quality variation
        if config.get('print_quality_variation') and random.random() < 0.4:
            image = self._add_print_artifacts(image)
        
        # Scan artifacts
        if config.get('scan_artifacts') and random.random() < 0.3:
            image = self._add_scan_lines(image)
        
        return image
    
    def _apply_digital_effects(self, image: np.ndarray, config: Dict) -> np.ndarray:
        """Apply digital screen effects."""
        # Screen glare
        if config.get('screen_glare') and random.random() < 0.3:
            image = self._add_screen_glare(image)
        
        # Compression artifacts
        if config.get('compression_artifacts') and random.random() < 0.4:
            image = self._add_compression_artifacts(image)
        
        return image
    
    def _add_ink_variation(self, image: np.ndarray) -> np.ndarray:
        """Add ink density variations for handwritten text."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find text areas
        text_mask = gray < 200
        
        # Create ink variation
        variation = np.random.normal(1.0, 0.1, image.shape[:2])
        variation = cv2.GaussianBlur(variation, (3, 3), 0)
        
        result = image.copy()
        if len(image.shape) == 3:
            for c in range(3):
                result[:, :, c] = np.where(
                    text_mask,
                    np.clip(image[:, :, c] * variation, 0, 255).astype(np.uint8),
                    image[:, :, c]
                )
        else:
            result = np.where(
                text_mask,
                np.clip(image * variation, 0, 255).astype(np.uint8),
                image
            )
        
        return result
    
    def _add_aged_paper_texture(self, image: np.ndarray) -> np.ndarray:
        """Add aged paper texture."""
        rows, cols = image.shape[:2]
        
        # Create aging pattern
        aging = np.random.normal(0, 5, (rows, cols))
        aging = cv2.GaussianBlur(aging, (15, 15), 0)
        
        # Apply yellowing effect
        if len(image.shape) == 3:
            # Slight yellow tint
            result = image.copy().astype(np.float32)
            result[:, :, 0] *= 0.95  # Reduce blue
            result[:, :, 1] *= 0.98  # Slightly reduce green
            result = np.clip(result + aging[:, :, np.newaxis], 0, 255).astype(np.uint8)
        else:
            result = np.clip(image.astype(np.float32) + aging, 0, 255).astype(np.uint8)
        
        return result
    
    def _add_print_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Add printing artifacts like dot patterns."""
        # Add subtle halftone pattern
        rows, cols = image.shape[:2]
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Create dot pattern
        dot_pattern = np.sin(x * 0.3) * np.sin(y * 0.3)
        dot_pattern = (dot_pattern + 1) * 5  # Scale and shift
        
        result = image.copy().astype(np.float32)
        if len(image.shape) == 3:
            result += dot_pattern[:, :, np.newaxis]
        else:
            result += dot_pattern
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _add_scan_lines(self, image: np.ndarray) -> np.ndarray:
        """Add horizontal scan lines."""
        result = image.copy()
        
        # Add subtle horizontal lines
        for y in range(0, image.shape[0], random.randint(5, 15)):
            if random.random() < 0.3:
                result[y:y+1] = cv2.addWeighted(
                    result[y:y+1], 0.95,
                    np.full_like(result[y:y+1], 200), 0.05, 0
                )
        
        return result
    
    def _add_screen_glare(self, image: np.ndarray) -> np.ndarray:
        """Add screen glare effect."""
        rows, cols = image.shape[:2]
        
        # Create glare spot
        center_x = random.randint(cols // 4, 3 * cols // 4)
        center_y = random.randint(rows // 4, 3 * rows // 4)
        
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create glare mask
        glare_radius = min(rows, cols) // 3
        glare_mask = np.exp(-(distance / glare_radius)**2)
        glare_intensity = random.uniform(20, 60)
        
        result = image.copy().astype(np.float32)
        if len(image.shape) == 3:
            result += glare_mask[:, :, np.newaxis] * glare_intensity
        else:
            result += glare_mask * glare_intensity
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _add_compression_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Add JPEG compression artifacts."""
        # Apply JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(70, 90)]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR if len(image.shape) == 3 else cv2.IMREAD_GRAYSCALE)
        
        return decoded_img


def create_augmentation_dataset(input_dir: str, output_dir: str, 
                              augmentations_per_image: int = 5,
                              recipe_formats: List[str] = None) -> None:
    """
    Create augmented dataset from input images.
    
    Args:
        input_dir: Directory containing original images and labels
        output_dir: Directory to save augmented dataset
        augmentations_per_image: Number of augmentations per original image
        recipe_formats: List of recipe formats to simulate
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    (output_path / 'images').mkdir(exist_ok=True)
    (output_path / 'labels').mkdir(exist_ok=True)
    
    # Initialize augmenters
    text_augmenter = TextAugmentationPipeline()
    format_augmenter = RecipeFormatAugmenter()
    
    recipe_formats = recipe_formats or ['handwritten', 'printed', 'digital']
    
    # Process each image
    image_files = list((input_path / 'images').glob('*.jpg')) + list((input_path / 'images').glob('*.png'))
    
    augmented_count = 0
    for img_file in image_files:
        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        # Load corresponding label file
        label_file = input_path / 'labels' / f"{img_file.stem}.txt"
        boxes, class_labels = [], []
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        boxes.append([float(x) for x in parts[1:]])
        
        # Create multiple augmentations
        for i in range(augmentations_per_image):
            # Choose random format
            format_type = random.choice(recipe_formats)
            
            # Apply format-specific augmentation
            aug_image = format_augmenter.augment_for_format(image, format_type)
            
            # Apply general text augmentations
            aug_image, aug_boxes, aug_labels = text_augmenter.augment_image_with_boxes(
                aug_image, boxes, class_labels
            )
            
            # Apply text-specific effects
            aug_image = text_augmenter.apply_text_specific_augmentations(aug_image)
            
            # Save augmented image
            output_img_path = output_path / 'images' / f"{img_file.stem}_aug_{i:03d}.jpg"
            cv2.imwrite(str(output_img_path), aug_image)
            
            # Save augmented labels
            if aug_boxes:
                output_label_path = output_path / 'labels' / f"{img_file.stem}_aug_{i:03d}.txt"
                with open(output_label_path, 'w') as f:
                    for box, label in zip(aug_boxes, aug_labels):
                        f.write(f"{label} {' '.join(map(str, box))}\n")
            
            augmented_count += 1
    
    print(f"Created {augmented_count} augmented images in {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Text Augmentation Pipeline")
    print("Usage examples:")
    print("1. Create augmented dataset:")
    print("   create_augmentation_dataset('data/processed/train', 'data/augmented', 5)")
    print("2. Test individual augmentations:")
    print("   augmenter = TextAugmentationPipeline()")
    print("   augmented = augmenter.apply_text_specific_augmentations(image)")