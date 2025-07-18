#!/usr/bin/env python3
"""
HelloFresh Recipe AI - Object Detection Script
Performs object detection on images and videos using YOLOv8 models.
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np


class FoodDetector:
    """Food detection class using YOLOv8."""
    
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.25):
        """
        Initialize the detector.
        
        Args:
            model_path (str): Path to YOLO model file
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def detect_image(self, image_path, output_path=None, save_crops=False):
        """
        Detect objects in a single image.
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save output image
            save_crops (bool): Whether to save cropped detections
            
        Returns:
            dict: Detection results
        """
        results = self.model(image_path, conf=self.conf_threshold)
        
        if output_path:
            # Save annotated image
            annotated = results[0].plot()
            cv2.imwrite(output_path, annotated)
            
        if save_crops:
            # Save cropped detections
            crops_dir = Path(output_path).parent / "crops" if output_path else Path("results/crops")
            crops_dir.mkdir(exist_ok=True)
            results[0].save_crop(crops_dir)
            
        return self._parse_results(results[0])
    
    def detect_video(self, video_path, output_path=None):
        """
        Detect objects in video.
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video
            
        Returns:
            list: Detection results for each frame
        """
        cap = cv2.VideoCapture(video_path)
        frame_results = []
        
        if output_path:
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run detection
            results = self.model(frame, conf=self.conf_threshold)
            frame_results.append(self._parse_results(results[0]))
            
            if output_path:
                # Write annotated frame
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        if output_path:
            out.release()
            
        return frame_results
    
    def detect_batch(self, input_dir, output_dir=None):
        """
        Detect objects in batch of images.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save output images
            
        Returns:
            dict: Detection results for each image
        """
        input_path = Path(input_dir)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        
        results_dict = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for img_file in input_path.iterdir():
            if img_file.suffix.lower() in image_extensions:
                print(f"Processing {img_file.name}...")
                
                output_file = None
                if output_dir:
                    output_file = output_path / f"detected_{img_file.name}"
                
                results_dict[img_file.name] = self.detect_image(
                    str(img_file), str(output_file) if output_file else None
                )
        
        return results_dict
    
    def _parse_results(self, result):
        """Parse YOLO results into readable format."""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                detection = {
                    'bbox': boxes[i].tolist(),
                    'confidence': float(confidences[i]),
                    'class': int(classes[i]),
                    'class_name': result.names[int(classes[i])]
                }
                detections.append(detection)
        
        return {
            'detections': detections,
            'count': len(detections)
        }


def main():
    parser = argparse.ArgumentParser(description='HelloFresh Recipe AI - Object Detection')
    parser.add_argument('input', help='Input image, video, or directory path')
    parser.add_argument('--model', default='yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--output', help='Output path for results')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save-crops', action='store_true', help='Save cropped detections')
    parser.add_argument('--batch', action='store_true', help='Process directory of images')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = FoodDetector(args.model, args.conf)
    
    # Determine input type and process
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        sys.exit(1)
    
    if input_path.is_dir() or args.batch:
        # Batch processing
        print(f"Processing images in directory: {input_path}")
        results = detector.detect_batch(str(input_path), args.output)
        
        # Print summary
        total_detections = sum(r['count'] for r in results.values())
        print(f"\nProcessed {len(results)} images")
        print(f"Total detections: {total_detections}")
        
    elif input_path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
        # Video processing
        print(f"Processing video: {input_path}")
        results = detector.detect_video(str(input_path), args.output)
        
        total_detections = sum(r['count'] for r in results)
        print(f"Processed {len(results)} frames")
        print(f"Total detections: {total_detections}")
        
    else:
        # Single image processing
        print(f"Processing image: {input_path}")
        results = detector.detect_image(
            str(input_path), args.output, args.save_crops
        )
        
        print(f"Found {results['count']} objects:")
        for detection in results['detections']:
            print(f"  - {detection['class_name']}: {detection['confidence']:.2f}")


if __name__ == "__main__":
    main()