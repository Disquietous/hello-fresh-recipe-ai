#!/usr/bin/env python3
"""
Interactive annotation tool for creating text bounding boxes in recipe images.
Supports creating YOLO format annotations for text detection training.
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import os
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from utils.annotation_utils import AnnotationConverter, TextAnnotation


@dataclass
class BoundingBox:
    """Bounding box with metadata."""
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    text: str = ""
    confidence: float = 1.0
    
    def to_yolo_format(self, img_width: int, img_height: int) -> str:
        """Convert to YOLO format string."""
        x_center = (self.x1 + self.x2) / 2 / img_width
        y_center = (self.y1 + self.y2) / 2 / img_height
        width = (self.x2 - self.x1) / img_width
        height = (self.y2 - self.y1) / img_height
        
        return f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def get_area(self) -> int:
        """Get bounding box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class AnnotationTool:
    """Interactive annotation tool for text detection."""
    
    def __init__(self, image_path: str, output_dir: str = "annotations"):
        """
        Initialize annotation tool.
        
        Args:
            image_path: Path to image to annotate
            output_dir: Directory to save annotations
        """
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image
        self.original_image = cv2.imread(str(self.image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load image: {self.image_path}")
        
        self.image_height, self.image_width = self.original_image.shape[:2]
        self.display_image = self.original_image.copy()
        
        # Annotation state
        self.bounding_boxes: List[BoundingBox] = []
        self.current_box: Optional[BoundingBox] = None
        self.drawing = False
        self.start_point = None
        self.current_class_id = 0
        
        # Class definitions
        self.classes = {
            0: "ingredient_line",
            1: "ingredient_block", 
            2: "instruction_text",
            3: "recipe_title",
            4: "metadata_text"
        }
        
        self.class_colors = {
            0: (0, 255, 0),    # Green - ingredient_line
            1: (0, 255, 255),  # Yellow - ingredient_block
            2: (255, 0, 0),    # Blue - instruction_text
            3: (255, 0, 255),  # Magenta - recipe_title
            4: (0, 165, 255),  # Orange - metadata_text
        }
        
        # UI settings
        self.window_name = f"Annotation Tool - {self.image_path.name}"
        self.zoom_factor = 1.0
        self.display_width = 1200
        self.display_height = 800
        self.pan_x = 0
        self.pan_y = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load existing annotations if they exist
        self.load_existing_annotations()
        
        self.logger.info(f"Initialized annotation tool for: {self.image_path}")
    
    def load_existing_annotations(self):
        """Load existing annotations if they exist."""
        annotation_file = self.output_dir / f"{self.image_path.stem}.json"
        
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                
                for ann_data in data.get('annotations', []):
                    bbox = ann_data.get('bbox', [])
                    if len(bbox) == 4:
                        box = BoundingBox(
                            x1=bbox[0],
                            y1=bbox[1],
                            x2=bbox[0] + bbox[2],
                            y2=bbox[1] + bbox[3],
                            class_id=ann_data.get('class_id', 0),
                            class_name=ann_data.get('class_name', 'ingredient_line'),
                            text=ann_data.get('text', ''),
                            confidence=ann_data.get('confidence', 1.0)
                        )
                        self.bounding_boxes.append(box)
                
                self.logger.info(f"Loaded {len(self.bounding_boxes)} existing annotations")
                
            except Exception as e:
                self.logger.warning(f"Failed to load existing annotations: {e}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes."""
        # Convert display coordinates to image coordinates
        img_x = int((x - self.pan_x) / self.zoom_factor)
        img_y = int((y - self.pan_y) / self.zoom_factor)
        
        # Clamp to image bounds
        img_x = max(0, min(img_x, self.image_width - 1))
        img_y = max(0, min(img_y, self.image_height - 1))
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (img_x, img_y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Draw current box
            self.display_image = self.original_image.copy()
            self.draw_existing_boxes()
            
            # Draw current box being drawn
            cv2.rectangle(self.display_image, self.start_point, (img_x, img_y), 
                         self.class_colors[self.current_class_id], 2)
            
            # Add class label
            label = f"{self.classes[self.current_class_id]}"
            cv2.putText(self.display_image, label, 
                       (self.start_point[0], self.start_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       self.class_colors[self.current_class_id], 2)
        
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            
            # Create bounding box
            if self.start_point:
                x1, y1 = self.start_point
                x2, y2 = img_x, img_y
                
                # Ensure proper ordering
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # Check minimum size
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    # Ask for text content
                    text = self.get_text_input()
                    
                    box = BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        class_id=self.current_class_id,
                        class_name=self.classes[self.current_class_id],
                        text=text,
                        confidence=1.0
                    )
                    
                    self.bounding_boxes.append(box)
                    self.logger.info(f"Added bounding box: {box.class_name} at ({x1},{y1}) to ({x2},{y2})")
                
                self.start_point = None
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Delete box at this location
            self.delete_box_at_point(img_x, img_y)
        
        # Update display
        self.update_display()
    
    def get_text_input(self) -> str:
        """Get text input from user using tkinter dialog."""
        try:
            # Create root window (hidden)
            root = tk.Tk()
            root.withdraw()
            
            # Ask for text
            text = simpledialog.askstring(
                "Text Input", 
                f"Enter text for {self.classes[self.current_class_id]}:",
                initialvalue=""
            )
            
            root.destroy()
            
            return text if text else ""
            
        except Exception as e:
            self.logger.warning(f"Failed to get text input: {e}")
            return ""
    
    def delete_box_at_point(self, x: int, y: int):
        """Delete bounding box at given point."""
        for i, box in enumerate(self.bounding_boxes):
            if box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2:
                deleted_box = self.bounding_boxes.pop(i)
                self.logger.info(f"Deleted bounding box: {deleted_box.class_name}")
                break
    
    def draw_existing_boxes(self):
        """Draw all existing bounding boxes."""
        for box in self.bounding_boxes:
            color = self.class_colors[box.class_id]
            
            # Draw rectangle
            cv2.rectangle(self.display_image, (box.x1, box.y1), (box.x2, box.y2), color, 2)
            
            # Draw label
            label = f"{box.class_name}"
            if box.text:
                label += f": {box.text[:20]}..."
            
            # Background for text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(self.display_image, 
                         (box.x1, box.y1 - text_height - 10),
                         (box.x1 + text_width, box.y1),
                         color, -1)
            
            # Text
            cv2.putText(self.display_image, label, 
                       (box.x1, box.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def update_display(self):
        """Update the display with current annotations."""
        # Create display image
        display = self.display_image.copy()
        
        # Apply zoom and pan
        if self.zoom_factor != 1.0:
            new_width = int(self.image_width * self.zoom_factor)
            new_height = int(self.image_height * self.zoom_factor)
            display = cv2.resize(display, (new_width, new_height))
        
        # Create canvas
        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Calculate placement
        y_offset = max(0, (self.display_height - display.shape[0]) // 2)
        x_offset = max(0, (self.display_width - display.shape[1]) // 2)
        
        # Place image on canvas
        end_y = min(self.display_height, y_offset + display.shape[0])
        end_x = min(self.display_width, x_offset + display.shape[1])
        
        canvas[y_offset:end_y, x_offset:end_x] = display[:end_y-y_offset, :end_x-x_offset]
        
        # Add UI overlay
        self.draw_ui_overlay(canvas)
        
        cv2.imshow(self.window_name, canvas)
    
    def draw_ui_overlay(self, canvas: np.ndarray):
        """Draw UI overlay with instructions and class info."""
        # Instructions
        instructions = [
            f"Current class: {self.classes[self.current_class_id]} ({self.current_class_id})",
            f"Boxes: {len(self.bounding_boxes)}",
            "",
            "Controls:",
            "Left click + drag: Draw box",
            "Right click: Delete box",
            "1-5: Change class",
            "S: Save annotations",
            "R: Reset view",
            "Q: Quit"
        ]
        
        # Draw instructions
        y_start = 30
        for i, line in enumerate(instructions):
            y_pos = y_start + i * 25
            if y_pos >= canvas.shape[0] - 10:
                break
            
            # Background
            if line:
                (text_width, text_height), _ = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(canvas, (10, y_pos - text_height - 5),
                             (15 + text_width, y_pos + 5), (0, 0, 0), -1)
            
            # Text
            color = self.class_colors[self.current_class_id] if i == 0 else (255, 255, 255)
            cv2.putText(canvas, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def handle_key(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Args:
            key: Key code
            
        Returns:
            True to continue, False to exit
        """
        if key == ord('q') or key == 27:  # Q or ESC
            return False
        
        elif key == ord('s'):  # Save
            self.save_annotations()
        
        elif key == ord('r'):  # Reset view
            self.zoom_factor = 1.0
            self.pan_x = 0
            self.pan_y = 0
        
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:  # Change class
            new_class = key - ord('1')
            if new_class in self.classes:
                self.current_class_id = new_class
                self.logger.info(f"Changed to class: {self.classes[self.current_class_id]}")
        
        elif key == ord('d'):  # Delete all boxes
            if messagebox.askyesno("Confirm", "Delete all annotations?"):
                self.bounding_boxes.clear()
                self.logger.info("Deleted all annotations")
        
        elif key == ord('h'):  # Help
            self.show_help()
        
        return True
    
    def show_help(self):
        """Show help dialog."""
        help_text = """
Annotation Tool Help:

Mouse Controls:
- Left click + drag: Draw bounding box
- Right click: Delete box at cursor

Keyboard Controls:
- 1-5: Change annotation class
- S: Save annotations
- R: Reset view
- D: Delete all annotations
- H: Show this help
- Q/ESC: Quit

Classes:
1. ingredient_line - Single ingredient lines
2. ingredient_block - Multi-line ingredient sections  
3. instruction_text - Cooking instructions
4. recipe_title - Recipe titles and headings
5. metadata_text - Serving size, time, etc.

Tips:
- Draw tight bounding boxes around text
- Use appropriate classes for different text types
- Save frequently to avoid losing work
- Right-click to delete incorrect boxes
"""
        
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("Help", help_text)
            root.destroy()
        except Exception as e:
            print(help_text)
    
    def save_annotations(self):
        """Save annotations to JSON and YOLO format."""
        try:
            # Save JSON format
            json_data = {
                'image_id': self.image_path.name,
                'image_path': str(self.image_path),
                'image_width': self.image_width,
                'image_height': self.image_height,
                'annotations': []
            }
            
            for box in self.bounding_boxes:
                annotation = {
                    'class_id': box.class_id,
                    'class_name': box.class_name,
                    'bbox': [box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1],
                    'text': box.text,
                    'confidence': box.confidence,
                    'area': box.get_area()
                }
                json_data['annotations'].append(annotation)
            
            # Save JSON
            json_file = self.output_dir / f"{self.image_path.stem}.json"
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            # Save YOLO format
            yolo_file = self.output_dir / f"{self.image_path.stem}.txt"
            with open(yolo_file, 'w') as f:
                for box in self.bounding_boxes:
                    yolo_line = box.to_yolo_format(self.image_width, self.image_height)
                    f.write(yolo_line + '\n')
            
            self.logger.info(f"Saved {len(self.bounding_boxes)} annotations to {json_file} and {yolo_file}")
            
            # Show confirmation
            try:
                root = tk.Tk()
                root.withdraw()
                messagebox.showinfo("Saved", f"Saved {len(self.bounding_boxes)} annotations")
                root.destroy()
            except:
                print(f"Saved {len(self.bounding_boxes)} annotations")
            
        except Exception as e:
            self.logger.error(f"Failed to save annotations: {e}")
            try:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Error", f"Failed to save: {e}")
                root.destroy()
            except:
                print(f"Failed to save: {e}")
    
    def run(self):
        """Run the annotation tool."""
        self.logger.info("Starting annotation tool...")
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Initial display
        self.display_image = self.original_image.copy()
        self.draw_existing_boxes()
        self.update_display()
        
        # Main loop
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key != 255:  # Key pressed
                if not self.handle_key(key):
                    break
            
            # Update display
            self.display_image = self.original_image.copy()
            self.draw_existing_boxes()
            self.update_display()
        
        # Cleanup
        cv2.destroyAllWindows()
        self.logger.info("Annotation tool closed")


class BatchAnnotationTool:
    """Tool for batch annotation of multiple images."""
    
    def __init__(self, image_dir: str, output_dir: str = "annotations"):
        """
        Initialize batch annotation tool.
        
        Args:
            image_dir: Directory containing images to annotate
            output_dir: Directory to save annotations
        """
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find images
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.image_files = []
        
        for ext in self.image_extensions:
            self.image_files.extend(list(self.image_dir.glob(f'*{ext}')))
            self.image_files.extend(list(self.image_dir.glob(f'*{ext.upper()}')))
        
        self.image_files.sort()
        self.current_index = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Found {len(self.image_files)} images to annotate")
    
    def run(self):
        """Run batch annotation."""
        if not self.image_files:
            print("No images found to annotate")
            return
        
        print(f"Starting batch annotation of {len(self.image_files)} images")
        print("Controls: N=Next, P=Previous, Q=Quit")
        
        while self.current_index < len(self.image_files):
            current_image = self.image_files[self.current_index]
            
            print(f"\nAnnotating image {self.current_index + 1}/{len(self.image_files)}: {current_image.name}")
            
            # Run annotation tool for current image
            try:
                tool = AnnotationTool(str(current_image), str(self.output_dir))
                tool.run()
                
                # Ask what to do next
                while True:
                    try:
                        choice = input("Next (n), Previous (p), Quit (q): ").lower().strip()
                        
                        if choice in ['n', 'next', '']:
                            self.current_index += 1
                            break
                        elif choice in ['p', 'previous', 'prev']:
                            self.current_index = max(0, self.current_index - 1)
                            break
                        elif choice in ['q', 'quit', 'exit']:
                            print("Exiting batch annotation")
                            return
                        else:
                            print("Invalid choice. Please enter n, p, or q.")
                    
                    except KeyboardInterrupt:
                        print("\nExiting batch annotation")
                        return
                
            except Exception as e:
                print(f"Error annotating {current_image}: {e}")
                self.current_index += 1
        
        print(f"\nCompleted batch annotation of {len(self.image_files)} images")


def main():
    """Main function for annotation tool."""
    parser = argparse.ArgumentParser(description='Text Detection Annotation Tool')
    parser.add_argument('--image', '-i', help='Single image to annotate')
    parser.add_argument('--image-dir', '-d', help='Directory of images to annotate')
    parser.add_argument('--output-dir', '-o', default='annotations', help='Output directory')
    parser.add_argument('--batch', '-b', action='store_true', help='Batch annotation mode')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.batch or args.image_dir:
            # Batch mode
            if not args.image_dir:
                print("Error: --image-dir required for batch mode")
                return 1
            
            if not Path(args.image_dir).exists():
                print(f"Error: Image directory not found: {args.image_dir}")
                return 1
            
            batch_tool = BatchAnnotationTool(args.image_dir, args.output_dir)
            batch_tool.run()
        
        elif args.image:
            # Single image mode
            if not Path(args.image).exists():
                print(f"Error: Image not found: {args.image}")
                return 1
            
            tool = AnnotationTool(args.image, args.output_dir)
            tool.run()
        
        else:
            # Interactive file selection
            try:
                root = tk.Tk()
                root.withdraw()
                
                image_path = filedialog.askopenfilename(
                    title="Select image to annotate",
                    filetypes=[
                        ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                        ("All files", "*.*")
                    ]
                )
                
                root.destroy()
                
                if image_path:
                    tool = AnnotationTool(image_path, args.output_dir)
                    tool.run()
                else:
                    print("No image selected")
                    return 1
                
            except Exception as e:
                print(f"Error in file selection: {e}")
                parser.print_help()
                return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())