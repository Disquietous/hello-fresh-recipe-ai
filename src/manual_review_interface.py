#!/usr/bin/env python3
"""
Manual Review Interface for Error Correction
Provides a GUI interface for manual review and correction of text extraction errors.
Supports annotation of detection errors, OCR corrections, and ingredient parsing fixes.
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from comprehensive_evaluation_system import ComprehensiveEvaluationSystem
from ingredient_pipeline import IngredientExtractionPipeline


class ManualReviewInterface:
    """GUI interface for manual review and correction of text extraction errors."""
    
    def __init__(self):
        """Initialize the manual review interface."""
        self.root = tk.Tk()
        self.root.title("Recipe Text Extraction - Manual Review Interface")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.current_image_path = None
        self.current_image = None
        self.current_results = None
        self.review_data = {}
        self.corrections = {}
        self.image_list = []
        self.current_image_index = 0
        
        # Initialize extraction pipeline
        self.extraction_pipeline = IngredientExtractionPipeline()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Create GUI components
        self._create_gui()
        
        # Load initial data
        self._load_initial_data()
        
        self.logger.info("Manual Review Interface initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the interface."""
        logger = logging.getLogger('manual_review_interface')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_gui(self):
        """Create the GUI components."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Top control panel
        self._create_control_panel(main_frame)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Left side - Image display
        self._create_image_panel(content_frame)
        
        # Right side - Review and correction panel
        self._create_review_panel(content_frame)
        
        # Bottom status bar
        self._create_status_bar(main_frame)
    
    def _create_control_panel(self, parent):
        """Create the top control panel."""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # File operations
        ttk.Button(control_frame, text="Load Images", command=self._load_images).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Load Results", command=self._load_results).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Save Corrections", command=self._save_corrections).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Export Report", command=self._export_report).grid(row=0, column=3, padx=5)
        
        # Separator
        ttk.Separator(control_frame, orient='vertical').grid(row=0, column=4, sticky=(tk.N, tk.S), padx=10)
        
        # Image navigation
        ttk.Button(control_frame, text="Previous", command=self._previous_image).grid(row=0, column=5, padx=5)
        ttk.Button(control_frame, text="Next", command=self._next_image).grid(row=0, column=6, padx=5)
        
        # Image counter
        self.image_counter_var = tk.StringVar(value="0 / 0")
        ttk.Label(control_frame, textvariable=self.image_counter_var).grid(row=0, column=7, padx=10)
        
        # Separator
        ttk.Separator(control_frame, orient='vertical').grid(row=0, column=8, sticky=(tk.N, tk.S), padx=10)
        
        # Quick actions
        ttk.Button(control_frame, text="Reprocess Image", command=self._reprocess_image).grid(row=0, column=9, padx=5)
        ttk.Button(control_frame, text="Mark as Reviewed", command=self._mark_reviewed).grid(row=0, column=10, padx=5)
    
    def _create_image_panel(self, parent):
        """Create the image display panel."""
        image_frame = ttk.LabelFrame(parent, text="Image Display", padding="10")
        image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Image canvas with scrollbars
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        self.image_canvas = tk.Canvas(canvas_frame, bg='white')
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.image_canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Bind mouse events for region selection
        self.image_canvas.bind("<Button-1>", self._on_canvas_click)
        self.image_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        
        # Image info
        info_frame = ttk.Frame(image_frame)
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.image_info_var = tk.StringVar(value="No image loaded")
        ttk.Label(info_frame, textvariable=self.image_info_var).grid(row=0, column=0, sticky=tk.W)
        
        # Zoom controls
        zoom_frame = ttk.Frame(info_frame)
        zoom_frame.grid(row=0, column=1, sticky=tk.E)
        
        ttk.Button(zoom_frame, text="Zoom In", command=self._zoom_in).grid(row=0, column=0, padx=2)
        ttk.Button(zoom_frame, text="Zoom Out", command=self._zoom_out).grid(row=0, column=1, padx=2)
        ttk.Button(zoom_frame, text="Reset Zoom", command=self._reset_zoom).grid(row=0, column=2, padx=2)
        
        self.zoom_scale = 1.0
        self.selection_start = None
        self.selection_rect = None
    
    def _create_review_panel(self, parent):
        """Create the review and correction panel."""
        review_frame = ttk.LabelFrame(parent, text="Review & Corrections", padding="10")
        review_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        review_frame.columnconfigure(0, weight=1)
        review_frame.rowconfigure(0, weight=1)
        
        # Notebook for different tabs
        self.notebook = ttk.Notebook(review_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Detection Results Tab
        self._create_detection_tab()
        
        # OCR Results Tab
        self._create_ocr_tab()
        
        # Ingredient Parsing Tab
        self._create_parsing_tab()
        
        # Quality Review Tab
        self._create_quality_tab()
        
        # Comments Tab
        self._create_comments_tab()
    
    def _create_detection_tab(self):
        """Create the detection results tab."""
        detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(detection_frame, text="Detection")
        
        detection_frame.columnconfigure(0, weight=1)
        detection_frame.rowconfigure(1, weight=1)
        
        # Detection summary
        summary_frame = ttk.LabelFrame(detection_frame, text="Detection Summary", padding="5")
        summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.detection_summary_var = tk.StringVar(value="No detection results")
        ttk.Label(summary_frame, textvariable=self.detection_summary_var).grid(row=0, column=0, sticky=tk.W)
        
        # Detection results list
        results_frame = ttk.LabelFrame(detection_frame, text="Detection Results", padding="5")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Treeview for detection results
        columns = ('ID', 'Confidence', 'Class', 'Bbox', 'Status')
        self.detection_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=10)
        
        # Configure columns
        for col in columns:
            self.detection_tree.heading(col, text=col)
            self.detection_tree.column(col, width=100)
        
        # Scrollbars for treeview
        det_v_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.detection_tree.yview)
        det_v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.detection_tree.configure(yscrollcommand=det_v_scroll.set)
        
        self.detection_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Detection action buttons
        det_buttons_frame = ttk.Frame(results_frame)
        det_buttons_frame.grid(row=1, column=0, columnspan=2, pady=(5, 0))
        
        ttk.Button(det_buttons_frame, text="Mark Correct", command=self._mark_detection_correct).grid(row=0, column=0, padx=5)
        ttk.Button(det_buttons_frame, text="Mark Incorrect", command=self._mark_detection_incorrect).grid(row=0, column=1, padx=5)
        ttk.Button(det_buttons_frame, text="Add Missing", command=self._add_missing_detection).grid(row=0, column=2, padx=5)
        ttk.Button(det_buttons_frame, text="Delete", command=self._delete_detection).grid(row=0, column=3, padx=5)
        
        # Bind selection event
        self.detection_tree.bind('<<TreeviewSelect>>', self._on_detection_select)
    
    def _create_ocr_tab(self):
        """Create the OCR results tab."""
        ocr_frame = ttk.Frame(self.notebook)
        self.notebook.add(ocr_frame, text="OCR")
        
        ocr_frame.columnconfigure(0, weight=1)
        ocr_frame.rowconfigure(1, weight=1)
        
        # OCR summary
        summary_frame = ttk.LabelFrame(ocr_frame, text="OCR Summary", padding="5")
        summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.ocr_summary_var = tk.StringVar(value="No OCR results")
        ttk.Label(summary_frame, textvariable=self.ocr_summary_var).grid(row=0, column=0, sticky=tk.W)
        
        # OCR results list
        results_frame = ttk.LabelFrame(ocr_frame, text="OCR Results", padding="5")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Treeview for OCR results
        columns = ('ID', 'Extracted Text', 'Confidence', 'Status')
        self.ocr_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=8)
        
        # Configure columns
        for col in columns:
            self.ocr_tree.heading(col, text=col)
            if col == 'Extracted Text':
                self.ocr_tree.column(col, width=300)
            else:
                self.ocr_tree.column(col, width=100)
        
        # Scrollbars for treeview
        ocr_v_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.ocr_tree.yview)
        ocr_v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.ocr_tree.configure(yscrollcommand=ocr_v_scroll.set)
        
        self.ocr_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # OCR correction frame
        correction_frame = ttk.LabelFrame(results_frame, text="Text Correction", padding="5")
        correction_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        correction_frame.columnconfigure(1, weight=1)
        
        ttk.Label(correction_frame, text="Corrected Text:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.corrected_text_var = tk.StringVar()
        corrected_entry = ttk.Entry(correction_frame, textvariable=self.corrected_text_var, width=40)
        corrected_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(correction_frame, text="Apply Correction", command=self._apply_ocr_correction).grid(row=0, column=2, padx=5)
        
        # Bind selection event
        self.ocr_tree.bind('<<TreeviewSelect>>', self._on_ocr_select)
    
    def _create_parsing_tab(self):
        """Create the ingredient parsing tab."""
        parsing_frame = ttk.Frame(self.notebook)
        self.notebook.add(parsing_frame, text="Parsing")
        
        parsing_frame.columnconfigure(0, weight=1)
        parsing_frame.rowconfigure(1, weight=1)
        
        # Parsing summary
        summary_frame = ttk.LabelFrame(parsing_frame, text="Parsing Summary", padding="5")
        summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.parsing_summary_var = tk.StringVar(value="No parsing results")
        ttk.Label(summary_frame, textvariable=self.parsing_summary_var).grid(row=0, column=0, sticky=tk.W)
        
        # Parsing results list
        results_frame = ttk.LabelFrame(parsing_frame, text="Parsed Ingredients", padding="5")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Treeview for parsing results
        columns = ('ID', 'Ingredient', 'Quantity', 'Unit', 'Confidence', 'Status')
        self.parsing_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=8)
        
        # Configure columns
        for col in columns:
            self.parsing_tree.heading(col, text=col)
            if col == 'Ingredient':
                self.parsing_tree.column(col, width=200)
            else:
                self.parsing_tree.column(col, width=100)
        
        # Scrollbars for treeview
        parsing_v_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.parsing_tree.yview)
        parsing_v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.parsing_tree.configure(yscrollcommand=parsing_v_scroll.set)
        
        self.parsing_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Ingredient correction frame
        correction_frame = ttk.LabelFrame(results_frame, text="Ingredient Correction", padding="5")
        correction_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        correction_frame.columnconfigure(1, weight=1)
        correction_frame.columnconfigure(3, weight=1)
        correction_frame.columnconfigure(5, weight=1)
        
        ttk.Label(correction_frame, text="Ingredient:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.corrected_ingredient_var = tk.StringVar()
        ttk.Entry(correction_frame, textvariable=self.corrected_ingredient_var, width=20).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Label(correction_frame, text="Quantity:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.corrected_quantity_var = tk.StringVar()
        ttk.Entry(correction_frame, textvariable=self.corrected_quantity_var, width=10).grid(row=0, column=3, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Label(correction_frame, text="Unit:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.corrected_unit_var = tk.StringVar()
        ttk.Entry(correction_frame, textvariable=self.corrected_unit_var, width=10).grid(row=0, column=5, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(correction_frame, text="Apply Correction", command=self._apply_parsing_correction).grid(row=0, column=6, padx=5)
        
        # Bind selection event
        self.parsing_tree.bind('<<TreeviewSelect>>', self._on_parsing_select)
    
    def _create_quality_tab(self):
        """Create the quality review tab."""
        quality_frame = ttk.Frame(self.notebook)
        self.notebook.add(quality_frame, text="Quality")
        
        quality_frame.columnconfigure(0, weight=1)
        quality_frame.rowconfigure(1, weight=1)
        
        # Overall quality assessment
        assessment_frame = ttk.LabelFrame(quality_frame, text="Overall Quality Assessment", padding="10")
        assessment_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Quality scores
        scores_frame = ttk.Frame(assessment_frame)
        scores_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(scores_frame, text="Detection Quality:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.detection_quality_var = tk.StringVar(value="N/A")
        ttk.Label(scores_frame, textvariable=self.detection_quality_var).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(scores_frame, text="OCR Quality:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.ocr_quality_var = tk.StringVar(value="N/A")
        ttk.Label(scores_frame, textvariable=self.ocr_quality_var).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(scores_frame, text="Parsing Quality:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        self.parsing_quality_var = tk.StringVar(value="N/A")
        ttk.Label(scores_frame, textvariable=self.parsing_quality_var).grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(scores_frame, text="Overall Quality:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10))
        self.overall_quality_var = tk.StringVar(value="N/A")
        ttk.Label(scores_frame, textvariable=self.overall_quality_var, font=('TkDefaultFont', 10, 'bold')).grid(row=3, column=1, sticky=tk.W)
        
        # Manual quality rating
        rating_frame = ttk.LabelFrame(quality_frame, text="Manual Quality Rating", padding="10")
        rating_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(rating_frame, text="Manual Rating (1-5):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.manual_rating_var = tk.IntVar(value=3)
        rating_scale = ttk.Scale(rating_frame, from_=1, to=5, orient=tk.HORIZONTAL, variable=self.manual_rating_var)
        rating_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.rating_label_var = tk.StringVar(value="3")
        ttk.Label(rating_frame, textvariable=self.rating_label_var).grid(row=0, column=2, sticky=tk.W)
        
        # Bind scale change
        rating_scale.bind('<Motion>', self._on_rating_change)
        rating_scale.bind('<ButtonRelease-1>', self._on_rating_change)
        
        # Quality categories
        categories_frame = ttk.LabelFrame(rating_frame, text="Quality Categories", padding="5")
        categories_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Checkboxes for quality issues
        self.quality_issues = {}
        issues = ['Poor Image Quality', 'Complex Layout', 'Handwritten Text', 'Multiple Languages', 'Overlapping Text']
        
        for i, issue in enumerate(issues):
            var = tk.BooleanVar()
            ttk.Checkbutton(categories_frame, text=issue, variable=var).grid(row=i//2, column=i%2, sticky=tk.W, padx=5, pady=2)
            self.quality_issues[issue] = var
    
    def _create_comments_tab(self):
        """Create the comments tab."""
        comments_frame = ttk.Frame(self.notebook)
        self.notebook.add(comments_frame, text="Comments")
        
        comments_frame.columnconfigure(0, weight=1)
        comments_frame.rowconfigure(1, weight=1)
        
        # Comments header
        header_frame = ttk.Frame(comments_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(header_frame, text="Review Comments and Notes", font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=0, sticky=tk.W)
        
        # Comments text area
        text_frame = ttk.LabelFrame(comments_frame, text="Comments", padding="5")
        text_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        # Text widget with scrollbar
        self.comments_text = tk.Text(text_frame, wrap=tk.WORD, width=50, height=15)
        self.comments_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        comments_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.comments_text.yview)
        comments_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.comments_text.configure(yscrollcommand=comments_scroll.set)
        
        # Comments buttons
        buttons_frame = ttk.Frame(comments_frame)
        buttons_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(buttons_frame, text="Save Comments", command=self._save_comments).grid(row=0, column=0, padx=5)
        ttk.Button(buttons_frame, text="Clear Comments", command=self._clear_comments).grid(row=0, column=1, padx=5)
        ttk.Button(buttons_frame, text="Load Template", command=self._load_comment_template).grid(row=0, column=2, padx=5)
    
    def _create_status_bar(self, parent):
        """Create the status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def _load_initial_data(self):
        """Load initial data if available."""
        # Check for default directories
        default_dirs = ["results", "evaluation_results", "manual_review"]
        for dir_name in default_dirs:
            if Path(dir_name).exists():
                self._load_from_directory(dir_name)
                break
    
    def _load_images(self):
        """Load images for review."""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self._load_from_directory(directory)
    
    def _load_from_directory(self, directory):
        """Load images from a directory."""
        directory = Path(directory)
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.image_list = []
        
        for ext in image_extensions:
            self.image_list.extend(directory.glob(f"*{ext}"))
            self.image_list.extend(directory.glob(f"*{ext.upper()}"))
        
        if self.image_list:
            self.current_image_index = 0
            self._update_image_counter()
            self._load_current_image()
            self.status_var.set(f"Loaded {len(self.image_list)} images from {directory}")
        else:
            messagebox.showwarning("No Images", "No image files found in the selected directory.")
    
    def _load_results(self):
        """Load existing results for review."""
        file_path = filedialog.askopenfilename(
            title="Select Results File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.review_data = json.load(f)
                self.status_var.set(f"Loaded results from {file_path}")
                self._update_display()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load results: {str(e)}")
    
    def _save_corrections(self):
        """Save corrections to file."""
        if not self.corrections:
            messagebox.showinfo("No Corrections", "No corrections to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Corrections",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                corrections_data = {
                    'corrections': self.corrections,
                    'timestamp': datetime.now().isoformat(),
                    'total_images_reviewed': len(self.corrections),
                    'review_metadata': {
                        'interface_version': '1.0',
                        'reviewer': 'manual_review_interface'
                    }
                }
                
                with open(file_path, 'w') as f:
                    json.dump(corrections_data, f, indent=2)
                
                self.status_var.set(f"Corrections saved to {file_path}")
                messagebox.showinfo("Success", f"Corrections saved successfully to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save corrections: {str(e)}")
    
    def _export_report(self):
        """Export review report."""
        if not self.corrections:
            messagebox.showinfo("No Data", "No review data to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Review Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self._generate_review_report(file_path)
                self.status_var.set(f"Report exported to {file_path}")
                messagebox.showinfo("Success", f"Report exported successfully to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {str(e)}")
    
    def _generate_review_report(self, file_path):
        """Generate a comprehensive review report."""
        report_lines = [
            "Manual Review Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Images Reviewed: {len(self.corrections)}",
            "",
            "Review Summary:",
            "-" * 20,
        ]
        
        # Statistics
        total_corrections = 0
        detection_corrections = 0
        ocr_corrections = 0
        parsing_corrections = 0
        
        for image_path, corrections in self.corrections.items():
            if 'detection_corrections' in corrections:
                detection_corrections += len(corrections['detection_corrections'])
                total_corrections += len(corrections['detection_corrections'])
            
            if 'ocr_corrections' in corrections:
                ocr_corrections += len(corrections['ocr_corrections'])
                total_corrections += len(corrections['ocr_corrections'])
            
            if 'parsing_corrections' in corrections:
                parsing_corrections += len(corrections['parsing_corrections'])
                total_corrections += len(corrections['parsing_corrections'])
        
        report_lines.extend([
            f"Total Corrections Made: {total_corrections}",
            f"Detection Corrections: {detection_corrections}",
            f"OCR Corrections: {ocr_corrections}",
            f"Parsing Corrections: {parsing_corrections}",
            "",
            "Detailed Corrections:",
            "-" * 30,
        ])
        
        # Detailed corrections
        for image_path, corrections in self.corrections.items():
            report_lines.extend([
                f"\nImage: {Path(image_path).name}",
                f"Path: {image_path}",
            ])
            
            if 'manual_rating' in corrections:
                report_lines.append(f"Manual Rating: {corrections['manual_rating']}/5")
            
            if 'quality_issues' in corrections:
                issues = [issue for issue, present in corrections['quality_issues'].items() if present]
                if issues:
                    report_lines.append(f"Quality Issues: {', '.join(issues)}")
            
            if 'comments' in corrections and corrections['comments']:
                report_lines.append(f"Comments: {corrections['comments']}")
            
            if 'detection_corrections' in corrections:
                report_lines.append(f"Detection Corrections: {len(corrections['detection_corrections'])}")
            
            if 'ocr_corrections' in corrections:
                report_lines.append(f"OCR Corrections: {len(corrections['ocr_corrections'])}")
            
            if 'parsing_corrections' in corrections:
                report_lines.append(f"Parsing Corrections: {len(corrections['parsing_corrections'])}")
        
        # Write report
        with open(file_path, 'w') as f:
            f.write('\n'.join(report_lines))
    
    def _update_image_counter(self):
        """Update the image counter display."""
        if self.image_list:
            self.image_counter_var.set(f"{self.current_image_index + 1} / {len(self.image_list)}")
        else:
            self.image_counter_var.set("0 / 0")
    
    def _previous_image(self):
        """Go to previous image."""
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self._update_image_counter()
            self._load_current_image()
    
    def _next_image(self):
        """Go to next image."""
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self._update_image_counter()
            self._load_current_image()
    
    def _load_current_image(self):
        """Load and display the current image."""
        if not self.image_list:
            return
        
        try:
            self.current_image_path = str(self.image_list[self.current_image_index])
            self.current_image = cv2.imread(self.current_image_path)
            
            if self.current_image is None:
                messagebox.showerror("Error", f"Could not load image: {self.current_image_path}")
                return
            
            # Update image info
            height, width = self.current_image.shape[:2]
            self.image_info_var.set(f"Image: {Path(self.current_image_path).name} ({width}x{height})")
            
            # Display image
            self._display_image()
            
            # Process image and update results
            self._process_current_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def _display_image(self):
        """Display the current image on the canvas."""
        if self.current_image is None:
            return
        
        # Convert OpenCV image to PIL
        image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Apply zoom
        if self.zoom_scale != 1.0:
            new_width = int(pil_image.width * self.zoom_scale)
            new_height = int(pil_image.height * self.zoom_scale)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        
        # Update scroll region
        self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
    
    def _process_current_image(self):
        """Process the current image and update results."""
        if not self.current_image_path:
            return
        
        try:
            # Process image with extraction pipeline
            results = self.extraction_pipeline.process_image(self.current_image_path)
            self.current_results = results
            
            # Update displays
            self._update_detection_display(results)
            self._update_ocr_display(results)
            self._update_parsing_display(results)
            self._update_quality_display(results)
            
            # Load existing corrections if available
            self._load_existing_corrections()
            
        except Exception as e:
            self.logger.error(f"Failed to process image: {str(e)}")
            messagebox.showerror("Processing Error", f"Failed to process image: {str(e)}")
    
    def _update_detection_display(self, results):
        """Update the detection results display."""
        # Clear existing items
        for item in self.detection_tree.get_children():
            self.detection_tree.delete(item)
        
        # Update summary
        total_detections = len(results.get('ingredients', []))
        high_conf = len([ing for ing in results.get('ingredients', []) if ing['confidence_scores']['overall'] >= 0.8])
        self.detection_summary_var.set(f"Total Detections: {total_detections}, High Confidence: {high_conf}")
        
        # Add detection results
        for i, ingredient in enumerate(results.get('ingredients', [])):
            bbox = ingredient['bounding_box']
            bbox_str = f"({bbox['x1']},{bbox['y1']})-({bbox['x2']},{bbox['y2']})"
            
            self.detection_tree.insert('', 'end', values=(
                i,
                f"{ingredient['confidence_scores']['text_detection']:.3f}",
                ingredient['detection_class'],
                bbox_str,
                'Pending'
            ))
    
    def _update_ocr_display(self, results):
        """Update the OCR results display."""
        # Clear existing items
        for item in self.ocr_tree.get_children():
            self.ocr_tree.delete(item)
        
        # Update summary
        total_texts = len(results.get('ingredients', []))
        avg_conf = np.mean([ing['confidence_scores']['ocr_quality'] for ing in results.get('ingredients', [])]) if results.get('ingredients') else 0
        self.ocr_summary_var.set(f"Total Texts: {total_texts}, Avg Confidence: {avg_conf:.3f}")
        
        # Add OCR results
        for i, ingredient in enumerate(results.get('ingredients', [])):
            self.ocr_tree.insert('', 'end', values=(
                i,
                ingredient['raw_text'],
                f"{ingredient['confidence_scores']['ocr_quality']:.3f}",
                'Pending'
            ))
    
    def _update_parsing_display(self, results):
        """Update the parsing results display."""
        # Clear existing items
        for item in self.parsing_tree.get_children():
            self.parsing_tree.delete(item)
        
        # Update summary
        total_ingredients = len(results.get('ingredients', []))
        valid_ingredients = len([ing for ing in results.get('ingredients', []) if ing['ingredient_name']])
        self.parsing_summary_var.set(f"Total Parsed: {total_ingredients}, Valid: {valid_ingredients}")
        
        # Add parsing results
        for i, ingredient in enumerate(results.get('ingredients', [])):
            self.parsing_tree.insert('', 'end', values=(
                i,
                ingredient['ingredient_name'],
                ingredient['quantity'],
                ingredient['unit'],
                f"{ingredient['confidence_scores']['ingredient_recognition']:.3f}",
                'Pending'
            ))
    
    def _update_quality_display(self, results):
        """Update the quality assessment display."""
        # Calculate quality scores
        if results.get('ingredients'):
            detection_quality = np.mean([ing['confidence_scores']['text_detection'] for ing in results['ingredients']])
            ocr_quality = np.mean([ing['confidence_scores']['ocr_quality'] for ing in results['ingredients']])
            parsing_quality = np.mean([ing['confidence_scores']['ingredient_recognition'] for ing in results['ingredients']])
            overall_quality = np.mean([ing['confidence_scores']['overall'] for ing in results['ingredients']])
            
            self.detection_quality_var.set(f"{detection_quality:.3f}")
            self.ocr_quality_var.set(f"{ocr_quality:.3f}")
            self.parsing_quality_var.set(f"{parsing_quality:.3f}")
            self.overall_quality_var.set(f"{overall_quality:.3f}")
        else:
            self.detection_quality_var.set("0.000")
            self.ocr_quality_var.set("0.000")
            self.parsing_quality_var.set("0.000")
            self.overall_quality_var.set("0.000")
    
    def _load_existing_corrections(self):
        """Load existing corrections for the current image."""
        if self.current_image_path in self.corrections:
            corrections = self.corrections[self.current_image_path]
            
            # Load manual rating
            if 'manual_rating' in corrections:
                self.manual_rating_var.set(corrections['manual_rating'])
                self.rating_label_var.set(str(corrections['manual_rating']))
            
            # Load quality issues
            if 'quality_issues' in corrections:
                for issue, var in self.quality_issues.items():
                    var.set(corrections['quality_issues'].get(issue, False))
            
            # Load comments
            if 'comments' in corrections:
                self.comments_text.delete(1.0, tk.END)
                self.comments_text.insert(1.0, corrections['comments'])
    
    def _on_canvas_click(self, event):
        """Handle canvas click for region selection."""
        # Convert canvas coordinates to image coordinates
        canvas_x = self.image_canvas.canvasx(event.x)
        canvas_y = self.image_canvas.canvasy(event.y)
        
        # Adjust for zoom
        image_x = int(canvas_x / self.zoom_scale)
        image_y = int(canvas_y / self.zoom_scale)
        
        self.selection_start = (image_x, image_y)
        
        # Start selection rectangle
        if self.selection_rect:
            self.image_canvas.delete(self.selection_rect)
        self.selection_rect = self.image_canvas.create_rectangle(
            canvas_x, canvas_y, canvas_x, canvas_y,
            outline='red', width=2
        )
    
    def _on_canvas_drag(self, event):
        """Handle canvas drag for region selection."""
        if self.selection_start and self.selection_rect:
            canvas_x = self.image_canvas.canvasx(event.x)
            canvas_y = self.image_canvas.canvasy(event.y)
            
            start_x = self.selection_start[0] * self.zoom_scale
            start_y = self.selection_start[1] * self.zoom_scale
            
            self.image_canvas.coords(self.selection_rect, start_x, start_y, canvas_x, canvas_y)
    
    def _on_canvas_release(self, event):
        """Handle canvas release for region selection."""
        if self.selection_start and self.selection_rect:
            canvas_x = self.image_canvas.canvasx(event.x)
            canvas_y = self.image_canvas.canvasy(event.y)
            
            # Convert to image coordinates
            image_x = int(canvas_x / self.zoom_scale)
            image_y = int(canvas_y / self.zoom_scale)
            
            # Create bounding box
            x1 = min(self.selection_start[0], image_x)
            y1 = min(self.selection_start[1], image_y)
            x2 = max(self.selection_start[0], image_x)
            y2 = max(self.selection_start[1], image_y)
            
            if x2 - x1 > 10 and y2 - y1 > 10:  # Minimum size
                # This region could be used for adding missing detections
                messagebox.showinfo("Selection", f"Selected region: ({x1},{y1}) to ({x2},{y2})")
    
    def _zoom_in(self):
        """Zoom in on the image."""
        self.zoom_scale *= 1.2
        self._display_image()
    
    def _zoom_out(self):
        """Zoom out on the image."""
        self.zoom_scale /= 1.2
        self._display_image()
    
    def _reset_zoom(self):
        """Reset zoom to original size."""
        self.zoom_scale = 1.0
        self._display_image()
    
    def _on_detection_select(self, event):
        """Handle detection tree selection."""
        selection = self.detection_tree.selection()
        if selection:
            item = self.detection_tree.item(selection[0])
            detection_id = item['values'][0]
            # You could highlight the corresponding region on the image
    
    def _on_ocr_select(self, event):
        """Handle OCR tree selection."""
        selection = self.ocr_tree.selection()
        if selection:
            item = self.ocr_tree.item(selection[0])
            ocr_id = item['values'][0]
            extracted_text = item['values'][1]
            self.corrected_text_var.set(extracted_text)
    
    def _on_parsing_select(self, event):
        """Handle parsing tree selection."""
        selection = self.parsing_tree.selection()
        if selection:
            item = self.parsing_tree.item(selection[0])
            parsing_id = item['values'][0]
            ingredient = item['values'][1]
            quantity = item['values'][2]
            unit = item['values'][3]
            
            self.corrected_ingredient_var.set(ingredient)
            self.corrected_quantity_var.set(quantity)
            self.corrected_unit_var.set(unit)
    
    def _on_rating_change(self, event):
        """Handle manual rating change."""
        rating = int(self.manual_rating_var.get())
        self.rating_label_var.set(str(rating))
        self._save_current_corrections()
    
    def _mark_detection_correct(self):
        """Mark selected detection as correct."""
        selection = self.detection_tree.selection()
        if selection:
            item = selection[0]
            self.detection_tree.set(item, 'Status', 'Correct')
            self._save_current_corrections()
    
    def _mark_detection_incorrect(self):
        """Mark selected detection as incorrect."""
        selection = self.detection_tree.selection()
        if selection:
            item = selection[0]
            self.detection_tree.set(item, 'Status', 'Incorrect')
            self._save_current_corrections()
    
    def _add_missing_detection(self):
        """Add a missing detection."""
        # This would typically involve drawing a bounding box
        messagebox.showinfo("Add Missing", "Draw a bounding box around the missing text region.")
    
    def _delete_detection(self):
        """Delete selected detection."""
        selection = self.detection_tree.selection()
        if selection:
            if messagebox.askyesno("Delete Detection", "Are you sure you want to delete this detection?"):
                self.detection_tree.delete(selection[0])
                self._save_current_corrections()
    
    def _apply_ocr_correction(self):
        """Apply OCR correction."""
        selection = self.ocr_tree.selection()
        if selection:
            item = selection[0]
            corrected_text = self.corrected_text_var.get()
            self.ocr_tree.set(item, 'Extracted Text', corrected_text)
            self.ocr_tree.set(item, 'Status', 'Corrected')
            self._save_current_corrections()
    
    def _apply_parsing_correction(self):
        """Apply parsing correction."""
        selection = self.parsing_tree.selection()
        if selection:
            item = selection[0]
            corrected_ingredient = self.corrected_ingredient_var.get()
            corrected_quantity = self.corrected_quantity_var.get()
            corrected_unit = self.corrected_unit_var.get()
            
            self.parsing_tree.set(item, 'Ingredient', corrected_ingredient)
            self.parsing_tree.set(item, 'Quantity', corrected_quantity)
            self.parsing_tree.set(item, 'Unit', corrected_unit)
            self.parsing_tree.set(item, 'Status', 'Corrected')
            self._save_current_corrections()
    
    def _save_current_corrections(self):
        """Save corrections for the current image."""
        if not self.current_image_path:
            return
        
        if self.current_image_path not in self.corrections:
            self.corrections[self.current_image_path] = {}
        
        corrections = self.corrections[self.current_image_path]
        
        # Save manual rating
        corrections['manual_rating'] = self.manual_rating_var.get()
        
        # Save quality issues
        corrections['quality_issues'] = {}
        for issue, var in self.quality_issues.items():
            corrections['quality_issues'][issue] = var.get()
        
        # Save comments
        corrections['comments'] = self.comments_text.get(1.0, tk.END).strip()
        
        # Save detection corrections
        corrections['detection_corrections'] = []
        for child in self.detection_tree.get_children():
            item = self.detection_tree.item(child)
            corrections['detection_corrections'].append({
                'id': item['values'][0],
                'status': item['values'][4],
                'confidence': item['values'][1],
                'class': item['values'][2],
                'bbox': item['values'][3]
            })
        
        # Save OCR corrections
        corrections['ocr_corrections'] = []
        for child in self.ocr_tree.get_children():
            item = self.ocr_tree.item(child)
            corrections['ocr_corrections'].append({
                'id': item['values'][0],
                'extracted_text': item['values'][1],
                'confidence': item['values'][2],
                'status': item['values'][3]
            })
        
        # Save parsing corrections
        corrections['parsing_corrections'] = []
        for child in self.parsing_tree.get_children():
            item = self.parsing_tree.item(child)
            corrections['parsing_corrections'].append({
                'id': item['values'][0],
                'ingredient': item['values'][1],
                'quantity': item['values'][2],
                'unit': item['values'][3],
                'confidence': item['values'][4],
                'status': item['values'][5]
            })
        
        # Save timestamp
        corrections['last_modified'] = datetime.now().isoformat()
    
    def _reprocess_image(self):
        """Reprocess the current image."""
        if self.current_image_path:
            self._process_current_image()
            self.status_var.set("Image reprocessed")
    
    def _mark_reviewed(self):
        """Mark current image as reviewed."""
        if self.current_image_path:
            self._save_current_corrections()
            if self.current_image_path not in self.corrections:
                self.corrections[self.current_image_path] = {}
            self.corrections[self.current_image_path]['reviewed'] = True
            self.corrections[self.current_image_path]['review_timestamp'] = datetime.now().isoformat()
            self.status_var.set("Image marked as reviewed")
    
    def _save_comments(self):
        """Save comments for current image."""
        self._save_current_corrections()
        self.status_var.set("Comments saved")
    
    def _clear_comments(self):
        """Clear comments."""
        self.comments_text.delete(1.0, tk.END)
        self._save_current_corrections()
    
    def _load_comment_template(self):
        """Load a comment template."""
        template = """Review Notes:
- Detection Quality: 
- OCR Quality: 
- Parsing Quality: 
- Issues Found:
- Corrections Made:
- Additional Notes:
"""
        self.comments_text.delete(1.0, tk.END)
        self.comments_text.insert(1.0, template)
    
    def _update_display(self):
        """Update all displays based on current data."""
        if self.current_image_path:
            self._load_current_image()
    
    def run(self):
        """Run the manual review interface."""
        self.root.mainloop()


def main():
    """Main function to run the manual review interface."""
    app = ManualReviewInterface()
    app.run()


if __name__ == "__main__":
    main()