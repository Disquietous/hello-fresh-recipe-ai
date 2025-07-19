#!/usr/bin/env python3
"""
Batch Recipe Processing System
Advanced batch processing for cookbook digitization with parallel processing,
progress tracking, and comprehensive reporting capabilities.
"""

import os
import json
import uuid
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue
import threading
import pickle

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from complete_recipe_analyzer import CompleteRecipeAnalyzer, RecipeAnalysisResult
from recipe_database import RecipeDatabase
from recipe_scaler import RecipeScaler, ScalingOptions
from recipe_image_preprocessor import RecipeImagePreprocessor


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing."""
    max_workers: int = 4
    processing_mode: str = "parallel"  # "parallel", "sequential", "hybrid"
    chunk_size: int = 10
    retry_attempts: int = 3
    timeout_seconds: int = 300
    save_intermediate_results: bool = True
    output_directory: str = "batch_results"
    preprocessing_enabled: bool = True
    database_storage: bool = True
    scaling_enabled: bool = False
    scaling_options: Optional[ScalingOptions] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_workers < 1:
            self.max_workers = 1
        if self.max_workers > mp.cpu_count():
            self.max_workers = mp.cpu_count()


@dataclass
class BatchProcessingJob:
    """Individual batch processing job."""
    job_id: str
    image_path: str
    output_path: str
    status: str = "pending"  # "pending", "processing", "completed", "failed"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_time: float = 0.0
    result: Optional[RecipeAnalysisResult] = None
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class BatchProcessingReport:
    """Batch processing report."""
    batch_id: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    processing_time: float
    success_rate: float
    average_processing_time: float
    jobs: List[BatchProcessingJob]
    config: BatchProcessingConfig
    created_at: str
    completed_at: Optional[str] = None


class BatchProcessor:
    """Advanced batch processing system for recipe digitization."""
    
    def __init__(self, config: Optional[BatchProcessingConfig] = None):
        """
        Initialize batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchProcessingConfig()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.analyzer = CompleteRecipeAnalyzer()
        self.database = RecipeDatabase() if self.config.database_storage else None
        self.scaler = RecipeScaler() if self.config.scaling_enabled else None
        self.preprocessor = RecipeImagePreprocessor() if self.config.preprocessing_enabled else None
        
        # Processing state
        self.current_batch: Optional[BatchProcessingReport] = None
        self.progress_callbacks: List[Callable] = []
        self.stop_processing = False
        
        # Create output directory
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Initialized BatchProcessor with {self.config.max_workers} workers")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for batch processor."""
        logger = logging.getLogger('batch_processor')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def add_progress_callback(self, callback: Callable[[BatchProcessingReport], None]):
        """Add progress callback function."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self):
        """Notify all progress callbacks."""
        if self.current_batch:
            for callback in self.progress_callbacks:
                try:
                    callback(self.current_batch)
                except Exception as e:
                    self.logger.error(f"Progress callback error: {e}")
    
    def process_directory(self, input_directory: str, pattern: str = "*.jpg") -> BatchProcessingReport:
        """
        Process all images in a directory.
        
        Args:
            input_directory: Directory containing recipe images
            pattern: File pattern to match (e.g., "*.jpg", "*.png")
            
        Returns:
            Batch processing report
        """
        input_path = Path(input_directory)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_directory}")
        
        # Find all matching image files
        image_files = list(input_path.glob(pattern))
        if not image_files:
            raise ValueError(f"No images found matching pattern: {pattern}")
        
        # Convert to string paths
        image_paths = [str(path) for path in image_files]
        
        return self.process_images(image_paths)
    
    def process_images(self, image_paths: List[str]) -> BatchProcessingReport:
        """
        Process a list of image paths.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Batch processing report
        """
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create batch jobs
        jobs = []
        for image_path in image_paths:
            job = BatchProcessingJob(
                job_id=str(uuid.uuid4()),
                image_path=image_path,
                output_path=str(self.output_dir / f"{Path(image_path).stem}_analysis.json")
            )
            jobs.append(job)
        
        # Initialize batch report
        self.current_batch = BatchProcessingReport(
            batch_id=batch_id,
            total_jobs=len(jobs),
            completed_jobs=0,
            failed_jobs=0,
            processing_time=0.0,
            success_rate=0.0,
            average_processing_time=0.0,
            jobs=jobs,
            config=self.config,
            created_at=datetime.now().isoformat()
        )
        
        self.logger.info(f"Starting batch processing: {len(jobs)} jobs")
        self._notify_progress()
        
        # Process jobs based on mode
        if self.config.processing_mode == "sequential":
            self._process_sequential(jobs)
        elif self.config.processing_mode == "parallel":
            self._process_parallel(jobs)
        elif self.config.processing_mode == "hybrid":
            self._process_hybrid(jobs)
        else:
            raise ValueError(f"Unknown processing mode: {self.config.processing_mode}")
        
        # Finalize report
        end_time = time.time()
        self.current_batch.processing_time = end_time - start_time
        self.current_batch.success_rate = (
            self.current_batch.completed_jobs / self.current_batch.total_jobs
            if self.current_batch.total_jobs > 0 else 0
        )
        
        # Calculate average processing time
        completed_jobs = [job for job in jobs if job.status == "completed"]
        if completed_jobs:
            self.current_batch.average_processing_time = (
                sum(job.processing_time for job in completed_jobs) / len(completed_jobs)
            )
        
        self.current_batch.completed_at = datetime.now().isoformat()
        
        # Save batch report
        self._save_batch_report(self.current_batch)
        
        self.logger.info(f"Batch processing completed: {self.current_batch.completed_jobs}/{self.current_batch.total_jobs} successful")
        self._notify_progress()
        
        return self.current_batch
    
    def _process_sequential(self, jobs: List[BatchProcessingJob]):
        """Process jobs sequentially."""
        for job in jobs:
            if self.stop_processing:
                break
            
            self._process_single_job(job)
            self._update_batch_progress(job)
    
    def _process_parallel(self, jobs: List[BatchProcessingJob]):
        """Process jobs in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all jobs
            future_to_job = {executor.submit(self._process_single_job, job): job for job in jobs}
            
            # Process completed jobs
            for future in as_completed(future_to_job, timeout=self.config.timeout_seconds):
                if self.stop_processing:
                    break
                
                job = future_to_job[future]
                try:
                    future.result()  # Get result to raise any exceptions
                except Exception as e:
                    self.logger.error(f"Job {job.job_id} failed: {e}")
                    job.status = "failed"
                    job.error = str(e)
                
                self._update_batch_progress(job)
    
    def _process_hybrid(self, jobs: List[BatchProcessingJob]):
        """Process jobs using hybrid approach (parallel preprocessing + sequential analysis)."""
        # First pass: parallel preprocessing
        if self.config.preprocessing_enabled:
            self.logger.info("Starting parallel preprocessing phase...")
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_job = {
                    executor.submit(self._preprocess_image, job): job 
                    for job in jobs
                }
                
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Preprocessing failed for {job.job_id}: {e}")
        
        # Second pass: sequential analysis for better resource management
        self.logger.info("Starting sequential analysis phase...")
        for job in jobs:
            if self.stop_processing:
                break
            
            self._analyze_single_job(job)
            self._update_batch_progress(job)
    
    def _process_single_job(self, job: BatchProcessingJob):
        """Process a single job."""
        job.start_time = datetime.now()
        job.status = "processing"
        
        try:
            start_time = time.time()
            
            # Analyze recipe
            result = self.analyzer.analyze_recipe(job.image_path)
            
            if result.success:
                job.result = result
                job.status = "completed"
                
                # Store in database if enabled
                if self.database:
                    try:
                        recipe_id = self.database.store_recipe_analysis(result)
                        self.logger.info(f"Stored recipe {recipe_id} in database")
                    except Exception as e:
                        self.logger.error(f"Database storage failed: {e}")
                
                # Apply scaling if enabled
                if self.scaler and self.config.scaling_options:
                    try:
                        scaled_recipe = self.scaler.scale_recipe(
                            [ing.parsed_ingredient for ing in result.analyzed_ingredients],
                            result.servings,
                            result.instructions,
                            self.config.scaling_options
                        )
                        # Store scaled recipe
                        if self.database:
                            self.database.store_scaled_recipe(scaled_recipe, recipe_id)
                    except Exception as e:
                        self.logger.error(f"Scaling failed: {e}")
                
                # Save intermediate results if enabled
                if self.config.save_intermediate_results:
                    self._save_job_result(job)
                
            else:
                job.status = "failed"
                job.error = "; ".join(result.errors) if result.errors else "Unknown error"
            
            job.processing_time = time.time() - start_time
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.processing_time = time.time() - start_time if 'start_time' in locals() else 0
        
        job.end_time = datetime.now()
    
    def _preprocess_image(self, job: BatchProcessingJob):
        """Preprocess image for a job."""
        if self.preprocessor:
            try:
                result = self.preprocessor.preprocess_image(job.image_path)
                # Store preprocessed image temporarily
                if result.success:
                    preprocessed_path = self.output_dir / f"{job.job_id}_preprocessed.jpg"
                    self.preprocessor.save_processed_image(result, str(preprocessed_path))
                    job.image_path = str(preprocessed_path)  # Update path to preprocessed image
            except Exception as e:
                self.logger.error(f"Preprocessing failed for {job.job_id}: {e}")
    
    def _analyze_single_job(self, job: BatchProcessingJob):
        """Analyze a single job (used in hybrid mode)."""
        if job.status == "processing":
            return  # Already processed
        
        job.start_time = datetime.now()
        job.status = "processing"
        
        try:
            start_time = time.time()
            result = self.analyzer.analyze_recipe(job.image_path)
            
            if result.success:
                job.result = result
                job.status = "completed"
                
                if self.config.save_intermediate_results:
                    self._save_job_result(job)
            else:
                job.status = "failed"
                job.error = "; ".join(result.errors) if result.errors else "Unknown error"
            
            job.processing_time = time.time() - start_time
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.processing_time = time.time() - start_time if 'start_time' in locals() else 0
        
        job.end_time = datetime.now()
    
    def _update_batch_progress(self, job: BatchProcessingJob):
        """Update batch progress after job completion."""
        if job.status == "completed":
            self.current_batch.completed_jobs += 1
        elif job.status == "failed":
            self.current_batch.failed_jobs += 1
        
        # Calculate current success rate
        total_processed = self.current_batch.completed_jobs + self.current_batch.failed_jobs
        if total_processed > 0:
            self.current_batch.success_rate = self.current_batch.completed_jobs / total_processed
        
        self._notify_progress()
    
    def _save_job_result(self, job: BatchProcessingJob):
        """Save individual job result."""
        if job.result:
            try:
                with open(job.output_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(job.result), f, indent=2, ensure_ascii=False, default=str)
            except Exception as e:
                self.logger.error(f"Failed to save job result: {e}")
    
    def _save_batch_report(self, report: BatchProcessingReport):
        """Save batch processing report."""
        try:
            report_path = self.output_dir / f"batch_report_{report.batch_id}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Batch report saved to: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save batch report: {e}")
    
    def stop_batch_processing(self):
        """Stop current batch processing."""
        self.stop_processing = True
        self.logger.info("Stopping batch processing...")
    
    def resume_batch_processing(self, batch_report_path: str) -> BatchProcessingReport:
        """
        Resume batch processing from a saved report.
        
        Args:
            batch_report_path: Path to saved batch report
            
        Returns:
            Updated batch processing report
        """
        try:
            with open(batch_report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # Reconstruct batch report
            report = BatchProcessingReport(**report_data)
            
            # Find incomplete jobs
            incomplete_jobs = [job for job in report.jobs if job.status in ["pending", "failed"]]
            
            if incomplete_jobs:
                self.logger.info(f"Resuming batch processing: {len(incomplete_jobs)} incomplete jobs")
                self.current_batch = report
                
                # Process incomplete jobs
                if self.config.processing_mode == "sequential":
                    self._process_sequential(incomplete_jobs)
                elif self.config.processing_mode == "parallel":
                    self._process_parallel(incomplete_jobs)
                elif self.config.processing_mode == "hybrid":
                    self._process_hybrid(incomplete_jobs)
                
                # Update report
                self.current_batch.completed_at = datetime.now().isoformat()
                self._save_batch_report(self.current_batch)
                
                return self.current_batch
            else:
                self.logger.info("All jobs already completed")
                return report
                
        except Exception as e:
            self.logger.error(f"Failed to resume batch processing: {e}")
            raise
    
    def generate_batch_summary(self, report: BatchProcessingReport) -> Dict[str, Any]:
        """
        Generate comprehensive batch summary.
        
        Args:
            report: Batch processing report
            
        Returns:
            Batch summary dictionary
        """
        successful_jobs = [job for job in report.jobs if job.status == "completed"]
        failed_jobs = [job for job in report.jobs if job.status == "failed"]
        
        # Performance metrics
        processing_times = [job.processing_time for job in successful_jobs if job.processing_time > 0]
        
        # Ingredient statistics
        total_ingredients = 0
        total_recipes_with_titles = 0
        total_recipes_with_nutrition = 0
        
        for job in successful_jobs:
            if job.result:
                total_ingredients += len(job.result.analyzed_ingredients)
                if job.result.recipe_title:
                    total_recipes_with_titles += 1
                if job.result.nutritional_analysis:
                    total_recipes_with_nutrition += 1
        
        return {
            "batch_info": {
                "batch_id": report.batch_id,
                "total_jobs": report.total_jobs,
                "completed_jobs": report.completed_jobs,
                "failed_jobs": report.failed_jobs,
                "success_rate": report.success_rate,
                "processing_time": report.processing_time,
                "average_processing_time": report.average_processing_time
            },
            "performance_metrics": {
                "fastest_job": min(processing_times) if processing_times else 0,
                "slowest_job": max(processing_times) if processing_times else 0,
                "median_processing_time": sorted(processing_times)[len(processing_times)//2] if processing_times else 0,
                "total_processing_time": sum(processing_times),
                "jobs_per_hour": len(successful_jobs) / (report.processing_time / 3600) if report.processing_time > 0 else 0
            },
            "content_statistics": {
                "total_ingredients_extracted": total_ingredients,
                "average_ingredients_per_recipe": total_ingredients / len(successful_jobs) if successful_jobs else 0,
                "recipes_with_titles": total_recipes_with_titles,
                "recipes_with_nutrition": total_recipes_with_nutrition,
                "title_detection_rate": total_recipes_with_titles / len(successful_jobs) if successful_jobs else 0,
                "nutrition_detection_rate": total_recipes_with_nutrition / len(successful_jobs) if successful_jobs else 0
            },
            "failure_analysis": {
                "failed_jobs": len(failed_jobs),
                "failure_rate": len(failed_jobs) / report.total_jobs if report.total_jobs > 0 else 0,
                "common_errors": self._analyze_common_errors(failed_jobs)
            },
            "configuration": asdict(report.config)
        }
    
    def _analyze_common_errors(self, failed_jobs: List[BatchProcessingJob]) -> Dict[str, int]:
        """Analyze common error patterns."""
        error_counts = {}
        
        for job in failed_jobs:
            if job.error:
                # Categorize errors
                if "timeout" in job.error.lower():
                    error_counts["timeout"] = error_counts.get("timeout", 0) + 1
                elif "memory" in job.error.lower():
                    error_counts["memory"] = error_counts.get("memory", 0) + 1
                elif "file" in job.error.lower():
                    error_counts["file_error"] = error_counts.get("file_error", 0) + 1
                elif "network" in job.error.lower():
                    error_counts["network"] = error_counts.get("network", 0) + 1
                else:
                    error_counts["other"] = error_counts.get("other", 0) + 1
        
        return error_counts


def main():
    """Main batch processing script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch recipe processing')
    parser.add_argument('--input', '-i', required=True, help='Input directory or file list')
    parser.add_argument('--output', '-o', default='batch_results', help='Output directory')
    parser.add_argument('--pattern', default='*.jpg', help='File pattern for directory input')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads/processes')
    parser.add_argument('--mode', choices=['sequential', 'parallel', 'hybrid'], default='parallel', help='Processing mode')
    parser.add_argument('--config', help='Configuration file (JSON)')
    parser.add_argument('--resume', help='Resume from batch report file')
    parser.add_argument('--database', action='store_true', help='Store results in database')
    parser.add_argument('--scaling', action='store_true', help='Enable recipe scaling')
    parser.add_argument('--target-servings', type=int, help='Target servings for scaling')
    
    args = parser.parse_args()
    
    # Load configuration
    config = BatchProcessingConfig()
    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            config = BatchProcessingConfig(**config_data)
    
    # Override with command line arguments
    config.max_workers = args.workers
    config.processing_mode = args.mode
    config.output_directory = args.output
    config.database_storage = args.database
    config.scaling_enabled = args.scaling
    
    if args.scaling and args.target_servings:
        config.scaling_options = ScalingOptions(target_servings=args.target_servings)
    
    # Initialize processor
    processor = BatchProcessor(config)
    
    # Add progress callback
    def progress_callback(report: BatchProcessingReport):
        completed = report.completed_jobs + report.failed_jobs
        percentage = (completed / report.total_jobs) * 100 if report.total_jobs > 0 else 0
        print(f"Progress: {completed}/{report.total_jobs} ({percentage:.1f}%) - Success rate: {report.success_rate:.1%}")
    
    processor.add_progress_callback(progress_callback)
    
    try:
        if args.resume:
            # Resume batch processing
            report = processor.resume_batch_processing(args.resume)
        else:
            # Start new batch processing
            if Path(args.input).is_dir():
                report = processor.process_directory(args.input, args.pattern)
            else:
                # Assume it's a file list
                with open(args.input, 'r') as f:
                    image_paths = [line.strip() for line in f if line.strip()]
                report = processor.process_images(image_paths)
        
        # Generate and display summary
        summary = processor.generate_batch_summary(report)
        
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total jobs: {summary['batch_info']['total_jobs']}")
        print(f"Completed: {summary['batch_info']['completed_jobs']}")
        print(f"Failed: {summary['batch_info']['failed_jobs']}")
        print(f"Success rate: {summary['batch_info']['success_rate']:.1%}")
        print(f"Total processing time: {summary['batch_info']['processing_time']:.1f}s")
        print(f"Average processing time: {summary['batch_info']['average_processing_time']:.1f}s")
        print(f"Jobs per hour: {summary['performance_metrics']['jobs_per_hour']:.1f}")
        print(f"Total ingredients extracted: {summary['content_statistics']['total_ingredients_extracted']}")
        print(f"Average ingredients per recipe: {summary['content_statistics']['average_ingredients_per_recipe']:.1f}")
        
        if summary['failure_analysis']['failed_jobs'] > 0:
            print(f"\nCommon errors: {summary['failure_analysis']['common_errors']}")
        
        print(f"\nResults saved to: {config.output_directory}")
        
        return 0 if report.success_rate > 0.8 else 1
        
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())