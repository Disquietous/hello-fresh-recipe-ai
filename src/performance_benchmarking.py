#!/usr/bin/env python3
"""
Performance Benchmarking and Optimization Module
Comprehensive performance testing, profiling, and optimization for recipe processing
"""

import asyncio
import time
import statistics
import psutil
import resource
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv
from datetime import datetime, timedelta
import logging
import concurrent.futures
import threading
from contextlib import contextmanager
import cProfile
import pstats
import io
import tracemalloc
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import sys
import os

# Import our API components
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.ingredient_pipeline import IngredientExtractionPipeline
from src.api.recipe_processing_api import RecipeProcessingAPI
from src.api.caching_system import MultiLevelCacheManager
from src.utils.data_utils import validate_dataset_structure

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    operation: str
    duration: float
    memory_peak: float
    memory_current: float
    cpu_percent: float
    throughput: float
    success_rate: float
    error_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'duration': self.duration,
            'memory_peak': self.memory_peak,
            'memory_current': self.memory_current,
            'cpu_percent': self.cpu_percent,
            'throughput': self.throughput,
            'success_rate': self.success_rate,
            'error_count': self.error_count,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    test_images_dir: str = "data/test/images"
    output_dir: str = "benchmarks/results"
    concurrent_requests: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    test_durations: List[int] = field(default_factory=lambda: [30, 60, 120])  # seconds
    image_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(800, 600), (1024, 768), (1920, 1080)])
    ocr_engines: List[str] = field(default_factory=lambda: ["easyocr", "tesseract", "paddleocr"])
    cache_scenarios: List[str] = field(default_factory=lambda: ["no_cache", "with_cache", "cache_hit"])
    memory_profiling: bool = True
    cpu_profiling: bool = True
    generate_reports: bool = True

class PerformanceBenchmark:
    """Main performance benchmarking class"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[PerformanceMetrics] = []
        self.pipeline = IngredientExtractionPipeline()
        self.cache_manager = MultiLevelCacheManager()
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # System monitoring
        self.process = psutil.Process()
        
        logger.info(f"Performance benchmark initialized with config: {self.config}")
    
    @contextmanager
    def performance_monitor(self, operation: str):
        """Context manager for monitoring performance metrics"""
        # Start monitoring
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()
        
        # Memory tracking
        if self.config.memory_profiling:
            tracemalloc.start()
        
        success_count = 0
        error_count = 0
        
        try:
            yield locals()  # Pass metrics tracking to the caller
            success_count = 1
        except Exception as e:
            error_count = 1
            logger.error(f"Error in {operation}: {e}")
            raise
        finally:
            # Calculate metrics
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = current_memory
            
            if self.config.memory_profiling:
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    peak_memory = max(peak_memory, peak / 1024 / 1024)  # MB
                    tracemalloc.stop()
                except:
                    pass
            
            cpu_percent = self.process.cpu_percent()
            throughput = 1 / duration if duration > 0 else 0
            success_rate = success_count / (success_count + error_count) if (success_count + error_count) > 0 else 0
            
            # Record metrics
            metrics = PerformanceMetrics(
                operation=operation,
                duration=duration,
                memory_peak=peak_memory,
                memory_current=current_memory,
                cpu_percent=cpu_percent,
                throughput=throughput,
                success_rate=success_rate,
                error_count=error_count
            )
            
            self.results.append(metrics)
            logger.info(f"Metrics for {operation}: {duration:.3f}s, {peak_memory:.1f}MB, {throughput:.2f} ops/s")
    
    def generate_test_images(self) -> List[str]:
        """Generate test images for benchmarking"""
        test_images = []
        test_dir = Path(self.config.test_images_dir)
        
        if test_dir.exists():
            # Use existing test images
            for img_path in test_dir.glob("*.{jpg,jpeg,png,webp}"):
                test_images.append(str(img_path))
        
        # Generate synthetic test images if needed
        if len(test_images) < 10:
            logger.info("Generating synthetic test images for benchmarking")
            synthetic_dir = Path(self.config.output_dir) / "synthetic_images"
            synthetic_dir.mkdir(exist_ok=True)
            
            for i, (width, height) in enumerate(self.config.image_sizes):
                for j in range(3):  # 3 images per size
                    # Create a simple synthetic recipe image
                    img = Image.new('RGB', (width, height), color='white')
                    img_path = synthetic_dir / f"test_recipe_{width}x{height}_{j}.jpg"
                    img.save(img_path, 'JPEG', quality=85)
                    test_images.append(str(img_path))
        
        return test_images[:50]  # Limit to 50 test images
    
    async def benchmark_single_processing(self, test_images: List[str]) -> None:
        """Benchmark single image processing performance"""
        logger.info("Running single image processing benchmarks")
        
        for ocr_engine in self.config.ocr_engines:
            for image_path in test_images[:10]:  # Test with first 10 images
                operation = f"single_processing_{ocr_engine}"
                
                with self.performance_monitor(operation):
                    try:
                        result = await self.pipeline.process_image_async(
                            image_path,
                            ocr_engine=ocr_engine,
                            enable_caching=False
                        )
                    except Exception as e:
                        logger.warning(f"Failed to process {image_path} with {ocr_engine}: {e}")
    
    async def benchmark_concurrent_processing(self, test_images: List[str]) -> None:
        """Benchmark concurrent processing performance"""
        logger.info("Running concurrent processing benchmarks")
        
        for concurrency in self.config.concurrent_requests:
            operation = f"concurrent_processing_{concurrency}"
            
            with self.performance_monitor(operation):
                tasks = []
                for i in range(concurrency):
                    image_path = test_images[i % len(test_images)]
                    task = self.pipeline.process_image_async(
                        image_path,
                        enable_caching=False
                    )
                    tasks.append(task)
                
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    successful = sum(1 for r in results if not isinstance(r, Exception))
                    logger.info(f"Concurrent {concurrency}: {successful}/{len(tasks)} successful")
                except Exception as e:
                    logger.error(f"Concurrent processing failed: {e}")
    
    async def benchmark_cache_performance(self, test_images: List[str]) -> None:
        """Benchmark caching system performance"""
        logger.info("Running cache performance benchmarks")
        
        # Test cache miss (no cache)
        operation = "cache_miss"
        with self.performance_monitor(operation):
            result = await self.pipeline.process_image_async(
                test_images[0],
                enable_caching=False
            )
        
        # Test cache write
        operation = "cache_write"
        with self.performance_monitor(operation):
            result = await self.pipeline.process_image_async(
                test_images[0],
                enable_caching=True
            )
        
        # Test cache hit
        operation = "cache_hit"
        with self.performance_monitor(operation):
            result = await self.pipeline.process_image_async(
                test_images[0],
                enable_caching=True
            )
    
    async def benchmark_batch_processing(self, test_images: List[str]) -> None:
        """Benchmark batch processing performance"""
        logger.info("Running batch processing benchmarks")
        
        batch_sizes = [5, 10, 20]
        
        for batch_size in batch_sizes:
            operation = f"batch_processing_{batch_size}"
            batch_images = test_images[:batch_size]
            
            with self.performance_monitor(operation):
                try:
                    results = await self.pipeline.process_batch_async(
                        batch_images,
                        max_concurrent=min(4, batch_size)
                    )
                    logger.info(f"Batch {batch_size}: {len(results)} processed")
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
    
    def benchmark_memory_usage(self, test_images: List[str]) -> None:
        """Detailed memory usage analysis"""
        logger.info("Running memory usage analysis")
        
        # Memory usage over time
        memory_samples = []
        start_time = time.time()
        
        def monitor_memory():
            while time.time() - start_time < 60:  # Monitor for 1 minute
                memory_samples.append({
                    'timestamp': time.time() - start_time,
                    'memory_mb': self.process.memory_info().rss / 1024 / 1024,
                    'memory_percent': self.process.memory_percent()
                })
                time.sleep(1)
        
        # Start memory monitoring in background
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()
        
        # Process images while monitoring
        for i, image_path in enumerate(test_images[:5]):
            operation = f"memory_test_{i}"
            with self.performance_monitor(operation):
                try:
                    result = self.pipeline.process_image(image_path)
                    if i % 2 == 0:
                        gc.collect()  # Force garbage collection
                except Exception as e:
                    logger.warning(f"Memory test failed for {image_path}: {e}")
        
        monitor_thread.join()
        
        # Save memory usage data
        memory_file = Path(self.config.output_dir) / "memory_usage.json"
        with open(memory_file, 'w') as f:
            json.dump(memory_samples, f, indent=2)
    
    def benchmark_cpu_profiling(self, test_images: List[str]) -> None:
        """CPU profiling and hotspot analysis"""
        logger.info("Running CPU profiling analysis")
        
        profile_file = Path(self.config.output_dir) / "cpu_profile.prof"
        
        # Profile the processing
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            for image_path in test_images[:3]:
                result = self.pipeline.process_image(image_path)
        finally:
            profiler.disable()
        
        # Save profile data
        profiler.dump_stats(str(profile_file))
        
        # Generate text report
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        profile_report = Path(self.config.output_dir) / "cpu_profile_report.txt"
        with open(profile_report, 'w') as f:
            f.write(s.getvalue())
    
    def load_test(self, test_images: List[str], duration_seconds: int = 60) -> None:
        """Load testing with sustained traffic"""
        logger.info(f"Running load test for {duration_seconds} seconds")
        
        start_time = time.time()
        request_count = 0
        error_count = 0
        response_times = []
        
        with self.performance_monitor(f"load_test_{duration_seconds}s"):
            while time.time() - start_time < duration_seconds:
                try:
                    request_start = time.time()
                    image_path = test_images[request_count % len(test_images)]
                    result = self.pipeline.process_image(image_path, enable_caching=True)
                    response_time = time.time() - request_start
                    response_times.append(response_time)
                    request_count += 1
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Load test request failed: {e}")
                
                # Small delay to simulate realistic load
                time.sleep(0.1)
        
        # Calculate load test metrics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            
            load_metrics = {
                'duration': duration_seconds,
                'total_requests': request_count,
                'error_count': error_count,
                'success_rate': (request_count - error_count) / request_count,
                'requests_per_second': request_count / duration_seconds,
                'avg_response_time': avg_response_time,
                'p95_response_time': p95_response_time,
                'p99_response_time': p99_response_time
            }
            
            load_file = Path(self.config.output_dir) / f"load_test_{duration_seconds}s.json"
            with open(load_file, 'w') as f:
                json.dump(load_metrics, f, indent=2)
            
            logger.info(f"Load test completed: {request_count} requests, {avg_response_time:.3f}s avg response")
    
    def generate_performance_report(self) -> None:
        """Generate comprehensive performance report"""
        logger.info("Generating performance report")
        
        if not self.results:
            logger.warning("No performance results to report")
            return
        
        # Aggregate results by operation
        operations = {}
        for result in self.results:
            op = result.operation
            if op not in operations:
                operations[op] = []
            operations[op].append(result)
        
        # Generate summary statistics
        summary = {}
        for op, results in operations.items():
            durations = [r.duration for r in results]
            memory_peaks = [r.memory_peak for r in results]
            throughputs = [r.throughput for r in results]
            
            summary[op] = {
                'count': len(results),
                'avg_duration': statistics.mean(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'avg_memory_peak': statistics.mean(memory_peaks),
                'max_memory_peak': max(memory_peaks),
                'avg_throughput': statistics.mean(throughputs),
                'success_rate': statistics.mean([r.success_rate for r in results])
            }
        
        # Save summary report
        report_file = Path(self.config.output_dir) / "performance_summary.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_file = Path(self.config.output_dir) / "performance_detailed.csv"
        with open(detailed_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
        
        # Generate visualizations
        self.generate_performance_visualizations(summary)
        
        logger.info(f"Performance report saved to {self.config.output_dir}")
    
    def generate_performance_visualizations(self, summary: Dict) -> None:
        """Generate performance visualization charts"""
        try:
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Recipe Processing Performance Analysis', fontsize=16)
            
            operations = list(summary.keys())
            
            # Duration comparison
            durations = [summary[op]['avg_duration'] for op in operations]
            axes[0, 0].bar(operations, durations)
            axes[0, 0].set_title('Average Processing Duration')
            axes[0, 0].set_ylabel('Seconds')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Memory usage comparison
            memory = [summary[op]['avg_memory_peak'] for op in operations]
            axes[0, 1].bar(operations, memory)
            axes[0, 1].set_title('Average Peak Memory Usage')
            axes[0, 1].set_ylabel('MB')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Throughput comparison
            throughput = [summary[op]['avg_throughput'] for op in operations]
            axes[1, 0].bar(operations, throughput)
            axes[1, 0].set_title('Average Throughput')
            axes[1, 0].set_ylabel('Operations/Second')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Success rate comparison
            success_rates = [summary[op]['success_rate'] * 100 for op in operations]
            axes[1, 1].bar(operations, success_rates)
            axes[1, 1].set_title('Success Rate')
            axes[1, 1].set_ylabel('Percentage')
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save the plot
            plot_file = Path(self.config.output_dir) / "performance_charts.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance charts saved to {plot_file}")
            
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")
    
    async def run_full_benchmark_suite(self) -> None:
        """Run the complete benchmark suite"""
        logger.info("Starting full performance benchmark suite")
        start_time = time.time()
        
        try:
            # Generate test images
            test_images = self.generate_test_images()
            logger.info(f"Using {len(test_images)} test images")
            
            # Run all benchmark tests
            await self.benchmark_single_processing(test_images)
            await self.benchmark_concurrent_processing(test_images)
            await self.benchmark_cache_performance(test_images)
            await self.benchmark_batch_processing(test_images)
            
            # CPU and memory profiling (synchronous)
            if self.config.cpu_profiling:
                self.benchmark_cpu_profiling(test_images)
            
            if self.config.memory_profiling:
                self.benchmark_memory_usage(test_images)
            
            # Load testing
            for duration in self.config.test_durations:
                self.load_test(test_images, duration)
            
            # Generate reports
            if self.config.generate_reports:
                self.generate_performance_report()
            
            total_time = time.time() - start_time
            logger.info(f"Full benchmark suite completed in {total_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Benchmark suite failed: {e}", exc_info=True)
            raise

class PerformanceOptimizer:
    """Performance optimization recommendations and automated tuning"""
    
    def __init__(self, benchmark_results_dir: str):
        self.results_dir = Path(benchmark_results_dir)
        self.recommendations = []
    
    def analyze_results(self) -> List[Dict[str, Any]]:
        """Analyze benchmark results and generate optimization recommendations"""
        recommendations = []
        
        # Load performance summary
        summary_file = self.results_dir / "performance_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Analyze duration patterns
            for operation, metrics in summary.items():
                if metrics['avg_duration'] > 10.0:  # Slow operations
                    recommendations.append({
                        'type': 'performance',
                        'severity': 'high',
                        'operation': operation,
                        'issue': f"High processing time: {metrics['avg_duration']:.2f}s",
                        'recommendation': "Consider optimizing image preprocessing or using faster OCR engines"
                    })
                
                if metrics['avg_memory_peak'] > 1000:  # High memory usage
                    recommendations.append({
                        'type': 'memory',
                        'severity': 'medium',
                        'operation': operation,
                        'issue': f"High memory usage: {metrics['avg_memory_peak']:.1f}MB",
                        'recommendation': "Implement memory optimization or process images in smaller batches"
                    })
                
                if metrics['success_rate'] < 0.95:  # Low success rate
                    recommendations.append({
                        'type': 'reliability',
                        'severity': 'high',
                        'operation': operation,
                        'issue': f"Low success rate: {metrics['success_rate']*100:.1f}%",
                        'recommendation': "Improve error handling and input validation"
                    })
        
        # Save recommendations
        recommendations_file = self.results_dir / "optimization_recommendations.json"
        with open(recommendations_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        return recommendations
    
    def generate_optimization_report(self) -> str:
        """Generate human-readable optimization report"""
        recommendations = self.analyze_results()
        
        report_lines = [
            "# Performance Optimization Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            f"Total recommendations: {len(recommendations)}",
            f"High severity: {sum(1 for r in recommendations if r['severity'] == 'high')}",
            f"Medium severity: {sum(1 for r in recommendations if r['severity'] == 'medium')}",
            "",
            "## Detailed Recommendations",
            ""
        ]
        
        for i, rec in enumerate(recommendations, 1):
            report_lines.extend([
                f"### {i}. {rec['operation']} - {rec['type'].title()} Issue",
                f"**Severity:** {rec['severity'].title()}",
                f"**Issue:** {rec['issue']}",
                f"**Recommendation:** {rec['recommendation']}",
                ""
            ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.results_dir / "optimization_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        return report_content

# CLI interface for running benchmarks
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recipe Processing Performance Benchmarking")
    parser.add_argument("--output-dir", default="benchmarks/results", help="Output directory for results")
    parser.add_argument("--test-images", default="data/test/images", help="Test images directory")
    parser.add_argument("--duration", type=int, nargs="+", default=[30, 60], help="Load test durations (seconds)")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8], help="Concurrent request counts")
    parser.add_argument("--skip-profiling", action="store_true", help="Skip CPU/memory profiling")
    parser.add_argument("--optimize", action="store_true", help="Generate optimization recommendations")
    
    args = parser.parse_args()
    
    # Configure benchmark
    config = BenchmarkConfig(
        test_images_dir=args.test_images,
        output_dir=args.output_dir,
        concurrent_requests=args.concurrency,
        test_durations=args.duration,
        cpu_profiling=not args.skip_profiling,
        memory_profiling=not args.skip_profiling
    )
    
    # Run benchmark
    benchmark = PerformanceBenchmark(config)
    
    try:
        asyncio.run(benchmark.run_full_benchmark_suite())
        
        if args.optimize:
            optimizer = PerformanceOptimizer(args.output_dir)
            report = optimizer.generate_optimization_report()
            print("\n" + "="*50)
            print("OPTIMIZATION RECOMMENDATIONS")
            print("="*50)
            print(report)
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)