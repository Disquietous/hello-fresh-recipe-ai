#!/usr/bin/env python3
"""
Active Learning System for Problematic Cases
Identifies challenging cases that would benefit from manual review,
prioritizes them for annotation, and helps improve the model through
targeted training data augmentation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add src to path
sys.path.append(str(Path(__file__).parent))

from comprehensive_evaluation_system import ComprehensiveEvaluationSystem
from manual_review_interface import ManualReviewInterface


@dataclass
class ActiveLearningMetrics:
    """Metrics for active learning performance."""
    total_samples: int
    high_uncertainty_samples: int
    diversity_score: float
    improvement_potential: float
    annotation_efficiency: float
    coverage_score: float


@dataclass
class ProblematicCase:
    """Represents a problematic case identified for active learning."""
    image_path: str
    uncertainty_score: float
    difficulty_score: float
    error_types: List[str]
    confidence_scores: Dict[str, float]
    feature_vector: List[float]
    priority_score: float
    review_needed: bool
    annotation_suggestions: List[str]


@dataclass
class ActiveLearningResult:
    """Result of active learning analysis."""
    metrics: ActiveLearningMetrics
    problematic_cases: List[ProblematicCase]
    cluster_analysis: Dict[str, Any]
    improvement_recommendations: List[str]
    annotation_priorities: List[str]
    model_update_suggestions: List[str]
    timestamp: str


class ActiveLearningSystem:
    """System for identifying problematic cases and improving models through active learning."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the active learning system."""
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize components
        self.evaluation_system = ComprehensiveEvaluationSystem(config)
        
        # Active learning parameters
        self.uncertainty_threshold = self.config.get('uncertainty_threshold', 0.5)
        self.diversity_weight = self.config.get('diversity_weight', 0.3)
        self.difficulty_weight = self.config.get('difficulty_weight', 0.4)
        self.annotation_budget = self.config.get('annotation_budget', 100)
        
        # Clustering parameters
        self.n_clusters = self.config.get('n_clusters', 5)
        self.cluster_model = None
        self.scaler = StandardScaler()
        
        # Model improvement tracking
        self.improvement_history = []
        self.problematic_cases_db = {}
        
        self.logger.info("Active Learning System initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for active learning."""
        logger = logging.getLogger('active_learning_system')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def identify_problematic_cases(self, evaluation_results: List[Dict[str, Any]], 
                                 output_dir: str = "active_learning_results") -> ActiveLearningResult:
        """
        Identify problematic cases for active learning.
        
        Args:
            evaluation_results: Results from comprehensive evaluation
            output_dir: Directory to save results
            
        Returns:
            Active learning result with problematic cases
        """
        self.logger.info("Identifying problematic cases for active learning...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract features and calculate uncertainty scores
        features_data = self._extract_features(evaluation_results)
        
        # Calculate uncertainty scores
        uncertainty_scores = self._calculate_uncertainty_scores(features_data)
        
        # Calculate difficulty scores
        difficulty_scores = self._calculate_difficulty_scores(features_data)
        
        # Identify error patterns
        error_patterns = self._identify_error_patterns(evaluation_results)
        
        # Perform clustering for diversity
        cluster_analysis = self._perform_clustering(features_data)
        
        # Create problematic cases
        problematic_cases = self._create_problematic_cases(
            evaluation_results, features_data, uncertainty_scores, 
            difficulty_scores, error_patterns, cluster_analysis
        )
        
        # Prioritize cases for annotation
        prioritized_cases = self._prioritize_cases(problematic_cases)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(prioritized_cases, cluster_analysis)
        
        # Calculate metrics
        metrics = self._calculate_active_learning_metrics(
            evaluation_results, problematic_cases, cluster_analysis
        )
        
        # Create result
        result = ActiveLearningResult(
            metrics=metrics,
            problematic_cases=prioritized_cases,
            cluster_analysis=cluster_analysis,
            improvement_recommendations=recommendations['improvements'],
            annotation_priorities=recommendations['annotations'],
            model_update_suggestions=recommendations['model_updates'],
            timestamp=datetime.now().isoformat()
        )
        
        # Save results
        self._save_active_learning_results(result, output_path)
        
        # Generate visualizations
        self._generate_active_learning_visualizations(result, output_path)
        
        self.logger.info(f"Identified {len(prioritized_cases)} problematic cases")
        
        return result
    
    def _extract_features(self, evaluation_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract features from evaluation results."""
        features = []
        
        for result in evaluation_results:
            # Basic metrics
            detection_accuracy = result.get('detection_results', {}).get('detection_accuracy', 0.0)
            ocr_accuracy = result.get('ocr_results', {}).get('ocr_accuracy', 0.0)
            parsing_accuracy = result.get('parsing_results', {}).get('parsing_accuracy', 0.0)
            
            # Confidence scores
            detection_confidence = np.mean([
                r.get('confidence', 0.0) for r in result.get('detection_results', {}).get('regions', [])
            ]) if result.get('detection_results', {}).get('regions') else 0.0
            
            ocr_confidence = np.mean([
                r.get('confidence', 0.0) for r in result.get('ocr_results', {}).get('ocr_data', [])
            ]) if result.get('ocr_results', {}).get('ocr_data') else 0.0
            
            parsing_confidence = np.mean([
                r.get('confidence', 0.0) for r in result.get('parsing_results', {}).get('parsed_ingredients', [])
            ]) if result.get('parsing_results', {}).get('parsed_ingredients') else 0.0
            
            # Counts
            num_detections = result.get('detection_results', {}).get('num_detections', 0)
            num_texts = result.get('ocr_results', {}).get('num_texts_extracted', 0)
            num_ingredients = result.get('parsing_results', {}).get('num_ingredients_parsed', 0)
            
            # Error indicators
            has_errors = len(result.get('errors', [])) > 0
            pipeline_success = result.get('pipeline_results', {}).get('pipeline_success', False)
            
            # Image characteristics (if available)
            image_info = result.get('image_info', {})
            image_width = image_info.get('width', 0)
            image_height = image_info.get('height', 0)
            image_aspect_ratio = image_width / image_height if image_height > 0 else 1.0
            
            features.append({
                'filename': result.get('filename', ''),
                'image_path': result.get('image_path', ''),
                'detection_accuracy': detection_accuracy,
                'ocr_accuracy': ocr_accuracy,
                'parsing_accuracy': parsing_accuracy,
                'detection_confidence': detection_confidence,
                'ocr_confidence': ocr_confidence,
                'parsing_confidence': parsing_confidence,
                'num_detections': num_detections,
                'num_texts': num_texts,
                'num_ingredients': num_ingredients,
                'has_errors': has_errors,
                'pipeline_success': pipeline_success,
                'image_width': image_width,
                'image_height': image_height,
                'image_aspect_ratio': image_aspect_ratio,
                'overall_confidence': np.mean([detection_confidence, ocr_confidence, parsing_confidence]),
                'overall_accuracy': np.mean([detection_accuracy, ocr_accuracy, parsing_accuracy])
            })
        
        return pd.DataFrame(features)
    
    def _calculate_uncertainty_scores(self, features_data: pd.DataFrame) -> np.ndarray:
        """Calculate uncertainty scores for each sample."""
        # Use confidence score variance as uncertainty measure
        confidence_cols = ['detection_confidence', 'ocr_confidence', 'parsing_confidence']
        confidence_matrix = features_data[confidence_cols].values
        
        # Calculate uncertainty as inverse of confidence and add variance
        uncertainty_scores = []
        for i in range(len(features_data)):
            confidences = confidence_matrix[i]
            
            # Base uncertainty (inverse of mean confidence)
            mean_confidence = np.mean(confidences)
            base_uncertainty = 1.0 - mean_confidence
            
            # Add variance penalty (high variance = more uncertain)
            confidence_variance = np.var(confidences)
            variance_penalty = confidence_variance * 0.5
            
            # Combine uncertainties
            total_uncertainty = base_uncertainty + variance_penalty
            uncertainty_scores.append(min(total_uncertainty, 1.0))
        
        return np.array(uncertainty_scores)
    
    def _calculate_difficulty_scores(self, features_data: pd.DataFrame) -> np.ndarray:
        """Calculate difficulty scores based on various factors."""
        difficulty_scores = []
        
        for i in range(len(features_data)):
            row = features_data.iloc[i]
            
            # Factors that contribute to difficulty
            difficulty_factors = []
            
            # Low accuracy scores
            accuracy_score = row['overall_accuracy']
            accuracy_difficulty = 1.0 - accuracy_score
            difficulty_factors.append(accuracy_difficulty)
            
            # High number of detections (complex images)
            num_detections = row['num_detections']
            complexity_difficulty = min(num_detections / 20.0, 1.0)  # Normalize to [0,1]
            difficulty_factors.append(complexity_difficulty)
            
            # Image aspect ratio (very wide or tall images are harder)
            aspect_ratio = row['image_aspect_ratio']
            aspect_difficulty = abs(aspect_ratio - 1.0)  # Deviation from square
            aspect_difficulty = min(aspect_difficulty, 1.0)
            difficulty_factors.append(aspect_difficulty)
            
            # Pipeline failure
            pipeline_failure = 1.0 if not row['pipeline_success'] else 0.0
            difficulty_factors.append(pipeline_failure)
            
            # Errors present
            has_errors = 1.0 if row['has_errors'] else 0.0
            difficulty_factors.append(has_errors)
            
            # Calculate overall difficulty
            overall_difficulty = np.mean(difficulty_factors)
            difficulty_scores.append(overall_difficulty)
        
        return np.array(difficulty_scores)
    
    def _identify_error_patterns(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Identify common error patterns."""
        error_patterns = defaultdict(list)
        
        for result in evaluation_results:
            filename = result.get('filename', '')
            
            # Detection errors
            if result.get('detection_results', {}).get('num_detections', 0) == 0:
                error_patterns['no_detections'].append(filename)
            
            # OCR errors
            if result.get('ocr_results', {}).get('num_texts_extracted', 0) == 0:
                error_patterns['no_text_extracted'].append(filename)
            
            # Parsing errors
            if result.get('parsing_results', {}).get('num_ingredients_parsed', 0) == 0:
                error_patterns['no_ingredients_parsed'].append(filename)
            
            # Low confidence
            overall_confidence = np.mean([
                result.get('detection_results', {}).get('detection_accuracy', 0.0),
                result.get('ocr_results', {}).get('ocr_accuracy', 0.0),
                result.get('parsing_results', {}).get('parsing_accuracy', 0.0)
            ])
            
            if overall_confidence < 0.3:
                error_patterns['low_confidence'].append(filename)
            
            # Pipeline failures
            if not result.get('pipeline_results', {}).get('pipeline_success', False):
                error_patterns['pipeline_failure'].append(filename)
            
            # Processing errors
            if result.get('errors'):
                error_patterns['processing_errors'].append(filename)
        
        return dict(error_patterns)
    
    def _perform_clustering(self, features_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering for diversity analysis."""
        # Select features for clustering
        feature_cols = [
            'detection_accuracy', 'ocr_accuracy', 'parsing_accuracy',
            'detection_confidence', 'ocr_confidence', 'parsing_confidence',
            'num_detections', 'num_texts', 'num_ingredients',
            'image_aspect_ratio', 'overall_confidence'
        ]
        
        feature_matrix = features_data[feature_cols].values
        
        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix)
        
        # Scale features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Perform K-means clustering
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.cluster_model.fit_predict(feature_matrix_scaled)
        
        # Analyze clusters
        cluster_analysis = {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': self.cluster_model.cluster_centers_.tolist(),
            'cluster_sizes': [np.sum(cluster_labels == i) for i in range(self.n_clusters)],
            'cluster_characteristics': {}
        }
        
        # Analyze each cluster
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = features_data[cluster_mask]
            
            cluster_analysis['cluster_characteristics'][str(cluster_id)] = {
                'size': np.sum(cluster_mask),
                'mean_accuracy': cluster_data['overall_accuracy'].mean(),
                'mean_confidence': cluster_data['overall_confidence'].mean(),
                'mean_detections': cluster_data['num_detections'].mean(),
                'success_rate': cluster_data['pipeline_success'].mean(),
                'representative_samples': cluster_data['filename'].head(5).tolist()
            }
        
        return cluster_analysis
    
    def _create_problematic_cases(self, evaluation_results: List[Dict[str, Any]], 
                                features_data: pd.DataFrame, uncertainty_scores: np.ndarray,
                                difficulty_scores: np.ndarray, error_patterns: Dict[str, List[str]],
                                cluster_analysis: Dict[str, Any]) -> List[ProblematicCase]:
        """Create problematic cases from analysis results."""
        problematic_cases = []
        
        for i, result in enumerate(evaluation_results):
            filename = result.get('filename', '')
            image_path = result.get('image_path', '')
            
            # Get scores
            uncertainty_score = uncertainty_scores[i]
            difficulty_score = difficulty_scores[i]
            
            # Identify error types
            error_types = []
            for error_type, filenames in error_patterns.items():
                if filename in filenames:
                    error_types.append(error_type)
            
            # Get confidence scores
            confidence_scores = {
                'detection': features_data.iloc[i]['detection_confidence'],
                'ocr': features_data.iloc[i]['ocr_confidence'],
                'parsing': features_data.iloc[i]['parsing_confidence'],
                'overall': features_data.iloc[i]['overall_confidence']
            }
            
            # Create feature vector
            feature_vector = [
                uncertainty_score,
                difficulty_score,
                confidence_scores['overall'],
                features_data.iloc[i]['overall_accuracy'],
                features_data.iloc[i]['num_detections'],
                features_data.iloc[i]['num_texts'],
                features_data.iloc[i]['num_ingredients'],
                len(error_types)
            ]
            
            # Calculate priority score
            priority_score = (
                uncertainty_score * 0.4 +
                difficulty_score * 0.3 +
                (1.0 - confidence_scores['overall']) * 0.2 +
                len(error_types) * 0.1
            )
            
            # Determine if review is needed
            review_needed = (
                uncertainty_score > self.uncertainty_threshold or
                difficulty_score > 0.7 or
                len(error_types) > 2 or
                confidence_scores['overall'] < 0.3
            )
            
            # Generate annotation suggestions
            annotation_suggestions = self._generate_annotation_suggestions(
                result, error_types, confidence_scores
            )
            
            # Create problematic case
            case = ProblematicCase(
                image_path=image_path,
                uncertainty_score=uncertainty_score,
                difficulty_score=difficulty_score,
                error_types=error_types,
                confidence_scores=confidence_scores,
                feature_vector=feature_vector,
                priority_score=priority_score,
                review_needed=review_needed,
                annotation_suggestions=annotation_suggestions
            )
            
            problematic_cases.append(case)
        
        return problematic_cases
    
    def _generate_annotation_suggestions(self, result: Dict[str, Any], 
                                       error_types: List[str], 
                                       confidence_scores: Dict[str, float]) -> List[str]:
        """Generate annotation suggestions for a problematic case."""
        suggestions = []
        
        # Detection suggestions
        if 'no_detections' in error_types or confidence_scores['detection'] < 0.5:
            suggestions.append("Review and correct text region detection")
            suggestions.append("Add missing text bounding boxes")
        
        # OCR suggestions
        if 'no_text_extracted' in error_types or confidence_scores['ocr'] < 0.5:
            suggestions.append("Verify and correct OCR text extraction")
            suggestions.append("Check for text preprocessing issues")
        
        # Parsing suggestions
        if 'no_ingredients_parsed' in error_types or confidence_scores['parsing'] < 0.5:
            suggestions.append("Review ingredient parsing accuracy")
            suggestions.append("Correct ingredient name, quantity, and unit extraction")
        
        # General suggestions
        if 'pipeline_failure' in error_types:
            suggestions.append("Full pipeline review required")
            suggestions.append("Check for preprocessing or configuration issues")
        
        if 'low_confidence' in error_types:
            suggestions.append("Focus on improving confidence scores")
            suggestions.append("Consider additional training data for this case type")
        
        return suggestions
    
    def _prioritize_cases(self, problematic_cases: List[ProblematicCase]) -> List[ProblematicCase]:
        """Prioritize cases for annotation based on various criteria."""
        # Sort by priority score (descending)
        sorted_cases = sorted(problematic_cases, key=lambda x: x.priority_score, reverse=True)
        
        # Apply diversity constraint
        selected_cases = []
        cluster_counts = defaultdict(int)
        
        for case in sorted_cases:
            # Simple diversity check - limit cases per cluster
            # (This is a simplified approach; more sophisticated methods could be used)
            case_cluster = self._get_case_cluster(case)
            
            if cluster_counts[case_cluster] < self.annotation_budget // self.n_clusters:
                selected_cases.append(case)
                cluster_counts[case_cluster] += 1
            
            if len(selected_cases) >= self.annotation_budget:
                break
        
        # Add remaining high-priority cases if budget allows
        for case in sorted_cases:
            if case not in selected_cases and len(selected_cases) < self.annotation_budget:
                selected_cases.append(case)
        
        return selected_cases[:self.annotation_budget]
    
    def _get_case_cluster(self, case: ProblematicCase) -> int:
        """Get the cluster assignment for a case."""
        # This is a simplified approach - in practice, you'd use the trained cluster model
        # For now, we'll use a hash-based approach
        return hash(case.image_path) % self.n_clusters
    
    def _generate_recommendations(self, prioritized_cases: List[ProblematicCase],
                                cluster_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate improvement recommendations."""
        recommendations = {
            'improvements': [],
            'annotations': [],
            'model_updates': []
        }
        
        # Analyze common issues
        error_type_counts = defaultdict(int)
        for case in prioritized_cases:
            for error_type in case.error_types:
                error_type_counts[error_type] += 1
        
        # Generate improvement recommendations
        if error_type_counts['no_detections'] > len(prioritized_cases) * 0.3:
            recommendations['improvements'].append(
                "High rate of detection failures - consider improving text detection model"
            )
        
        if error_type_counts['no_text_extracted'] > len(prioritized_cases) * 0.3:
            recommendations['improvements'].append(
                "High rate of OCR failures - consider improving text preprocessing or OCR engine"
            )
        
        if error_type_counts['no_ingredients_parsed'] > len(prioritized_cases) * 0.3:
            recommendations['improvements'].append(
                "High rate of parsing failures - consider improving ingredient parser"
            )
        
        # Generate annotation recommendations
        high_priority_cases = [case for case in prioritized_cases if case.priority_score > 0.7]
        
        recommendations['annotations'].extend([
            f"Prioritize annotation of {len(high_priority_cases)} high-priority cases",
            "Focus on cases with multiple error types",
            "Ensure diversity in annotation selection across different image types"
        ])
        
        # Generate model update recommendations
        recommendations['model_updates'].extend([
            "Consider data augmentation for underrepresented error patterns",
            "Implement uncertainty-based training data selection",
            "Add hard negative mining for challenging cases"
        ])
        
        return recommendations
    
    def _calculate_active_learning_metrics(self, evaluation_results: List[Dict[str, Any]],
                                         problematic_cases: List[ProblematicCase],
                                         cluster_analysis: Dict[str, Any]) -> ActiveLearningMetrics:
        """Calculate active learning metrics."""
        total_samples = len(evaluation_results)
        high_uncertainty_samples = len([case for case in problematic_cases if case.uncertainty_score > self.uncertainty_threshold])
        
        # Calculate diversity score based on cluster representation
        cluster_representation = np.array(cluster_analysis['cluster_sizes'])
        diversity_score = 1.0 - np.var(cluster_representation) / np.mean(cluster_representation)
        
        # Calculate improvement potential
        low_accuracy_samples = len([case for case in problematic_cases if case.confidence_scores['overall'] < 0.5])
        improvement_potential = low_accuracy_samples / total_samples
        
        # Calculate annotation efficiency (higher is better)
        review_needed_samples = len([case for case in problematic_cases if case.review_needed])
        annotation_efficiency = review_needed_samples / min(len(problematic_cases), self.annotation_budget)
        
        # Calculate coverage score
        coverage_score = len(problematic_cases) / total_samples
        
        return ActiveLearningMetrics(
            total_samples=total_samples,
            high_uncertainty_samples=high_uncertainty_samples,
            diversity_score=diversity_score,
            improvement_potential=improvement_potential,
            annotation_efficiency=annotation_efficiency,
            coverage_score=coverage_score
        )
    
    def _save_active_learning_results(self, result: ActiveLearningResult, output_path: Path):
        """Save active learning results."""
        # Save main results
        result_dict = asdict(result)
        
        results_file = output_path / 'active_learning_results.json'
        with open(results_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        # Save prioritized cases separately
        cases_file = output_path / 'prioritized_cases.json'
        cases_data = [asdict(case) for case in result.problematic_cases]
        with open(cases_file, 'w') as f:
            json.dump(cases_data, f, indent=2, default=str)
        
        # Save cluster model
        if self.cluster_model:
            model_file = output_path / 'cluster_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'cluster_model': self.cluster_model,
                    'scaler': self.scaler
                }, f)
        
        self.logger.info(f"Active learning results saved to: {output_path}")
    
    def _generate_active_learning_visualizations(self, result: ActiveLearningResult, output_path: Path):
        """Generate visualizations for active learning results."""
        viz_dir = output_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Uncertainty vs Difficulty scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        uncertainty_scores = [case.uncertainty_score for case in result.problematic_cases]
        difficulty_scores = [case.difficulty_score for case in result.problematic_cases]
        priority_scores = [case.priority_score for case in result.problematic_cases]
        
        scatter = ax.scatter(uncertainty_scores, difficulty_scores, 
                           c=priority_scores, cmap='viridis', alpha=0.6)
        ax.set_xlabel('Uncertainty Score')
        ax.set_ylabel('Difficulty Score')
        ax.set_title('Uncertainty vs Difficulty (colored by Priority)')
        plt.colorbar(scatter, label='Priority Score')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'uncertainty_vs_difficulty.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Error type distribution
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        error_counts = defaultdict(int)
        for case in result.problematic_cases:
            for error_type in case.error_types:
                error_counts[error_type] += 1
        
        if error_counts:
            error_types = list(error_counts.keys())
            counts = list(error_counts.values())
            
            bars = ax.bar(error_types, counts, color='skyblue')
            ax.set_title('Distribution of Error Types')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'error_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Cluster analysis visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        cluster_sizes = result.cluster_analysis['cluster_sizes']
        cluster_labels = [f'Cluster {i}' for i in range(len(cluster_sizes))]
        
        ax.bar(cluster_labels, cluster_sizes, color='lightcoral')
        ax.set_title('Cluster Size Distribution')
        ax.set_ylabel('Number of Samples')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Priority score distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.hist(priority_scores, bins=20, color='lightgreen', alpha=0.7)
        ax.set_xlabel('Priority Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Priority Score Distribution')
        ax.axvline(np.mean(priority_scores), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(priority_scores):.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'priority_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Active learning visualizations saved to: {viz_dir}")
    
    def generate_annotation_tasks(self, active_learning_result: ActiveLearningResult,
                                output_dir: str = "annotation_tasks") -> Dict[str, Any]:
        """Generate annotation tasks for the prioritized cases."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        annotation_tasks = {
            'metadata': {
                'total_tasks': len(active_learning_result.problematic_cases),
                'generation_timestamp': datetime.now().isoformat(),
                'annotation_budget': self.annotation_budget
            },
            'tasks': []
        }
        
        for i, case in enumerate(active_learning_result.problematic_cases):
            task = {
                'task_id': f"task_{i:04d}",
                'image_path': case.image_path,
                'priority_score': case.priority_score,
                'uncertainty_score': case.uncertainty_score,
                'difficulty_score': case.difficulty_score,
                'error_types': case.error_types,
                'confidence_scores': case.confidence_scores,
                'annotation_suggestions': case.annotation_suggestions,
                'review_needed': case.review_needed,
                'status': 'pending'
            }
            annotation_tasks['tasks'].append(task)
        
        # Save annotation tasks
        tasks_file = output_path / 'annotation_tasks.json'
        with open(tasks_file, 'w') as f:
            json.dump(annotation_tasks, f, indent=2, default=str)
        
        # Generate task summary
        summary_file = output_path / 'annotation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("Active Learning Annotation Tasks Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total tasks: {len(annotation_tasks['tasks'])}\n")
            f.write(f"High priority tasks: {len([t for t in annotation_tasks['tasks'] if t['priority_score'] > 0.7])}\n")
            f.write(f"Review needed tasks: {len([t for t in annotation_tasks['tasks'] if t['review_needed']])}\n")
            f.write(f"Generated: {annotation_tasks['metadata']['generation_timestamp']}\n\n")
            
            f.write("Top 10 Priority Tasks:\n")
            f.write("-" * 30 + "\n")
            
            sorted_tasks = sorted(annotation_tasks['tasks'], key=lambda x: x['priority_score'], reverse=True)
            for i, task in enumerate(sorted_tasks[:10]):
                f.write(f"{i+1}. {Path(task['image_path']).name} (Priority: {task['priority_score']:.3f})\n")
                f.write(f"   Errors: {', '.join(task['error_types'])}\n")
                f.write(f"   Suggestions: {'; '.join(task['annotation_suggestions'][:2])}\n\n")
        
        self.logger.info(f"Generated {len(annotation_tasks['tasks'])} annotation tasks")
        return annotation_tasks
    
    def track_improvement(self, before_results: List[Dict[str, Any]], 
                         after_results: List[Dict[str, Any]], 
                         intervention_type: str = "annotation") -> Dict[str, Any]:
        """Track improvement after active learning intervention."""
        improvement_analysis = {
            'intervention_type': intervention_type,
            'timestamp': datetime.now().isoformat(),
            'before_metrics': self._calculate_summary_metrics(before_results),
            'after_metrics': self._calculate_summary_metrics(after_results),
            'improvements': {},
            'recommendations': []
        }
        
        # Calculate improvements
        before_metrics = improvement_analysis['before_metrics']
        after_metrics = improvement_analysis['after_metrics']
        
        for metric in ['accuracy', 'confidence', 'success_rate']:
            improvement = after_metrics[metric] - before_metrics[metric]
            improvement_analysis['improvements'][metric] = {
                'absolute_improvement': improvement,
                'relative_improvement': improvement / before_metrics[metric] if before_metrics[metric] > 0 else 0,
                'improved': improvement > 0
            }
        
        # Generate recommendations
        if improvement_analysis['improvements']['accuracy']['improved']:
            improvement_analysis['recommendations'].append(
                "Accuracy improved - continue with current active learning strategy"
            )
        else:
            improvement_analysis['recommendations'].append(
                "No accuracy improvement - consider adjusting selection criteria"
            )
        
        # Store in history
        self.improvement_history.append(improvement_analysis)
        
        return improvement_analysis
    
    def _calculate_summary_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate summary metrics from evaluation results."""
        accuracies = []
        confidences = []
        successes = []
        
        for result in results:
            # Calculate overall accuracy
            accuracy = np.mean([
                result.get('detection_results', {}).get('detection_accuracy', 0.0),
                result.get('ocr_results', {}).get('ocr_accuracy', 0.0),
                result.get('parsing_results', {}).get('parsing_accuracy', 0.0)
            ])
            accuracies.append(accuracy)
            
            # Calculate overall confidence
            confidence = np.mean([
                result.get('detection_results', {}).get('detection_accuracy', 0.0),
                result.get('ocr_results', {}).get('ocr_accuracy', 0.0),
                result.get('parsing_results', {}).get('parsing_accuracy', 0.0)
            ])
            confidences.append(confidence)
            
            # Pipeline success
            success = 1.0 if result.get('pipeline_results', {}).get('pipeline_success', False) else 0.0
            successes.append(success)
        
        return {
            'accuracy': np.mean(accuracies),
            'confidence': np.mean(confidences),
            'success_rate': np.mean(successes),
            'total_samples': len(results)
        }


def main():
    """Main function for active learning system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Active Learning System for Recipe Text Extraction')
    parser.add_argument('--evaluation-results', '-e', required=True, 
                       help='Path to evaluation results JSON file')
    parser.add_argument('--output', '-o', default='active_learning_results', 
                       help='Output directory for results')
    parser.add_argument('--config', '-c', help='Configuration file (JSON)')
    parser.add_argument('--annotation-budget', type=int, default=100, 
                       help='Number of samples to select for annotation')
    parser.add_argument('--uncertainty-threshold', type=float, default=0.5, 
                       help='Uncertainty threshold for case selection')
    parser.add_argument('--generate-tasks', action='store_true', 
                       help='Generate annotation tasks')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line arguments
    config.update({
        'annotation_budget': args.annotation_budget,
        'uncertainty_threshold': args.uncertainty_threshold
    })
    
    # Initialize active learning system
    al_system = ActiveLearningSystem(config)
    
    # Load evaluation results
    try:
        with open(args.evaluation_results, 'r') as f:
            evaluation_results = json.load(f)
        
        # Handle different result formats
        if isinstance(evaluation_results, dict) and 'detailed_results' in evaluation_results:
            evaluation_results = evaluation_results['detailed_results']
        
        # Run active learning analysis
        result = al_system.identify_problematic_cases(evaluation_results, args.output)
        
        # Generate annotation tasks if requested
        if args.generate_tasks:
            tasks = al_system.generate_annotation_tasks(result, args.output + "/annotation_tasks")
            print(f"Generated {len(tasks['tasks'])} annotation tasks")
        
        # Print summary
        print(f"\nActive Learning Analysis Results:")
        print(f"=================================")
        print(f"Total samples analyzed: {result.metrics.total_samples}")
        print(f"High uncertainty samples: {result.metrics.high_uncertainty_samples}")
        print(f"Diversity score: {result.metrics.diversity_score:.3f}")
        print(f"Improvement potential: {result.metrics.improvement_potential:.3f}")
        print(f"Annotation efficiency: {result.metrics.annotation_efficiency:.3f}")
        print(f"Coverage score: {result.metrics.coverage_score:.3f}")
        print(f"Problematic cases identified: {len(result.problematic_cases)}")
        print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Active learning analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())