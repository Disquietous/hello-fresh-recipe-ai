# Complete Text Extraction Quality Evaluation System

This comprehensive evaluation system provides end-to-end assessment of text extraction quality for recipe images, including text detection, OCR accuracy, ingredient parsing, and active learning capabilities.

## System Overview

The evaluation system consists of several interconnected components:

### 1. Text Detection Accuracy (IoU-based metrics)
- **Precision/Recall/F1-Score** for text region detection
- **Mean IoU** calculation between predicted and ground truth bounding boxes
- **Per-class performance** metrics for different text types (ingredients, amounts, units)
- **Detection rate** and false positive/negative analysis

### 2. OCR Accuracy Measurement
- **Character Error Rate (CER)** - character-level accuracy
- **Word Error Rate (WER)** - word-level accuracy  
- **Levenshtein distance** - edit distance between predicted and ground truth text
- **BLEU scores** for sequence-level accuracy
- **Confidence correlation** analysis

### 3. Ingredient Parsing Accuracy
- **Ingredient name extraction** accuracy
- **Quantity parsing** accuracy (numbers, fractions)
- **Unit recognition** accuracy (cups, tbsp, oz, etc.)
- **Complete ingredient match** rates
- **Structured data validation** and consistency checks

### 4. End-to-End Pipeline Evaluation
- **Overall success rate** of the complete pipeline
- **Stage-wise success rates** (detection → OCR → parsing)
- **Error propagation analysis** between pipeline stages
- **Quality degradation** assessment through the pipeline
- **Processing time** analysis for performance optimization

### 5. Error Analysis and Failure Pattern Detection
- **Common failure patterns** identification
- **Error categorization** (detection, OCR, parsing errors)
- **Image quality impact** analysis
- **Text complexity correlation** with error rates
- **Recovery suggestions** for different error types

### 6. Quality Scoring System
- **Overall quality score** combining all pipeline stages
- **Confidence-weighted scoring** based on model uncertainty
- **Completeness scoring** for recipe extraction
- **Consistency scoring** across similar images
- **Actionability scoring** for extracted recipe data

### 7. Manual Review Interface
- **Interactive GUI** for reviewing extraction results
- **Image display** with zoom and pan capabilities
- **Tabbed interface** for detection, OCR, and parsing results
- **Correction tools** for fixing errors
- **Quality assessment** and annotation capabilities
- **Export functionality** for corrected data

### 8. Active Learning System
- **Uncertainty sampling** to identify problematic cases
- **Diversity-based selection** for annotation efficiency
- **Clustering analysis** for case categorization
- **Improvement potential** assessment
- **Annotation task generation** with priorities
- **Model improvement tracking** over time

## Quick Start

### Basic Usage

Run the complete evaluation system:

```bash
python src/evaluation_orchestrator.py \
    --model models/text_detection_model.pt \
    --test-dataset data/test/ \
    --ground-truth data/ground_truth/ \
    --output evaluation_results/
```

### Individual Components

**1. Comprehensive Evaluation Only:**
```bash
python src/comprehensive_evaluation_system.py \
    --model models/text_detection_model.pt \
    --test-dataset data/test/ \
    --ground-truth data/ground_truth/ \
    --output comprehensive_results/
```

**2. Manual Review Interface:**
```bash
python src/manual_review_interface.py
```

**3. Active Learning Analysis:**
```bash
python src/active_learning_system.py \
    --evaluation-results evaluation_results/detailed_results.json \
    --output active_learning_results/ \
    --generate-tasks
```

## Configuration

### System Configuration

Create a configuration file `config.json`:

```json
{
  "confidence_threshold": 0.25,
  "iou_threshold": 0.5,
  "annotation_budget": 100,
  "uncertainty_threshold": 0.5,
  "parallel_processing": true,
  "max_workers": 4,
  "run_comprehensive_evaluation": true,
  "run_manual_review": false,
  "run_active_learning": true,
  "generate_annotation_tasks": true
}
```

### Ground Truth Format

The system expects ground truth data in the following structure:

```
ground_truth/
├── text_regions.json          # Text detection ground truth
├── ocr_text.json             # OCR ground truth
├── ingredients.json          # Ingredient parsing ground truth
└── quality_scores.json       # Quality assessment ground truth
```

**Text Regions Format:**
```json
{
  "image_001.jpg": {
    "regions": [
      {
        "bbox": [x1, y1, x2, y2],
        "class": "ingredient_name",
        "text": "2 cups flour"
      }
    ]
  }
}
```

**OCR Ground Truth Format:**
```json
{
  "image_001.jpg": {
    "texts": [
      "2 cups flour",
      "1 tsp salt",
      "3 tbsp olive oil"
    ]
  }
}
```

**Ingredients Ground Truth Format:**
```json
{
  "image_001.jpg": [
    {
      "ingredient_name": "flour",
      "quantity": "2",
      "unit": "cups",
      "preparation": null
    },
    {
      "ingredient_name": "salt", 
      "quantity": "1",
      "unit": "tsp",
      "preparation": null
    }
  ]
}
```

## Output Structure

The evaluation system generates comprehensive outputs:

```
evaluation_results/
├── evaluation_workflow_summary.json
├── final_evaluation_report.md
├── comprehensive_evaluation/
│   ├── comprehensive_evaluation_results.json
│   ├── metrics/
│   │   ├── detection_accuracy.json
│   │   ├── ocr_accuracy.json
│   │   ├── parsing_accuracy.json
│   │   └── quality_scoring.json
│   └── visualizations/
│       ├── overall_metrics.png
│       ├── error_analysis.png
│       └── quality_scoring.png
├── active_learning/
│   ├── active_learning_results.json
│   ├── prioritized_cases.json
│   └── visualizations/
│       ├── uncertainty_vs_difficulty.png
│       └── priority_score_distribution.png
└── annotation_tasks/
    ├── annotation_tasks.json
    └── annotation_summary.txt
```

## Key Features

### 1. Comprehensive Metrics
- **Detection Metrics**: Precision, Recall, F1-Score, mAP, IoU analysis
- **OCR Metrics**: Character/Word accuracy, Edit distance, BLEU scores
- **Parsing Metrics**: Ingredient extraction accuracy, structured data quality
- **End-to-End Metrics**: Pipeline success rates, error propagation analysis

### 2. Advanced Analysis
- **Error Pattern Detection**: Identifies common failure modes
- **Quality Scoring**: Multi-dimensional quality assessment
- **Uncertainty Analysis**: Model confidence evaluation
- **Diversity Analysis**: Clustering-based case categorization

### 3. Interactive Tools
- **Manual Review GUI**: User-friendly interface for result inspection
- **Correction Tools**: Built-in editing capabilities
- **Visualization**: Charts and graphs for metric analysis
- **Export Options**: Multiple output formats (JSON, CSV, reports)

### 4. Active Learning
- **Smart Case Selection**: Identifies most valuable cases for annotation
- **Improvement Tracking**: Monitors model performance over time
- **Annotation Efficiency**: Optimizes human annotation effort
- **Model Update Guidance**: Provides improvement recommendations

## Performance Optimization

### Parallel Processing
Enable parallel processing for faster evaluation:

```bash
python src/evaluation_orchestrator.py \
    --config config.json \
    ... other args
```

With `config.json`:
```json
{
  "parallel_processing": true,
  "max_workers": 8
}
```

### Memory Management
For large datasets, consider:
- Processing in batches
- Limiting visualization generation
- Using streaming evaluation for very large datasets

### GPU Acceleration
The system supports GPU acceleration for:
- Text detection models (YOLO)
- OCR processing (when available)
- Clustering analysis

## Integration with Existing Pipeline

The evaluation system integrates seamlessly with the existing ingredient extraction pipeline:

```python
from src.comprehensive_evaluation_system import ComprehensiveEvaluationSystem
from src.ingredient_pipeline import IngredientExtractionPipeline

# Initialize systems
pipeline = IngredientExtractionPipeline()
evaluator = ComprehensiveEvaluationSystem()

# Process and evaluate
results = pipeline.process_image("recipe.jpg")
evaluation = evaluator.evaluate_single_result(results, ground_truth)
```

## Advanced Usage

### Custom Evaluation Metrics

Add custom metrics by extending the evaluation classes:

```python
class CustomEvaluationSystem(ComprehensiveEvaluationSystem):
    def _calculate_custom_metric(self, predictions, ground_truth):
        # Custom evaluation logic
        return custom_score
```

### Batch Processing

Process multiple images efficiently:

```python
evaluator = ComprehensiveEvaluationSystem()
results = evaluator.evaluate_batch(
    image_paths=image_list,
    ground_truth_data=gt_data,
    batch_size=32
)
```

### Model Comparison

Compare different models using the evaluation system:

```python
models = ['model_v1.pt', 'model_v2.pt', 'model_v3.pt']
comparison_results = evaluator.compare_models(
    models=models,
    test_dataset=test_data,
    ground_truth=gt_data
)
```

## Troubleshooting

### Common Issues

**1. Missing Dependencies:**
```bash
pip install -r requirements.txt
```

**2. CUDA Out of Memory:**
- Reduce batch size
- Enable CPU processing
- Use gradient checkpointing

**3. Ground Truth Format Errors:**
- Validate JSON structure
- Check file paths
- Verify coordinate formats

**4. Performance Issues:**
- Enable parallel processing
- Reduce visualization generation
- Use appropriate hardware

### Debug Mode

Enable debug logging for detailed information:

```bash
python src/evaluation_orchestrator.py \
    --config config.json \
    --debug \
    ... other args
```

## Contributing

When adding new evaluation metrics or features:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Provide example usage
5. Include performance benchmarks

## License

This evaluation system is part of the HelloFresh Recipe AI project and follows the same licensing terms.

## Support

For issues, questions, or contributions:
- Create GitHub issues for bugs or feature requests
- Refer to the main project documentation
- Check the examples directory for usage patterns