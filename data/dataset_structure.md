# Recipe Text Detection Dataset Structure

This document describes the organization and structure of datasets for training text detection models on recipe images. This includes both manually collected recipe data and external text detection datasets for comprehensive training.

## Directory Structure

```
data/
├── recipe_cards/                    # Original recipe images organized by format
│   ├── handwritten/                # Handwritten recipe cards and notes
│   │   ├── recipe_cards/           # Individual recipe cards
│   │   ├── recipe_notebooks/       # Handwritten recipe notebooks
│   │   └── annotations/            # Manual annotations for handwritten recipes
│   ├── printed/                    # Printed cookbooks, magazines, and cards
│   │   ├── cookbooks/              # Cookbook pages
│   │   ├── magazines/              # Magazine recipe pages
│   │   ├── recipe_cards/           # Printed recipe cards
│   │   └── annotations/            # Manual annotations for printed recipes
│   └── digital/                    # Digital recipe screenshots and images
│       ├── websites/               # Recipe website screenshots
│       ├── apps/                   # Mobile app screenshots
│       ├── pdfs/                   # PDF recipe extracts
│       └── annotations/            # Manual annotations for digital recipes
├── external_datasets/              # Downloaded public text detection datasets
│   ├── icdar2015/                  # ICDAR 2015 Incidental Scene Text
│   │   ├── raw/                    # Original ICDAR format
│   │   ├── yolo_format/            # Converted to YOLO format
│   │   └── dataset_info.json      # Dataset metadata
│   ├── icdar2017/                  # ICDAR 2017 Robust Reading
│   │   ├── raw/
│   │   ├── yolo_format/
│   │   └── dataset_info.json
│   ├── textocr/                    # TextOCR dataset
│   │   ├── raw/
│   │   ├── yolo_format/
│   │   └── dataset_info.json
│   ├── coco_text/                  # COCO-Text dataset
│   │   ├── raw/
│   │   ├── yolo_format/
│   │   └── dataset_info.json
│   └── synthtext/                  # SynthText synthetic dataset
│       ├── raw/
│       ├── yolo_format/
│       └── download_instructions.txt
├── processed/                      # Preprocessed and ready-for-training data
│   ├── train/                      # Training split (70%)
│   │   ├── images/                 # Training images
│   │   └── labels/                 # YOLO format training labels
│   ├── val/                        # Validation split (20%)
│   │   ├── images/                 # Validation images
│   │   └── labels/                 # YOLO format validation labels
│   ├── test/                       # Test split (10%)
│   │   ├── images/                 # Test images
│   │   └── labels/                 # YOLO format test labels
│   ├── data.yaml                   # YOLOv8 dataset configuration
│   └── dataset_info.json          # Dataset metadata and statistics
└── annotations/                    # Original annotation files and templates
    ├── templates/                  # Annotation templates for different formats
    ├── manual/                     # Manual annotations in various formats
    └── converted/                  # Converted annotations during processing
```

## Dataset Categories

### 1. Recipe Cards (recipe_cards/)

#### Handwritten (`handwritten/`)
- Personal recipe cards
- Handwritten notebook pages
- Recipe notes and modifications
- Informal ingredient lists

**Characteristics:**
- Variable handwriting styles
- Different pen/pencil types
- Various paper textures
- Informal abbreviations

#### Printed (`printed/`)
- Cookbook pages
- Magazine recipe pages
- Printed recipe cards
- Restaurant menus

**Characteristics:**
- Consistent fonts
- Professional layouts
- High-quality printing
- Structured formatting

#### Digital (`digital/`)
- Website screenshots
- Recipe app interfaces
- Digital cookbook pages
- Social media recipe posts

**Characteristics:**
- Various screen resolutions
- Different UI designs
- Mobile vs desktop layouts
- Web fonts and styling

### 2. External Datasets (external_datasets/)

External text detection datasets are automatically downloaded and converted to YOLO format for training.

#### ICDAR Datasets
- **ICDAR 2015**: Incidental Scene Text Detection
  - Focus: Natural scene text detection
  - Format: Quad coordinates converted to bounding boxes
  - Usage: Robust text detection in varied conditions

- **ICDAR 2017**: Robust Reading Challenge
  - Focus: Multi-lingual text detection
  - Format: Multi-oriented text regions
  - Usage: Language-independent text detection

#### TextOCR Dataset
- **Content**: Large-scale text reading dataset
- **Images**: Natural images with readable text
- **Annotations**: COCO-style converted to YOLO format
- **Usage**: Large-scale training data for text detection

#### COCO-Text Dataset
- **Content**: Text detection and recognition on COCO images
- **Format**: COCO-style annotations
- **Usage**: Natural scene text in context

#### SynthText Dataset
- **Content**: Synthetic text in natural images
- **Size**: ~40GB (requires manual download)
- **Usage**: Data augmentation and pre-training
- **Format**: MATLAB format converted to YOLO

### 3. Dataset Conversion and Integration

All external datasets are automatically processed through:
1. **Download**: Automated download from official sources
2. **Conversion**: Format conversion to YOLO bounding boxes
3. **Classification**: Text regions classified into recipe-relevant categories
4. **Integration**: Merged with recipe-specific data for training

## Annotation Formats

### Text Detection Annotations

#### YOLO Format (for training)
```
# File: image_001.txt
# Format: class_id x_center y_center width height (normalized 0-1)
0 0.5 0.3 0.4 0.05    # ingredient_line
1 0.5 0.6 0.6 0.15    # ingredient_block
```

#### Original Format (for reference)
```json
{
  "image_id": "recipe_001.jpg",
  "image_width": 1024,
  "image_height": 768,
  "annotations": [
    {
      "id": 1,
      "category": "ingredient_line",
      "bbox": [100, 200, 400, 30],
      "text": "2 cups all-purpose flour",
      "confidence": 1.0,
      "recipe_type": "printed"
    },
    {
      "id": 2,
      "category": "ingredient_block",
      "bbox": [100, 250, 500, 120],
      "text": "Ingredients:\n2 cups flour\n1 tsp salt\n3 tbsp oil",
      "confidence": 1.0,
      "recipe_type": "printed"
    }
  ]
}
```

## Class Definitions

### Text Detection Classes

The system uses 5 text detection classes optimized for recipe content:

1. **ingredient_line** (Class ID: 0)
   - Single line ingredient entries with measurements
   - Format: "quantity unit ingredient_name [preparation]"
   - Examples: "2 cups flour", "1 tsp vanilla extract", "3 cloves garlic, minced"
   - Most common class in recipe images

2. **ingredient_block** (Class ID: 1)
   - Multi-line ingredient sections or lists
   - Contains multiple ingredients grouped together
   - May include section headers like "Ingredients:" or "For the sauce:"
   - Examples: Ingredient lists, shopping lists, recipe card sections

3. **instruction_text** (Class ID: 2)
   - Cooking instructions and method steps
   - Method descriptions and cooking techniques
   - Preparation notes and cooking tips
   - Examples: "Preheat oven to 350°F", "Mix dry ingredients"

4. **recipe_title** (Class ID: 3)
   - Recipe names and main titles
   - Dish descriptions and headings
   - Section headers (e.g., "Appetizers", "Main Course")
   - Examples: "Chocolate Chip Cookies", "Mom's Famous Lasagna"

5. **metadata_text** (Class ID: 4)
   - Cooking times and temperatures
   - Serving sizes and yield information
   - Difficulty ratings and skill levels
   - Nutritional information and dietary notes
   - Examples: "Serves 4", "Prep: 15 min", "Gluten-free"

### Class Distribution Guidelines

Target distribution for balanced training:
- **ingredient_line**: 40-50% (most common and important)
- **ingredient_block**: 20-25% (ingredient sections)
- **instruction_text**: 15-20% (cooking directions)
- **recipe_title**: 5-10% (titles and headings)
- **metadata_text**: 5-10% (timing and serving info)

## Data Collection Guidelines

### Image Quality Requirements

#### Minimum Standards
- Resolution: 640x480 or higher
- Format: JPEG, PNG
- Color: RGB (preferred) or Grayscale
- File size: 100KB - 10MB

#### Quality Criteria
- Clear, readable text
- Good contrast between text and background
- Minimal motion blur
- Proper exposure (not too dark/bright)

### Recipe Format Coverage

#### Handwritten Recipes
- **Sources**: Personal collections, recipe exchanges
- **Variations**: Cursive, print, mixed styles
- **Languages**: Primarily English, some multilingual
- **Paper types**: Lined, unlined, index cards

#### Printed Materials
- **Sources**: Cookbooks, magazines, newspapers
- **Layouts**: Single column, multi-column, boxed
- **Fonts**: Various serif and sans-serif
- **Quality**: Professional printing to photocopies

#### Digital Sources
- **Websites**: Recipe blogs, cooking sites
- **Apps**: Mobile recipe applications
- **Social Media**: Instagram, Pinterest posts
- **E-books**: Digital cookbook pages

## Annotation Guidelines

### Text Bounding Boxes

#### Principles
1. **Tight bounding**: Boxes should tightly enclose text
2. **Complete words**: Don't split words across boxes
3. **Logical grouping**: Group related text elements
4. **Consistent labeling**: Use standardized class labels

#### Ingredient Line Detection
```
✓ Good: [2 cups all-purpose flour]
✗ Bad:  [2 cups] [all-purpose flour]
✗ Bad:  [2 cups all-purpose flou][r]
```

#### Multi-line Handling
```
For ingredient blocks spanning multiple lines:
┌─────────────────────────┐
│ Ingredients:            │
│ • 2 cups flour          │
│ • 1 tsp salt            │
│ • 3 tbsp olive oil      │
└─────────────────────────┘
Label as: ingredient_block
```

### Special Cases

#### Fractional Measurements
```
✓ Include: "1 1/2 cups sugar"
✓ Include: "2¼ teaspoons vanilla"
✓ Include: "½ cup butter"
```

#### Abbreviated Units
```
✓ Include: "2 tbsp", "1 tsp", "3 oz"
✓ Include: "2 c.", "1 T.", "3 lbs."
```

#### Ingredient Modifications
```
✓ Include: "2 cups flour, sifted"
✓ Include: "1 onion, diced"
✓ Include: "3 cloves garlic, minced"
```

## Quality Control

### Annotation Validation

#### Automated Checks
- Bounding box coordinates within image bounds
- Non-zero area bounding boxes
- Valid class IDs
- Text consistency with OCR output

#### Manual Review
- 10% random sample review
- Inter-annotator agreement testing
- Difficult case resolution
- Edge case documentation

### Dataset Balance

#### Class Distribution Targets
- ingredient_line: 60-70%
- ingredient_block: 20-25%
- instruction_text: 5-10%
- recipe_title: 2-5%
- metadata_text: 2-5%

#### Recipe Type Balance
- Handwritten: 30%
- Printed: 40%
- Digital: 30%

## Usage Notes

### Training Preparation
1. Convert annotations to YOLO format
2. Apply data augmentation
3. Split into train/val/test sets (70/20/10)
4. Generate class weights for imbalanced data

### Validation Strategy
- Cross-validation by recipe source
- Temporal splits for digital data
- Geographic diversity consideration

### Data Privacy
- Remove personal information from handwritten recipes
- Respect copyright for published materials
- Use public domain or licensed content when possible