# Data Directory

This directory contains all datasets and annotations for the HelloFresh Recipe AI project.

## Structure

- `raw/` - Original, unprocessed images and videos
- `processed/` - Preprocessed images ready for training/inference
- `annotations/` - YOLO format annotation files (.txt) and label definitions

## Data Organization

Place your food images in the following structure:
```
raw/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```