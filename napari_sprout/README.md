# napari-SPROUT

A napari plugin for SPROUT (Semi-automated Parcellation of Region Outputs Using Thresholding and Transformation).

## Description

This plugin provides an interactive interface for the SPROUT segmentation workflow within napari, including:
- Interactive seed generation with real-time threshold preview
- Seed selection and editing
- Growth visualization
- Batch processing capabilities

## Installation

```bash
pip install -e .
```

## Usage

1. Open napari
2. Go to Plugins → SPROUT to open the widget
3. Load your image and follow the three-step workflow:
   - Generate seeds
   - Select/edit seeds
   - Grow regions

## Features

- **Interactive Visualization**: Real-time preview of thresholding and morphological operations
- **Flexible Workflow**: Use existing SPROUT functionality through an intuitive GUI
- **Non-invasive**: Uses existing SPROUT code without modifications
- **Batch Processing**: Process multiple images with saved parameters

## Requirements

- napari >= 0.4.16
- SPROUT (must be installed separately)
- Python >= 3.8
