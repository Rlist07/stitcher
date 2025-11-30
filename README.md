# Balloon Panorama Processor

A Python computer vision application for processing unordered aerial images from a tethered balloon and creating 360° panoramas.

## Overview

This application takes unordered aerial images from a tethered balloon and processes them through several stages:
1. Pre-processing (blur detection and resizing)
2. Image stitching (high-level with fallback to low-level SIFT matching)
3. Projection to equirectangular format (2:1 aspect ratio)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py <input_directory> [options]
```

### Options:
- `-o, --output`: Output path for the final panorama (default: data/output/panorama.jpg)
- `-c, --config`: Path to the configuration file (default: config.json)
- `--verbose`: Enable verbose logging

### Example:
```bash
python src/main.py data/input/ -o data/output/my_panorama.jpg --verbose
```

## Configuration

The application uses a JSON configuration file (default: `config.json`) to tune parameters such as:
- Blur detection threshold
- Image resize dimensions
- SIFT feature parameters
- Matching ratio thresholds
- RANSAC parameters

## Project Structure

```
balloon_panorama/
├── src/
│   ├── main.py                 # Main entry point
│   ├── config/
│   │   ├── settings.py         # Configuration handling
│   │   └── ...
│   ├── processing/
│   │   ├── preprocessor.py     # Blur detection and resizing
│   │   ├── stitcher.py         # High-level and fallback stitching
│   │   ├── feature_matcher.py  # Low-level SIFT matching
│   │   ├── projector.py        # Equirectangular projection
│   │   ├── pipeline.py         # Main processing pipeline
│   │   └── utils.py            # Utility functions
│   └── models/
│       └── panorama.py         # Data models
├── data/
│   ├── input/                  # Input images directory
│   ├── output/                 # Output panoramas directory
│   └── temp/                   # Temporary processing files
├── requirements.txt           # Python dependencies
└── config.json               # Configuration file
```

## Processing Pipeline

The application follows this sequence:

1. **Pre-processing**: Filter out blurry images using Variance of Laplacian, resize images if needed
2. **Stitching**: Try high-level OpenCV stitcher first, fallback to SIFT-based matching if needed
3. **Projection**: Convert the stitched panorama to equirectangular format for 360° viewing
4. **Output**: Save the final panorama image in the specified output location

## Dependencies

- OpenCV-Python: For image processing and stitching
- NumPy: For numerical operations
- Pillow: For image validation
- TQDM: For progress tracking
- ConfigParser: For configuration management