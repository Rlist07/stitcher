"""
Test script to verify the balloon panorama processing pipeline.
This script provides a simple way to test the pipeline components.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import Settings
from src.processing.pipeline import ProcessingPipeline


def create_sample_images():
    """Create sample images for testing the pipeline."""
    # Create sample directory if it doesn't exist
    sample_dir = Path("data/input/test_samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a few simple test images that have some overlapping areas
    # to simulate images that can be stitched together
    
    # Create base image
    base_img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(base_img, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue square
    cv2.rectangle(base_img, (200, 100), (300, 200), (0, 255, 0), -1)  # Green square
    cv2.putText(base_img, 'Image 1', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Create second image with some overlap
    img2 = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img2, (100, 50), (200, 150), (255, 255, 0), -1)  # Cyan square
    cv2.rectangle(img2, (250, 150), (350, 250), (0, 0, 255), -1)  # Red square
    cv2.putText(img2, 'Image 2', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Create third image with some overlap to the second
    img3 = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img3, (150, 100), (250, 200), (128, 0, 128), -1)  # Purple square
    cv2.rectangle(img3, (300, 50), (380, 150), (0, 128, 128), -1)  # Teal square
    cv2.putText(img3, 'Image 3', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save the sample images
    cv2.imwrite(str(sample_dir / "sample1.jpg"), base_img)
    cv2.imwrite(str(sample_dir / "sample2.jpg"), img2)
    cv2.imwrite(str(sample_dir / "sample3.jpg"), img3)
    
    print(f"Created sample images in {sample_dir}")
    return sample_dir


def test_pipeline():
    """Test the complete pipeline with sample images."""
    print("Creating sample images for testing...")
    sample_dir = create_sample_images()
    
    # Load settings
    print("Loading settings...")
    settings = Settings()
    
    # Initialize processing pipeline
    print("Initializing pipeline...")
    pipeline = ProcessingPipeline(settings)
    
    # Define output path
    output_path = "data/output/test_panorama.jpg"
    
    print(f"Processing images from {sample_dir}...")
    print(f"Output will be saved to {output_path}")
    
    # Process the sample images
    success = pipeline.process_images(str(sample_dir), output_path)
    
    if success:
        print("Pipeline test completed successfully!")
        print(f"Panorama saved to: {output_path}")
        return True
    else:
        print("Pipeline test failed!")
        return False


if __name__ == "__main__":
    print("Testing the balloon panorama processing pipeline...")
    success = test_pipeline()
    
    if success:
        print("\nPipeline test PASSED!")
    else:
        print("\nPipeline test FAILED!")
        sys.exit(1)