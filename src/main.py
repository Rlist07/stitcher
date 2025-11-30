"""Main entry point for the balloon panorama processor."""
import argparse
from pathlib import Path
import logging
import sys

# Add the src directory to the path so we can import our modules
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path.parent))  # Go up to the src directory

from src.config.settings import Settings
from src.processing.pipeline import ProcessingPipeline


def main():
    parser = argparse.ArgumentParser(description="Balloon Panorama Processor")
    parser.add_argument(
        "input_dir", 
        help="Directory containing input images to process"
    )
    parser.add_argument(
        "-o", "--output", 
        default="data/output/panorama.jpg",
        help="Output path for the final panorama (default: data/output/panorama.jpg)"
    )
    parser.add_argument(
        "-c", "--config", 
        default="config.json",
        help="Path to the configuration file (default: config.json)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting balloon panorama processing...")
    
    # Load settings
    settings = Settings(args.config)
    logger.info(f"Loaded settings from {args.config}")
    
    # Initialize processing pipeline
    pipeline = ProcessingPipeline(settings)
    
    # Process images
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    success = pipeline.process_images(str(input_path), args.output)
    
    if success:
        logger.info("Processing completed successfully!")
        print(f"Panorama saved to: {args.output}")
    else:
        logger.error("Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()