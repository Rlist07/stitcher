"""Main entry point for the balloon panorama processor with viewer support."""
import argparse
from pathlib import Path
import logging
import sys

# Add the src directory to the path so we can import our modules
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path.parent))  # Go up to the project directory

from src.config.settings import Settings
from src.processing.pipeline import ProcessingPipeline
from src.viewer.viewer import PanoramaViewer


def main():
    parser = argparse.ArgumentParser(description="Balloon Panorama Processor")
    parser.add_argument(
        "input_dir",
        nargs='?',  # Makes the argument optional
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
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open the web viewer after processing (implies processing if input is provided)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the web viewer (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web viewer (default: 5000)"
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)
    
    # If no input directory is provided but view flag is set, just start the viewer
    if not args.input_dir and args.view:
        logger.info("Starting viewer without processing new images...")
        viewer = PanoramaViewer(host=args.host, port=args.port, output_path=args.output)
        logger.info(f"Viewer started at http://{args.host}:{args.port}")
        viewer.run_with_browser(debug=args.verbose)
        return

    # If no input directory and no view flag, show help
    if not args.input_dir:
        parser.print_help()
        return

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
        
        # If view flag is set, start the viewer
        if args.view:
            logger.info("Starting viewer...")
            viewer = PanoramaViewer(host=args.host, port=args.port, output_path=args.output)
            viewer.update_output_path(args.output)  # Explicitly update the path
            logger.info(f"Viewer started at http://{args.host}:{args.port}")
            viewer.run_with_browser(debug=args.verbose)
    else:
        logger.error("Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()