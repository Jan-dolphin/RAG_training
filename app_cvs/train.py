"""CLI for training RAG pipeline"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.training import TrainingPipeline


def setup_logging(verbose: bool = False, log_file: str = None) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=[console_handler]
    )


def main():
    """Main training CLI"""
    parser = argparse.ArgumentParser(
        description="Train RAG pipeline for CV search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default settings
  python train.py

  # Training with verbose output
  python train.py --verbose

  # Training with custom test queries
  python train.py --test-queries "Python developer" "AWS experience" "Java skills"

  # Training with custom configuration
  python train.py --data-dir ./custom_data --parent-size 3000 --child-size 500

  # Save logs to file
  python train.py --log-file training_$(date +%Y%m%d_%H%M%S).log
        """
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        help='Save training logs to file'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        help='Path to CV data directory (overrides config)'
    )

    parser.add_argument(
        '--persist-dir',
        type=str,
        help='Path to vector store persist directory (overrides config)'
    )

    parser.add_argument(
        '--parent-size',
        type=int,
        help='Parent chunk size (overrides config)'
    )

    parser.add_argument(
        '--child-size',
        type=int,
        help='Child chunk size (overrides config)'
    )

    parser.add_argument(
        '--test-queries',
        nargs='+',
        help='Custom test queries for retrieval testing'
    )

    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='Do not save metrics to file'
    )

    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_log_file = f"logs/training_{timestamp}.log"
    log_file = args.log_file or default_log_file

    setup_logging(verbose=args.verbose, log_file=log_file)

    logger = logging.getLogger(__name__)

    # Load configuration
    config = get_config()

    # Override config with CLI arguments
    if args.data_dir:
        config.rag.data_directory = args.data_dir
        logger.info(f"Using custom data directory: {args.data_dir}")

    if args.persist_dir:
        config.rag.persist_directory = args.persist_dir
        logger.info(f"Using custom persist directory: {args.persist_dir}")

    if args.parent_size:
        config.rag.parent_chunk_size = args.parent_size
        logger.info(f"Using custom parent chunk size: {args.parent_size}")

    if args.child_size:
        config.rag.child_chunk_size = args.child_size
        logger.info(f"Using custom child chunk size: {args.child_size}")

    # Create training pipeline
    pipeline = TrainingPipeline(config, log_file=log_file)

    try:
        # Run training
        result = pipeline.run_full_pipeline(
            test_queries=args.test_queries,
            save_metrics=not args.no_metrics
        )

        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Logs saved to: {log_file}")

        if not args.no_metrics:
            logger.info(f"Metrics saved to: training_metrics.json")

        logger.info("\nNext steps:")
        logger.info("  1. Review training logs and metrics")
        logger.info("  2. Adjust chunk sizes if needed (--parent-size, --child-size)")
        logger.info("  3. Test queries with different parameters")
        logger.info("  4. Run the Chainlit app: chainlit run app.py")

        return 0

    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("Training failed!")
        logger.error("=" * 80)
        logger.error(f"Error: {e}", exc_info=args.verbose)
        logger.error(f"\nLogs saved to: {log_file}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
