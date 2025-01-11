"""Clean up data directories before regenerating data."""
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_directories():
    """Remove contents of data directories."""
    base_path = Path(__file__).parent.parent
    
    directories = [
        base_path / 'data' / 'raw',
        base_path / 'data' / 'processed'
    ]
    
    for directory in directories:
        if directory.exists():
            logger.info(f"Cleaning directory: {directory}")
            shutil.rmtree(directory)
            directory.mkdir(parents=True)
        else:
            logger.info(f"Creating directory: {directory}")
            directory.mkdir(parents=True)

if __name__ == "__main__":
    cleanup_directories() 