"""Setup required directories for loan default model."""
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def setup_directories():
    """Create required directories if they don't exist."""
    base_path = Path(__file__).parent.parent
    
    directories = [
        base_path / 'data' / 'raw',
        base_path / 'data' / 'processed',
        base_path / 'outputs',
        base_path / 'reports',
        base_path / 'models'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

if __name__ == "__main__":
    setup_directories() 