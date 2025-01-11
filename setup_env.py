"""Environment setup script for AEPF project."""
import subprocess
import sys
import venv
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_virtual_environment():
    """Create and setup virtual environment."""
    try:
        base_path = Path(__file__).parent
        env_path = base_path / 'env'
        
        # Create virtual environment
        logger.info("Creating virtual environment...")
        venv.create(env_path, with_pip=True)
        
        # Determine pip path
        if sys.platform == 'win32':
            pip_path = env_path / 'Scripts' / 'pip.exe'
        else:
            pip_path = env_path / 'bin' / 'pip'
        
        # Upgrade pip
        subprocess.check_call([str(pip_path), 'install', '--upgrade', 'pip'])
        
        # Install requirements
        requirements_path = base_path / 'requirements.txt'
        logger.info("Installing required packages...")
        subprocess.check_call([
            str(pip_path),
            'install',
            '-r',
            str(requirements_path)
        ])
        
        logger.info("Virtual environment setup complete!")
        
        # Create project directories
        directories = [
            base_path / 'reports',
            base_path / 'reports' / 'AEPF',
            base_path / 'AI_Models' / 'Loan_default' / 'outputs',
            base_path / 'AI_Models' / 'Loan_default' / 'reports'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Create VS Code settings
        vscode_dir = base_path / '.vscode'
        vscode_dir.mkdir(exist_ok=True)
        
        settings = {
            "python.defaultInterpreterPath": str(env_path / "Scripts" / "python.exe"),
            "python.analysis.extraPaths": [str(base_path)],
            "python.linting.enabled": True,
            "python.linting.pylintEnabled": True,
            "python.formatting.provider": "black"
        }
        
        import json
        with open(vscode_dir / 'settings.json', 'w') as f:
            json.dump(settings, f, indent=4)
        
        # Print activation instructions
        if sys.platform == 'win32':
            activate_path = env_path / 'Scripts' / 'activate'
            logger.info("\nTo activate the environment, run:")
            logger.info(f"    {activate_path}.bat")
        else:
            activate_path = env_path / 'bin' / 'activate'
            logger.info("\nTo activate the environment, run:")
            logger.info(f"    source {activate_path}")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error setting up environment: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_virtual_environment() 