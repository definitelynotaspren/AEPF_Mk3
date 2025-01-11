import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Add loan_default package to Python path
package_path = os.path.join(project_root, 'loan_default')
if package_path not in sys.path:
    sys.path.append(package_path) 