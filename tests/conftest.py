"""
Pytest configuration file.

This file configures pytest to add the project root to the Python path,
allowing tests to import from the src directory.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
