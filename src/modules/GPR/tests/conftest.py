"""
Configuration file for pytest.

This file helps pytest understand the package structure and imports.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
