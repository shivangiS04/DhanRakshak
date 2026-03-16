#!/usr/bin/env python
"""Wrapper to run pipeline_v2 from project root"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src_enhanced.pipeline_v2 import main

if __name__ == '__main__':
    main()
