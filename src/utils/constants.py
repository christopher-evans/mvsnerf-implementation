"""
Define your constants here.

Example:
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, ".experiments")
"""

# mean and standard deviation per-channel for DTU dataset
DTU_RGB_MEAN = [0.485, 0.456, 0.406]
DTU_RGB_STD = [0.229, 0.224, 0.225]
