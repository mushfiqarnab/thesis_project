from __future__ import annotations
 
import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import List
 
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models import MultimodalThreatModel, count_trainable_params


def find_ffhq_images(ffhq_dir: Path, n: int) -> List[Path]:
    """Walk ffhq_dir recursively and collect first N image files."""
    exts = {".png", ".jpg", ".jpeg"}
    found: List[Path] = []
    for root, _, files in os.walk(str(ffhq_dir)):
        for f in sorted(files):
            if Path(f).suffix.lower() in exts:
                found.append(Path(root) / f)
            if len(found) >= n:
                return found
    return found

