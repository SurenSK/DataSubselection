import torch
import numpy as np
import time
import random
import itertools
from scipy.sparse.csgraph import shortest_path
import ot
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path


dataID = "humaneval"
i = 0
dataStr = f"data_{dataID}_{i}.pt"
data = torch.load(dataStr, map_location=torch.device('cpu'))
pass