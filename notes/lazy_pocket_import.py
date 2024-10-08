import os
import re
import cv2
import sys
import glob
import json
import math
import pickle
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path 
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Pool

import einops
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functorch import vmap # from pytorch 2.3 will be removed, use torch.vmap in that case. For now we still use functorch.
