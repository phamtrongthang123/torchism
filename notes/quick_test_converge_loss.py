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

import einops
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.optim import Adam

# make parameter
a = nn.Parameter(torch.randn((7,14,14), requires_grad=True))
b = torch.randn((7,14,14))
optim = Adam([a], lr=0.1)
loss = nn.MSELoss()
for i in range(10000):
    optim.zero_grad()
    l = loss(a, b)
    l.backward()
    optim.step()
    print(l.item())
