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
# for predict 1 value to another value from 0-1, it is also good with BCE with logit loss. 
# Or in case of MSE with output after sigmoid, i always believe that MSE on logit is better:
# loss(sigmoid(logit_pred), target) <<<< loss(logit_pred, target.logit) 
# But keep in mind that in the end, your goal is to use the loss landscape, created by the loss function, to guide the params of the model. 
# Sometimes, the common loss functions work. Sometimes, they don't. It's not always the case that "i think this is the most suitable" is the best answer.
# And sometimes, your loss function won't work because of the random seed .-. Try a few random seed may help. No kidding.
loss = nn.MSELoss() 

for i in range(10000):
    optim.zero_grad()
    l = loss(a, b)
    l.backward()
    optim.step()
    print(l.item())
