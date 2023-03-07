import os
import numpy as py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optimizer


device =  "mps" if torch.backends.mps.is_available() else "cpu" #change to cpu to reproduce cpu output 

