import os
import numpy as py
import torch
import torch.nn as nn

device =  "mps" if torch.backends.mps.is_available() else "cpu" #change to cpu to reproduce cpu output 

class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.layer = nn.Linear(23,1)

    def forward(self, x):
        x = self.layer(x)
        return x