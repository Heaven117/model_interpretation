import torch.nn as nn

class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.layer = nn.Linear(23,1)

    def forward(self, x):
        x = self.layer(x)
        return x