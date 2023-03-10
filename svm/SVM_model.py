import torch.nn as nn

class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.layer = nn.Linear(23,1)

    def forward(self, x):
        x = self.layer(x)
        return x
    
    def pred_prod(self,x):
        y_pred = self.forward(x)
        prob = nn.functional.softmax(y_pred)
        top_p, top_class = prob.topk(1)
        return top_p, top_class

class ModelError(Exception):
    pass
