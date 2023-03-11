import torch.nn as nn
import torch

# 定义激活函数，注意：计算损失时，不经过激活层
def sign(x):
    x[x>=0] = 1
    x[x<0] = -1
    return x

class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.layer = nn.Linear(23,1)
        
    def forward(self, x):
        x = self.layer(x)
        return x
    
    def pred(self,x):
        y_pred = self.forward(x)
        prob = sign(y_pred)
        return prob

    def batch_pred(self,samples):
        all_pred = torch.empty(samples.shape[0])
        for i in range(len(samples)):
            y_pred = self.pred(samples[i])
            all_pred[i] = y_pred
        return all_pred
    
    def pred_numpy(self,x):
        x = torch.from_numpy(x)
        y = self.pred(x)
        return y.detach().numpy().squeeze(1)



class ModelError(Exception):
    pass
