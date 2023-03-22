import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import csv

import sys
import os
sys.path.append(os.curdir)

from utils.parser import *
from utils.helper import display_progress
from models.data_process import *

args = parse_args()
device = args.device
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class MLP(nn.Module) :
    def __init__(self) :
        super(MLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(25, 12), 
                                nn.ReLU(), 
                                nn.Linear(12, 2)
                                )
    def forward(self, x) :
        out = self.net(x) 
#         print(out)
        return F.softmax(out,dim = 0)
    
    def predict_single(self,x):
        out = self.forward(x)
        _, pred = torch.max(out,dim = 0)
        return pred.item()
    
    def predict_detach(self,x):
        x = torch.from_numpy(x)
        out = self.forward(x)
        _, pred = torch.max(out,dim = 1)
        return pred.numpy()


def train_MLP(train_loader,test_loader):
    # writer = SummaryWriter(log_dir = args.log_dir)
    os.makedirs(args.model_path,exist_ok=True)


    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epoch) :
        train_loss = 0.0
        train_acc = 0.0
        model.train()

        test_id_num = 0
        for x, label in train_loader :
            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(out, 1)
            # print(pred)
            num_correct = (pred == label).sum().item()
            acc = num_correct / x.shape[0]
            train_acc += acc

            display_progress(f"train. MLP model : ", test_id_num, len(train_loader))

            
        print(f'epoch : {epoch + 1}, train_loss : {train_loss / len(train_loader.dataset)}, train_acc : {train_acc / len(train_loader)}')
        # writer.add_scalar('train_loss', train_loss / len(train_loader.dataset), epoch)

        # 验证
        # mse_loss = 1000000
        # with torch.no_grad() :
        #     total_loss = []
        #     model.eval()
        #     for x, label in test_loader :
        #         x, label = x.to(device), label.to(device)
        #         out = model(x)
        #         loss = criterion(out, label)
        #         total_loss.append(loss.item())
            
        #     val_loss = sum(total_loss) / len(total_loss)
        
        # if val_loss < mse_loss :
        #     mse_loss = val_loss 
        #     torch.save(model.state_dict(), args.model_path+f'MPL_{args.epoch}.pth')
    torch.save(model.state_dict(), args.model_path+f'MPL_{args.epoch}.pth')



def test_MLP(test_loader):
    # writer = SummaryWriter(log_dir = args.log_dir)
    os.makedirs(args.predict_path,exist_ok=True)

    best_model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    ckpt = torch.load(args.model_path+f'MPL_{args.epoch}.pth', map_location='cpu')
    best_model.load_state_dict(ckpt)

    test_loss = 0.0
    test_acc = 0.0
    result = []
    best_model.eval()
    for x, label in test_loader :
        x, label = x.to(device), label.to(device)
        out = best_model(x)
        loss = criterion(out, label)

        test_loss+= loss.item()
        _, pred = torch.max(out, 1)
        result.append(pred.detach())
        num_correct = (pred == label).sum().item()
        acc = num_correct / x.shape[0]
        test_acc += acc
    
    print(f'test_loss : {test_loss / len(test_loader.dataset)}, test_acc : {test_acc / len(test_loader)}')
    result = torch.cat(result, dim = 0).cpu().numpy()

    # with open(args.predict_path+f'MPL_{args.epoch}_pred.csv', 'w', newline = '') as file :
    #     writer = csv.writer(file)
    #     writer.writerow(['id', 'label','pred_result'])
    #     for i, pred in enumerate(result) :
    #         writer.writerow([i, test_loader.dataset[i][1].item() ,pred])


def load_model():
    best_model = MLP().to(device)
    ckpt = torch.load(args.model_path+f'MPL_{args.epoch}.pth', map_location='cpu')
    best_model.load_state_dict(ckpt)
    return best_model


if __name__ == "__main__":
    train_dataset = Adult_data(mode = 'train')
    test_dataset = Adult_data(mode = 'test')

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, drop_last = False)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, drop_last = False)
    train_MLP(train_loader,test_loader)
    test_MLP(test_loader)


    
    


