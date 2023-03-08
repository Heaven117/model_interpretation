import os
import sys
sys.path.append(os.curdir)

from svm.data_process import loader_data
from svm.train import load_model,train
from IF.IF_svm import calc_main
from utils import get_default_config


if __name__ == "__main__":
    model_config,IF_config = get_default_config()
    train_loader,test_loader= loader_data()

    save_path = model_config['save_path']
    if(os.path.exists(save_path)):
        model = load_model(save_path)
    else:
        model = train(train_loader,test_loader)

    calc_main(IF_config, model,train_loader,test_loader)
   
