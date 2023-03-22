import sys
sys.path.append('./')

import os
import numpy as np
import pandas as pd 
from utils.parser import *
from utils.helper import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing  


args = parse_args()

# 预处理adult数据集
def load_adult_income_dataset():
    # 获取数据集
    datafile=args.data_path + args.dataset + '.data'
    if(os.path.exists(datafile)):
        raw_data = np.genfromtxt(datafile,delimiter=', ', dtype=str, invalid_raise=False)
    else:
        raw_data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                            delimiter=', ', dtype=str, invalid_raise=False)
    adult_data = pd.DataFrame(raw_data, columns=adult_column_names)
    
    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data.drop('fnlwgt', axis = 1, inplace = True)
    adult_data.drop('capital-gain', axis = 1, inplace = True)
    adult_data.drop('capital-loss', axis = 1, inplace = True)
    adult_data.drop('education', axis = 1, inplace = True)
    adult_data.drop('relationship', axis = 1, inplace = True)
    adult_data.drop('native-country', axis = 1, inplace = True)
    adult_data = adult_data.astype({"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64})
    adult_data = adult_data.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government',
                                        'Local-gov': 'Government'}})
    adult_data = adult_data.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})
    adult_data = adult_data.replace(
            {
                'occupation': {
                    'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                    'Exec-managerial': 'White-Collar', 'Farming-fishing': 'Blue-Collar',
                    'Handlers-cleaners': 'Blue-Collar',
                    'Machine-op-inspct': 'Blue-Collar', 'Other-service': 'Service',
                    'Priv-house-serv': 'Service',
                    'Prof-specialty': 'Professional', 'Protective-serv': 'Service',
                    'Tech-support': 'Service',
                    'Transport-moving': 'Blue-Collar', 'Unknown': 'Other/Unknown',
                    'Armed-Forces': 'Other/Unknown', '?': 'Other/Unknown'
                }
            }
        )
    adult_data = adult_data.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
                                                            'Married-spouse-absent': 'Married', 'Never-married': 'Single'}})
    adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

    return adult_data

def one_hot(adult_data):
    # 数据集处理
    df_object_col = [col for col in adult_data.columns if adult_data[col].dtype.name == 'object']
    df_int_col = [col for col in adult_data.columns if adult_data[col].dtype.name != 'object' and col != 'income']
    target = adult_data["income"]
    dataset = pd.concat([adult_data[df_int_col], pd.get_dummies(adult_data[df_object_col])], axis = 1)
    
    return dataset,target

class Adult_data(Dataset) :
    def __init__(self,mode,tensor = True) :
        super(Adult_data, self).__init__()
        self.mode = mode
        adult_data = load_adult_income_dataset()
        dataset,target= one_hot(adult_data)
         # 划分数据集
        train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                        target,
                                                                        test_size=0.2,
                                                                        random_state=0,
                                                                        stratify=target)
        
        # 进行独热编码对齐
        # test_dataset = fix_columns(test_dataset, train_dataset.columns)
        # 归一化
        train_dataset = train_dataset.apply(lambda x : (x - x.mean()) / x.std())
        test_dataset = test_dataset.apply(lambda x : (x - x.mean()) / x.std())
        y_train, y_test = np.array(y_train), np.array(y_test)
        train_dataset, test_dataset = np.array(train_dataset, dtype = np.float32), np.array(test_dataset, dtype = np.float32)

        if tensor:
            if mode == 'train' : 
                self.target = torch.tensor(y_train)
                self.dataset = torch.FloatTensor(train_dataset)
            else :
                self.target = torch.tensor(y_test)
                self.dataset = torch.FloatTensor(test_dataset)
        else:
            if mode == 'train' : 
                self.target = y_train
                self.dataset = train_dataset
            else :
                self.target = y_test
                self.dataset = test_dataset

        print(self.dataset.shape, self.target.dtype)   
        
    def __getitem__(self, item) :
        return self.dataset[item], self.target[item]

    def __len__(self) :
        return len(self.dataset)
    

if __name__ == "__main__":
    train_dataset = Adult_data(mode = 'train')
    test_dataset = Adult_data(mode = 'test')

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, drop_last = False)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, drop_last = False)
