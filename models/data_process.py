import sys
sys.path.append('./')

import os
import numpy as np
import pandas as pd 
from utils.parser import *
from utils.helper import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder ,normalize,LabelEncoder,StandardScaler
import copy


args = parse_args()
int_col_len = -1

def load_adult_income_dataset():
    # 获取数据集
    datafile=args.data_path + args.dataset + '.data'
    # if(os.path.exists(datafile)):
    adult_data = pd.read_csv(datafile, header = None, skipinitialspace=True,names = adult_column_names)

    adult_data,target = data_process(adult_data)
    adult_data ,one_hot_encoder = data_encode_define(adult_data)

    return adult_data,target,one_hot_encoder

def data_process(df) :
    df.replace("?", pd.NaT, inplace = True)
    df.replace(">50K", 1, inplace = True)
    df.replace("<=50K", 0, inplace = True)
        
    trans = {'workclass' : df['workclass'].mode()[0], 'occupation' : df['occupation'].mode()[0], 'native-country' : df['native-country'].mode()[0]}
    df.fillna(trans, inplace = True)
    df.drop('fnlwgt', axis = 1, inplace = True)
    df.drop('capital-gain', axis = 1, inplace = True)
    df.drop('capital-loss', axis = 1, inplace = True)
    target = df["income"]
    # df.drop('income', axis = 1, inplace = True)

    target = np.array(target)
    # df.to_csv('final_data.csv',index_label='id')

    return  df,target


def data_encode_define(dataset):
    df_object_col = [col for col in dataset.columns if dataset[col].dtype.name == 'object'] # 3个
    df_int_col = [col for col in dataset.columns if dataset[col].dtype.name != 'object'and col != 'income']
    
    # 1.1原始ok编码
    # dataset = pd.concat([dataset[df_int_col], pd.get_dummies(dataset[df_object_col])], axis = 1)

    # 1.2适合Anchor的编码
    dataset = pd.concat([dataset[df_int_col],dataset[df_object_col]], axis = 1).values
    global int_col_len
    int_col_len = len(df_int_col)
    dataset = Labelencoder(dataset,int_col_len)
    dataset = np.array(dataset, dtype = np.float32)

    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(dataset[:,int_col_len:])

    dataset = np.array(dataset, dtype = np.float32)
    return dataset,one_hot_encoder

def Labelencoder(x_data,int_len):
    dataset = copy.deepcopy(x_data[:,0:int_len])
    encoder = LabelEncoder()

    object_col = x_data[:,int_len:]
    rows,cols = object_col.shape
    for c in range(cols):
        tmp = encoder.fit_transform(object_col[:,c].ravel()).reshape(rows,1)
        dataset = np.concatenate((dataset,tmp),axis = 1)
    return dataset

def encoder_process(x_data,encoder):
    dataset = copy.deepcopy(x_data[:,:int_col_len])
    tmp = encoder.transform(x_data[:,int_col_len:]).toarray()
    dataset = np.concatenate((dataset,tmp),axis = 1)

    dataset = np.array(dataset, dtype = np.float32)

    return dataset


class Adult_data(Dataset) :
    def __init__(self,mode,tensor = True) :
        super(Adult_data, self).__init__()
        self.mode = mode
        x_dataset,target,one_hot_encoder = load_adult_income_dataset()
        x_dataset = encoder_process(x_dataset,one_hot_encoder)
        x_dataset = normalize(x_dataset,axis = 0,norm = 'max')

        # 划分数据集
        train_dataset, test_dataset, y_train, y_test = train_test_split(x_dataset,
                                                                        target,
                                                                        test_size=0.2,
                                                                        random_state=0,
                                                                        stratify=target)
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
    # load_adult_income_dataset()
    train_dataset = Adult_data(mode = 'train')
    # test_dataset = Adult_data(mode = 'test')

    # train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, drop_last = False)
    # test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, drop_last = False)


# 废弃
def data_process2(adult_data):
    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data.replace("?", pd.NaT, inplace = True)
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
    target = adult_data["income"]
    target = np.array(target)

    # adult_data.to_csv(args.data_path+ 'final_adult.csv',index_label='id')

    return adult_data,target
