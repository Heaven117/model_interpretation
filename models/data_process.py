import sys
sys.path.append('./')

import os
import numpy as np
import pandas as pd 
from utils.parser import *
from utils.helper import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder ,normalize,LabelEncoder
import copy


args = parse_args()

# 预处理adult数据集
def load_adult_income_dataset():
    # 获取数据集
    datafile=args.data_path + args.dataset + '.data'
    raw_data = np.genfromtxt(datafile,delimiter=', ', dtype=str, invalid_raise=False)
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
    # target = adult_data["income"]
    # adult_data.drop('income', axis = 1, inplace = True)

    # adult_data.to_csv(args.data_path+ 'final_adult.csv',index_label='id')
    

    # df_object_col = [col for col in adult_data.columns if adult_data[col].dtype.name == 'object'] # 3个
    # df_int_col = [col for col in adult_data.columns if adult_data[col].dtype.name != 'object']

    # adult_data[df_object_col] = adult_data[df_object_col].apply(lambda x : lencode(x))
    
    # adult_data = pd.concat([adult_data[df_int_col], adult_data[df_object_col]], axis = 1)

    # adult_data = Labelencoder(adult_data)
    # adult_data = np.array(adult_data, dtype = np.float32)

    # encoder = OneHotEncoder()
    # encoder.fit(adult_data[:,3:])

    return adult_data

def Labelencoder(x_data):
    dataset = copy.deepcopy(x_data[:,0:3])
    encoder = LabelEncoder()

    object_col = x_data[:,3:]
    rows,cols = object_col.shape
    for c in range(cols):
        tmp = encoder.fit_transform(object_col[:,c].ravel()).reshape(rows,1)
        dataset = np.concatenate((dataset,tmp),axis = 1)
    return dataset


def encodeData(x_data,encoder):
    dataset = copy.deepcopy(x_data[:,:3])
    tmp = encoder.transform(x_data[:,3:]).toarray()
    dataset = np.concatenate((dataset,tmp),axis = 1)

    dataset = np.array(dataset, dtype = np.float32)

    return dataset

def encoder(x_data):
    dataset = copy.deepcopy(x_data[:,0:3])
    encoder = OneHotEncoder()

    # one-hot处理
    object_col = x_data[:,3:]
    rows,cols = object_col.shape
    for c in range(cols):
        tmp = encoder.fit_transform(object_col[:,c].reshape(rows,1)).toarray() 
        dataset = np.concatenate((dataset,tmp),axis = 1)

    # df_object_col = [col for col in x_data.columns if x_data[col].dtype.name == 'object']
    # df_int_col = [col for col in x_data.columns if x_data[col].dtype.name != 'object']
    # dataset = pd.concat([x_data[df_int_col], pd.get_dummies(x_data[df_object_col])], axis = 1)
    # tmp = encoder.fit_transform(x_data[:,3].reshape(rows,1)).toarray() 
    # new_data = np.concatenate((new_data,tmp),axis = 1)

    # 归一化
    # dataset = dataset.apply(lambda x : (x - x.mean()) / x.std())
    dataset = normalize(dataset,axis = 0,norm = 'max')
    dataset = np.array(dataset, dtype = np.float32)
    return dataset


class Adult_data(Dataset) :
    def __init__(self,mode,tensor = True) :
        super(Adult_data, self).__init__()
        self.mode = mode
        x_dataset,target,encoder = load_adult_income_dataset()
        target = np.array(target)

        x_dataset = encodeData(x_dataset,encoder)


         # 划分数据集
        train_dataset, test_dataset, y_train, y_test = train_test_split(x_dataset,
                                                                        target,
                                                                        test_size=0.2,
                                                                        random_state=0,
                                                                        stratify=target)
        
        # train_dataset= encoder(train_dataset)
        # test_dataset= encoder(test_dataset)

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



def load_adult_income_dataset2(only_train=True):
    """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares
       the data for data analysis based on https://rpubs.com/H_Zhu/235617

    :return adult_data: returns preprocessed adult income dataset.
    """
    datafile=args.data_path + args.dataset + '.data'
    raw_data = np.genfromtxt(datafile,delimiter=', ', dtype=str, invalid_raise=False)

    #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']

    adult_data = pd.DataFrame(raw_data, columns=column_names)

    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data = adult_data.astype({"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64})

    adult_data = adult_data.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government',
                                     'Local-gov': 'Government'}})
    adult_data = adult_data.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    # adult_data = adult_data.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
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

    adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                              'Amer-Indian-Eskimo': 'Other'}})

    adult_data = adult_data[['age', 'workclass', 'education', 'marital-status', 'occupation',
                             'race', 'gender', 'hours-per-week', 'income']]

    adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

    adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                                   '11th': 'School', '10th': 'School', '7th-8th': 'School',
                                                   '9th': 'School', '12th': 'School', '5th-6th': 'School',
                                                   '1st-4th': 'School', 'Preschool': 'School'}})

    adult_data = adult_data.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

    if only_train:
        train, _ = train_test_split(adult_data, test_size=0.2, random_state=17)
        adult_data = train.reset_index(drop=True)

    # Remove the downloaded dataset
    # if os.path.isdir('archive.ics.uci.edu'):
    #     entire_path = os.path.abspath('archive.ics.uci.edu')
    #     shutil.rmtree(entire_path)

    return adult_data

