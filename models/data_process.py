import copy

import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, normalize
from torch.utils.data import Dataset

from utils.helper import *

args = parse_args()
int_col_len = -1


def load_adult_income_dataset(encode=True, baseDir=None):
    if baseDir is not None:
        datafile = baseDir + args.data_path + 'final_data.csv'
    else:
        datafile = args.data_path + 'final_data.csv'
    adult_data = pd.read_csv(datafile, header=0)
    adult_data.drop('id', axis=1, inplace=True)
    target = adult_data['income']
    target = np.array(target)
    df_object_col = [col for col in adult_data.columns if adult_data[col].dtype.name == 'object']  # 3个
    df_int_col = [col for col in adult_data.columns if adult_data[col].dtype.name != 'object']
    adult_data = pd.concat([adult_data[df_int_col], adult_data[df_object_col]], axis=1)
    # print(adult_data.columns)

    if encode:
        adult_data.drop('income', axis=1, inplace=True)
        global int_col_len
        int_col_len = len(df_int_col) - 1
        adult_data, one_hot_encoder, categorical_names = data_encode_define(adult_data)
        return adult_data, target, one_hot_encoder, categorical_names
    else:
        return adult_data, target


def data_encode_define(dataset):
    categorical_names = {}
    for feature in discrete_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(dataset[feature])
        dataset[feature] = le.transform(dataset[feature])
        categorical_names[feature] = le.classes_

    # for feature in range(int_col_len, dataset.shape[1]):
    #     le = sklearn.preprocessing.LabelEncoder()
    #     le.fit(dataset[:, feature])
    #     dataset[:, feature] = le.transform(dataset[:, feature])
    #     categorical_names[feature] = le.classes_
    dataset = np.array(dataset, dtype=np.float32)
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(dataset[:, int_col_len:])
    return dataset, one_hot_encoder, categorical_names


def encoder_process(x_data, encoder):
    dataset = copy.deepcopy(x_data[:, :int_col_len])
    tmp = encoder.transform(x_data[:, int_col_len:]).toarray()
    dataset = np.concatenate((dataset, tmp), axis=1)
    dataset = np.array(dataset, dtype=np.float32)
    return dataset


def data_process():
    # 获取数据集
    datafile = args.data_path + args.dataset + '.data'
    df = pd.read_csv(datafile, header=None, skipinitialspace=True, names=adult_column_names)
    df.replace("?", pd.NaT, inplace=True)
    df.replace(">50K", 1, inplace=True)
    df.replace("<=50K", 0, inplace=True)

    trans = {'workclass': df['workclass'].mode()[0], 'occupation': df['occupation'].mode()[0],
             'native-country': df['native-country'].mode()[0]}
    df.fillna(trans, inplace=True)
    df.drop('fnlwgt', axis=1, inplace=True)
    df.drop('capital-gain', axis=1, inplace=True)
    df.drop('capital-loss', axis=1, inplace=True)
    target = df["income"]
    # df.drop('income', axis = 1, inplace = True)

    target = np.array(target)
    print(df.columns.values)
    df.to_csv(args.data_path + 'final_data.csv', index_label='id')

    return df, target


class Adult_data(Dataset):
    def __init__(self, mode: object, tensor: object = True, encode: object = True) -> object:
        super(Adult_data, self).__init__()
        self.mode = mode
        x_dataset, target, one_hot_encoder, _ = load_adult_income_dataset()
        self.encoder = one_hot_encoder
        if encode:
            x_dataset = encoder_process(x_dataset, one_hot_encoder)
            x_dataset = normalize(x_dataset, axis=0, norm='max')

        # 划分数据集
        train_dataset, test_dataset, y_train, y_test = train_test_split(x_dataset,
                                                                        target,
                                                                        test_size=0.2,
                                                                        random_state=args.random_state,
                                                                        stratify=target)

        if tensor:
            if mode == 'train':
                self.target = torch.tensor(y_train)
                self.dataset = torch.FloatTensor(train_dataset)
            elif mode == 'test':
                self.target = torch.tensor(y_test)
                self.dataset = torch.FloatTensor(test_dataset)
            else:
                self.target = torch.tensor(target)
                self.dataset = torch.FloatTensor(x_dataset)
        else:
            if mode == 'train':
                self.target = y_train
                self.dataset = train_dataset
            else:
                self.target = y_test
                self.dataset = test_dataset

        print(self.dataset.shape, self.target.dtype)

    def __getitem__(self, item):
        return self.dataset[item], self.target[item]

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    x_dataset, target = load_adult_income_dataset(False)
    # 划分数据集
    train_dataset, test_dataset, y_train, y_test = train_test_split(x_dataset,
                                                                    target,
                                                                    test_size=0.2,
                                                                    random_state=args.random_state,
                                                                    stratify=target)
    # df = pd.DataFrame(test_dataset)
    # df.to_csv(args.out_dir + getFileName('test', 'csv'), index='id', float_format='%.4f',
    #                 index_label='id')
