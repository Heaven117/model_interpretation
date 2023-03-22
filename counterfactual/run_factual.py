import os
import sys
sys.path.append(os.curdir)

from svm.run_model import load_model
import dice_ml
from utils import ft_names,save_path,target_name
from svm.data_process import load_data
import pandas as pd
import numpy as np

# 加载数据集
X_train,X_test,y_train,y_test = load_data()
train_set = np.hstack((y_train,X_train))
test_set = np.hstack((y_test,X_test))
ft_names.insert(0,target_name)

train_set =  pd.DataFrame(train_set, columns=ft_names)
test_set =  pd.DataFrame(test_set, columns=ft_names)
x_train = train_set.drop(target_name, axis=1)
x_test = test_set.drop(target_name, axis=1)

d = dice_ml.Data(dataframe=train_set, continuous_features=ft_names[1:], outcome_name=target_name)
# print(train_set.head())

# 加载模型
model = load_model(save_path)
m = dice_ml.Model(model=model, backend='PYT',  func="ohe-min-max")
# 实例化 DiCE 类
exp = dice_ml.Dice(d, m, method="gradient")

# get MAD
mads = d.get_mads(normalized=True)

# create feature weights
feature_weights = {}
for feature in mads:
    feature_weights[feature] = round(1/mads[feature], 2)
print(feature_weights)


feature_weights = {'feature Trades 90+': 1, 'feature Trades 60+': 1}
# generate counterfactuals
dice_exp = exp.generate_counterfactuals(x_test[1:3], total_CFs=4, desired_class="opposite" ,
                                        feature_weights = feature_weights)
# highlight only the changes
dice_exp.visualize_as_dataframe(show_only_changes=True)

