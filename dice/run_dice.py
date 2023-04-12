import json

import dice_ml
import numpy as np
from sklearn.model_selection import train_test_split

from models.data_process import load_adult_income_dataset
from models.run_MLP import load_model
from utils.parser import *

args = parse_args()
device = args.device
# 加载划分数据集
dataset, target = load_adult_income_dataset(False)
train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                target,
                                                                test_size=0.2,
                                                                random_state=args.random_state,
                                                                stratify=target)
x_train = train_dataset.drop('income', axis=1)
x_test = test_dataset.drop('income', axis=1)

d = dice_ml.Data(dataframe=train_dataset, continuous_features=['age', 'educational-num', 'hours-per-week'],
                 outcome_name='income')

# 加载模型
model = load_model()
m = dice_ml.Model(model=model, backend='PYT', func="ohe-min-max")
exp = dice_ml.Dice(d, m, method="gradient")

# get MAD
mads = d.get_mads(normalized=True)

# 生成反事实解释
dataset.drop('income', axis=1, inplace=True)
query_instance = x_test[2:3]
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=5, desired_class="opposite", proximity_weight=1.5,
                                        diversity_weight=1.0)

# imp = exp.local_feature_importance(query_instance, posthoc_sparsity_param=None)
# print(imp.local_importance)


# 生成反事实list
dice_exp.visualize_as_dataframe(show_only_changes=True)
dices = {}
dices['total'] = len(dice_exp.cf_examples_list)
if dice_exp.local_importance is not None:
    dices['local_importance'] = dice_exp.local_importance
if dice_exp.summary_importance is not None:
    dices['global_importance'] = dice_exp.summary_importance

for i in range(len(dice_exp.cf_examples_list)):
    cf_examples = dice_exp.cf_examples_list[i]
    cfs_list = []
    cf_examples_str = cf_examples.to_json(serialization_version='2.0')
    serialized_cf_examples = json.loads(cf_examples_str)
    dices[str(i)] = {}
    dices[str(i)]['cfs_list'] = np.array(serialized_cf_examples['final_cfs_list'])

# save_json(dices, args.out_dir + getFileName('dice', 'json'), overwrite_if_exists=True)
dice_exp.cf_examples_list[0].final_cfs_df.to_csv(path_or_buf='counterfactuals.csv', index=False)
