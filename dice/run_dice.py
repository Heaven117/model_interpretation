import json

import dice_ml
import numpy as np

from models.data_process import load_adult_income_dataset
from models.run_MLP import load_model
from utils.helper import save_json, getFileName
from utils.parser import *

args = parse_args()
device = args.device
# 加载数据集
dataset = load_adult_income_dataset(False)
d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'educational-num', 'hours-per-week'],
                 outcome_name='income')

# 加载模型
model = load_model()
m = dice_ml.Model(model=model, backend='PYT', func="ohe-min-max")
# 实例化 DiCE 类
exp = dice_ml.Dice(d, m, method="gradient")

# get MAD
mads = d.get_mads(normalized=True)

# create feature weights
feature_weights = {}
for feature in mads:
    feature_weights[feature] = round(1 / mads[feature], 2)
print(feature_weights)
features_to_vary = ['age', 'workclass', 'education', 'educational-num', 'marital-status',
                    'occupation', 'relationship', 'hours-per-week']

# 生成反事实解释
dataset.drop('income', axis=1, inplace=True)
query_instance = dataset[1:2]
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=5, desired_class="opposite", proximity_weight=1.5,
                                        diversity_weight=1.0, feature_weights=feature_weights,
                                        features_to_vary=features_to_vary)

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
    dices[str(i)]['cfs_list'] = np.array(serialized_cf_examples['final_cfs_list'])[:, :-1]

save_json(dices, args.out_dir + getFileName('dice', 'json'), overwrite_if_exists=True)
# dice_exp.cf_examples_list[0].final_cfs_df.to_csv(path_or_buf='counterfactuals.csv', index=False)
