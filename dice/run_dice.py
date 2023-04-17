import dice_ml
import pandas as pd
from sklearn.model_selection import train_test_split

from models.data_process import load_adult_income_dataset
from models.run_MLP import load_model
from utils.helper import getFileName
from utils.parser import *

args = parse_args()
device = args.device

total_CFs = 5


def findeSingelDice(exp, x_test):
    total = 10
    for idx in range(total):
        query_instance = x_test[idx: idx + 1]
        dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=total_CFs, desired_class="opposite",
                                                proximity_weight=1.5,
                                                diversity_weight=1.0)
        df = dice_exp.cf_examples_list[0].final_cfs_df
        df['from'] = idx
        if (idx == 0):
            dice_exp.cf_examples_list[0].final_cfs_df.to_csv(path_or_buf=args.out_dir + getFileName('dice', 'csv'),
                                                             index=False, mode='a')
        else:
            dice_exp.cf_examples_list[0].final_cfs_df.to_csv(path_or_buf=args.out_dir + getFileName('dice', 'csv'),
                                                             index=False, mode='a', header=0)


def findDice(exp, x_test):
    idx = 2
    query_instance = x_test[0: 10]
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=total_CFs, desired_class="opposite",
                                            proximity_weight=1.5,
                                            diversity_weight=1.0)
    # df = dice_exp.cf_examples_list[0].final_cfs_df
    # df['from'] = idx
    # dice_exp.cf_examples_list[0].final_cfs_df.to_csv(path_or_buf=args.out_dir + getFileName('dice', 'csv'),
    #                                                  index=False)
    # imp = exp.local_feature_importance(query_instance, posthoc_sparsity_param=None)
    # print(imp.local_importance)

    # 生成反事实list
    # dice_exp.visualize_as_dataframe(show_only_changes=True)
    # dices = {}
    # dices['total'] = len(dice_exp.cf_examples_list)
    # if dice_exp.local_importance is not None:
    #     dices['local_importance'] = dice_exp.local_importance
    # if dice_exp.summary_importance is not None:
    #     dices['global_importance'] = dice_exp.summary_importance
    df_list = []
    for i in range(len(dice_exp.cf_examples_list)):
        cf_examples = dice_exp.cf_examples_list[i]
        df = dice_exp.cf_examples_list[0].final_cfs_df
        df['from'] = i
        tmp = df.to_dict(orient='records')
        df_list.extend((tmp))
    ans_df = pd.DataFrame(df_list)
    ans_df.to_csv(path_or_buf=args.out_dir + getFileName('dice', 'csv'), index=False)

    #
    # save_json(dices, args.out_dir + getFileName('dice', 'json'), overwrite_if_exists=True)


def getImportance(exp, x_train):
    cobj = exp.global_feature_importance(x_train[0:10], total_CFs=10, posthoc_sparsity_param=None)
    json_str = cobj.to_json()
    print(json_str)
    print(cobj)


if __name__ == "__main__":
    # 加载划分数据集
    dataset, target = load_adult_income_dataset(False)
    train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                    target,
                                                                    test_size=0.2,
                                                                    random_state=args.random_state,
                                                                    stratify=target)
    x_train = train_dataset.drop('income', axis=1)
    x_test = test_dataset.drop('income', axis=1)

    columns = train_dataset.columns.values
    d = dice_ml.Data(dataframe=train_dataset, continuous_features=['age', 'educational-num', 'hours-per-week'],
                     outcome_name='income')

    # 加载模型
    model = load_model()
    m = dice_ml.Model(model=model, backend='PYT', func="ohe-min-max")
    # exp = dice_ml.Dice(d, m, method="gradient")
    exp = dice_ml.Dice(d, m, method="random")

    # get MAD
    mads = d.get_mads(normalized=True)

    # findDice(exp, x_test)
    findeSingelDice(exp, x_test)
    # getImportance(exp, x_train)
