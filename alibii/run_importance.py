from alibi.explainers import PermutationImportance, PartialDependenceVariance
from sklearn.model_selection import train_test_split

from models.data_process import load_adult_income_dataset
from models.run_MLP import load_model
from utils.helper import adult_process_names, adult_target_value
from utils.parser import *


def getPDV(train_dataset, predict_fn):
    predict_fn = lambda x: model.predict_anchor(x, encoder)
    explainer = PartialDependenceVariance(predictor=predict_fn,
                                          feature_names=adult_process_names,
                                          target_names=adult_target_value, verbose=True)
    exp_importance = explainer.explain(X=train_dataset,
                                       method='importance',
                                       grid_resolution=50)

    feature_importance = exp_importance.feature_importance
    pd_values = exp_importance.pd_values
    return feature_importance, pd_values


def getPI():
    pfi = PermutationImportance(predictor=predict_fn,
                                score_fns=['accuracy', 'f1'],
                                feature_names=adult_process_names)

    exp = pfi.explain(X=test_dataset, y=y_test)
    return exp.feature_importance


args = parse_args()
model = load_model()
# 加载划分数据集
dataset, target, encoder, categorical_names = load_adult_income_dataset()
train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                target,
                                                                test_size=0.2,
                                                                random_state=args.random_state,
                                                                stratify=target)
predict_fn = lambda x: model.predict_anchor(x, encoder)

pfi = PermutationImportance(predictor=predict_fn,
                            score_fns=['accuracy', 'f1'],
                            feature_names=adult_process_names)

exp = pfi.explain(X=test_dataset, y=y_test)
print(exp.feature_importance)
