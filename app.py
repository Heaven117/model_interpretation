import warnings

import pandas as pd
from alibi.explainers import PartialDependenceVariance
from flask import Flask
from flask import request
from sklearn.model_selection import train_test_split

from models.data_process import load_adult_income_dataset
from models.run_MLP import load_model, test_model
from utils.helper import *
from utils.parser import parse_args

args = parse_args()
warnings.filterwarnings("ignore")  # 忽略UserWarning兼容性警告

np.random.seed(12345)

# ------- Initialize file ------- #
data_file = args.data_path + 'final_data.csv'
pre_file = args.out_dir + getFileName('prediction', 'csv')
dice_file = args.out_dir + getFileName('dice', 'csv')
train_data = json.loads(pd.read_csv(args.data_path + 'train.csv', header=0).to_json(orient='records'))
pre_data = json.loads(pd.read_csv(pre_file, header=0).to_json(orient='records'))
pre_data_len = len(pre_data)
diceData = pd.read_csv(dice_file, header=0)
with open(args.out_dir + getFileName('influence', 'json'), 'r', encoding='utf-8') as fp:
    influenceData = json.load(fp)
    print('=====influence file read done')
fp.close()
with open(args.out_dir + getFileName('ianchors_beam', 'json'), 'r', encoding='utf-8') as fp:
    anchorData = json.load(fp)
    print('=====anchor file read done')
fp.close()
# with open(args.out_dir + getFileName('dice', 'json'), 'r', encoding='utf-8') as fp:
#     diceData = json.load(fp)
#     print('=====dice file read done')
# fp.close()

# ------- Initialize Model ------- #
dataset, target, encoder, categorical_names = load_adult_income_dataset()
train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                target,
                                                                test_size=0.2,
                                                                random_state=args.random_state,
                                                                stratify=target)
model = load_model()
loss, acc, FN, TN, FP, TP = test_model()

predict_fn = lambda x: model.predict_anchor(x, encoder)
explainer = PartialDependenceVariance(predictor=predict_fn,
                                      feature_names=adult_process_names,
                                      target_names=adult_target_value, verbose=True)
exp_importance = explainer.explain(X=train_dataset,
                                   method='importance',
                                   grid_resolution=50)
feature_importance = exp_importance.feature_importance
pd_values = exp_importance.pd_values


# ------ Help Function ------- #
# 获取训练集中的sample
def getSampleCovered(covered):
    def getAnchorSample(sam):
        for feature in range(3, dataset.shape[1]):
            sam[feature] = categorical_names[feature][sam[feature]]
        df = pd.DataFrame(sam, columns=adult_process_names)
        return df.to_json()

    new_ct = []
    for ct_item in covered:
        df = pd.DataFrame(ct_item, columns=adult_process_names)
        for feature in discrete_features:
            df[feature] = df[feature].map(lambda k: categorical_names[feature][int(k)])
        new_ct.append(df.to_dict(orient='record'))
    return np.array(new_ct)


# ------ Initialize WebApp ------- #
app = Flask(__name__)


@app.route('/api/getGlobalData')
def getGlobalData():
    ans = {'feature_importance': feature_importance, 'pd_values': pd_values}
    return toJson(ans)


@app.route('/api/getPredData')
def getPredData():
    idx = -1
    idx = request.args.get('idx')

    response = {'total': len(pre_data)}

    if idx is not None:
        idx = int(idx)
        response['data'] = pre_data[idx]
    else:
        response['data'] = pre_data

    return toJson(response)


@app.route('/api/getInstance')
def getInstance():
    idx = -10
    try:
        idx = int(request.args.get('params'))
        if idx < 0 or idx > pre_data_len:
            return f"Please enter a sample number in the range (1, ${pre_data_len})."
    except:
        return f"Please enter a sample number in the range (1, ${pre_data_len})."

    sample = pre_data[idx]
    response = {'id': idx, 'total': pre_data_len, 'sample': sample}
    return toJson(response)


@app.route('/api/getTrainSample')
def getTrainSample():
    idx = -10
    train_data_len = len(train_data)
    try:
        idx = int(request.args.get('params'))
        if idx < 0 or idx > train_data_len:
            return f"Please enter a sample number in the range (1, ${train_data_len})."
    except:
        return f"Please enter a sample number in the range (1, ${train_data_len})."

    sample = train_data[idx]
    response = {'id': idx, 'sample': sample}
    return toJson(response)


@app.route('/api/getAnchor')
def getAnchor():
    try:
        idx = request.args.get('params')
    except:
        return f"Please enter a sample number ."

    anchor = anchorData[idx]

    response = {}
    response['feature'] = anchor['feature']
    response['precision'] = anchor['precision']
    response['coverage'] = anchor['coverage']
    response['covered_false'] = getSampleCovered(anchor['covered_false'])
    response['covered_true'] = getSampleCovered(anchor['covered_true'])
    return toJson(response)


# 获取相似数据-影响训练点
@app.route('/api/getInfluenceData')
def getInfluenceData():
    try:
        idx = request.args.get('params')
    except:
        return f"Please enter a sample number."
    total = influenceData['total']
    inData = influenceData[str(idx)]
    time = inData['time_calc_influence_s']
    influence = inData['influence']
    harmful = inData['harmful']
    helpful = inData['helpful']
    harmPer = inData['harmful_len'] / (inData['helpful_len'] + inData['harmful_len'])

    response = {'total': total, 'time': time, 'influence': [], 'harmful': [], 'helpful': [], 'max': inData['max'],
                'min': inData['min'], 'harmPer': harmPer}
    for i in range(len(harmful)):
        response['harmful'].append({
            'id': harmful[i],
            'value': influence[harmful[i]],
        })
        response['helpful'].append({
            'id': helpful[i],
            'value': influence[helpful[i]],
        })
    for i in range(len(influence)):
        response['influence'].append({
            'id': i,
            'value': influence[i],
            'sign': (1 if influence[i] else 0)
        })

    return toJson(response)


@app.route('/api/getHelpfulData')
def getHelpfulData():
    try:
        idx = request.args.get('params')
    except:
        return f"Please enter a sample number."
    inData = influenceData[str(idx)]
    influence = inData['influence']
    helpful = inData['helpful']
    harmful = inData['harmful']

    response = {'harmful': [], 'helpful': []}
    for i in range(len(helpful)):
        response['helpful'].append({
            'id': helpful[i],
            'value': influence[helpful[i]],
            'data': pre_data[helpful[i]]
        })
    response['helpful'].sort(key=lambda x: x['value'], reverse=True)
    response['helpful'] = response['helpful'][:10]
    response['helpful'].insert(0, {
        'id': idx,
        'value': 1,
        'data': pre_data[int(idx)]
    })

    for i in range(len(harmful)):
        response['harmful'].append({
            'id': harmful[i],
            'value': influence[harmful[i]],
            'data': pre_data[harmful[i]]
        })
    response['harmful'].sort(key=lambda x: x['value'])
    response['harmful'] = response['harmful'][:10]
    response['harmful'].insert(0, {
        'id': idx,
        'value': 1,
        'data': pre_data[int(idx)]
    })
    response['helpful'].reverse()
    response['harmful'].reverse()
    return toJson(response)


@app.route('/api/getDiceData')
def getDiceData():
    try:
        idx = int(request.args.get('params'))
    except:
        return f"Please enter a sample number."

    # inData = diceData[str(idx)]
    # cfs_list = inData['cfs_list']
    # pf = pd.DataFrame(cfs_list, columns=adult_process_names)
    # cfs_list = pf.to_dict(orient='records')
    # response = {'cfs_list': cfs_list}
    # return toJson(response)

    ans = []
    sample = pre_data[idx]
    list = diceData[diceData['from'] == idx]
    list['prediction'] = list['income']
    list = list.to_dict(orient='records')
    ans.append(sample)
    ans.extend(list)
    return toJson(ans)


@app.route('/api/getModeForm')
def getModeForm():
    return toJson(categorical_names)


@app.route('/api/getModelInfo')
def getModelInfo():
    ans = {'loss': loss, 'acc': acc, 'FN': FN, 'TN': TN, 'FP': FP, 'TP': TP, 'categorical_names': categorical_names}
    return toJson(ans)


@app.route('/api/runModel', methods=['GET', 'POST'])
def runModel():
    params = request.json
    val = [[float(params[i]) for i in params]]
    val = np.array(val)
    out = model.predict_anchor(val, encoder)

    ans = {'prediction': out[0], 'percentage': out}
    print(ans)
    return toJson(ans)


# ------- Run WebApp ------- #

if __name__ == '__main__':
    app.run(port=5002, host="0.0.0.0", debug=True, threaded=True)
# serve(app, host="0.0.0.0",port=3001)
