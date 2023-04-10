from ianchor.anchor import Anchor, Tasktype
from models.data_process import *
from models.run_MLP import load_model


def greedyFind(exp_len):
    method_paras = {"desired_confidence": 1.0}

    ans = {}
    ans['total'] = exp_len
    for i in range(exp_len):
        anchors = explainer.explain_instance(
            input=test_dataset[i].reshape(1, -1),
            predict_fn=predict_fn,
            method="greedy",
            task_specific=task_paras,
            method_specific=method_paras,
            num_coverage_samples=100,
        )
        ans[str(i)] = {}
        exp_visu = [
            f"{k}"
            for i, k in enumerate(adult_process_names)
            if i in anchors.feature_mask
        ]
        ans[str(i)]['feature'] = exp_visu
        ans[str(i)]['precision'] = anchors.precision
        ans[str(i)]['coverage'] = anchors.coverage
        ans[str(i)]['covered_true'] = anchors.covered_true
        ans[str(i)]['covered_false'] = anchors.covered_false

    save_json(ans, args.out_dir + getFileName('ianchors_greedy', 'json'), overwrite_if_exists=True)


def beamFind(exp_len):
    method_paras = {"beam_size": 1, "desired_confidence": 1.0}

    ans = {}
    ans['total'] = exp_len
    for i in range(exp_len):
        anchor = explainer.explain_instance(
            input=test_dataset[i].reshape(1, -1),
            predict_fn=predict_fn,
            method="beam",
            task_specific=task_paras,
            method_specific=method_paras,
            num_coverage_samples=100,
        )
        best_of_size = anchor['best_of_size']
        best_candidate = best_of_size.get(list(best_of_size.keys())[-1])[0]
        ans[str(i)] = {}
        exp_visu = [
            f"{k}"
            for i, k in enumerate(adult_process_names)
            if i in best_candidate.feature_mask
        ]
        precision_list = []
        coverage_list = []
        covered_true_list = []
        covered_false_list = []
        for j in range(1, len(best_of_size)):
            cand = best_of_size[j][0]
            precision_list.append(cand.precision)
            coverage_list.append(cand.coverage)
            covered_true_list.append(np.array(cand.covered_true[:10]))
            covered_false_list.append(np.array(cand.covered_false[:10]))

        ans[str(i)]['feature'] = exp_visu
        ans[str(i)]['precision'] = precision_list
        ans[str(i)]['coverage'] = coverage_list
        ans[str(i)]['covered_true'] = covered_true_list
        ans[str(i)]['covered_false'] = covered_false_list

    save_json(ans, args.out_dir + getFileName('ianchors_beam', 'json'), overwrite_if_exists=True)


if __name__ == "__main__":
    model = load_model()
    # 加载划分数据集
    dataset, target, encoder, categorical_names = load_adult_income_dataset()
    train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                    target,
                                                                    test_size=0.2,
                                                                    random_state=args.random_state,
                                                                    stratify=target)

    predict_fn = lambda x: model.predict_anchor(x, encoder)

    explainer = Anchor(Tasktype.TABULAR)

    task_paras = {"dataset": train_dataset,
                  "column_names": adult_process_names}

    # exp_len = test_dataset.shape[0]
    exp_len = 10
    # greedyFind(exp_len)
    beamFind(exp_len)
    # print(anchor)
    # print(explainer.visualize(anchor, test_dataset[1], adult_process_names))
