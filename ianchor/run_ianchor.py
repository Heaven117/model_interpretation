from ianchor.anchor import Anchor, Tasktype
from models.data_process import *
from models.run_MLP import load_model


def anchor_to_json():
    anchors = {}
    anchors['total'] = dataset_len


if __name__ == "__main__":
    model = load_model()
    x_dataset, target, encoder, categorical_names = load_adult_income_dataset()

    # adult_process_names.insert(0, 'id')
    predict_fn = lambda x: model.predict_anchor(x, encoder)

    explainer = Anchor(Tasktype.TABULAR)

    task_paras = {"dataset": x_dataset,
                  "column_names": adult_process_names}
    method_paras = {"beam_size": 1, "desired_confidence": 1.0}

    exp_len = 3
    ans = {}
    ans['total'] = exp_len
    for i in range(exp_len):
        anchor = explainer.explain_instance(
            input=x_dataset[i].reshape(1, -1),
            predict_fn=predict_fn,
            method="beam",
            task_specific=task_paras,
            method_specific=method_paras,
            num_coverage_samples=100,
        )
        best_candidate = anchor['best_candidate']
        best_of_size = anchor['best_of_size']
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
        if len(exp_visu) > 0:
            for j in range(1, len(best_of_size)):
                cand = best_of_size[j][0]
                precision_list.append(cand.precision)
                coverage_list.append(cand.coverage)
                covered_true_list.append(np.array(cand.covered_true))
                covered_false_list.append(np.array(cand.covered_false))

            best_candidate = anchor['best_candidate']

        ans[str(i)]['feature'] = exp_visu
        ans[str(i)]['precision'] = precision_list
        ans[str(i)]['coverage'] = coverage_list
        ans[str(i)]['covered_true'] = covered_true_list
        ans[str(i)]['covered_false'] = covered_false_list

    save_json(ans, args.out_dir + getFileName('ianchors', 'json'), overwrite_if_exists=True)
    print(anchor)
    print(explainer.visualize(anchor, x_dataset[1], adult_process_names))
