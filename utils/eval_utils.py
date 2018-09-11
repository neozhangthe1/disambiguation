import numpy as np
from keras import backend as K
from sklearn.metrics import roc_auc_score


def pairwise_precision_recall_f1(preds, truths):
    tp = 0
    fp = 0
    fn = 0
    n_samples = len(preds)
    for i in range(n_samples - 1):
        pred_i = preds[i]
        for j in range(i + 1, n_samples):
            pred_j = preds[j]
            if pred_i == pred_j:
                if truths[i] == truths[j]:
                    tp += 1
                else:
                    fp += 1
            elif truths[i] == truths[j]:
                fn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    if tp_plus_fp == 0:
        precision = 0.
    else:
        precision = tp / tp_plus_fp
    if tp_plus_fn == 0:
        recall = 0.
    else:
        recall = tp / tp_plus_fn

    if not precision or not recall:
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def cal_f1(prec, rec):
    return 2*prec*rec/(prec+rec)


def get_hidden_output(model, inp):
    get_activations = K.function(model.inputs[:1] + [K.learning_phase()], [model.layers[5].get_output_at(0), ])
    activations = get_activations([inp, 0])
    return activations[0]


def predict(anchor_emb, test_embs):
    score1 = np.linalg.norm(anchor_emb-test_embs[0])
    score2 = np.linalg.norm(anchor_emb-test_embs[1])
    return [score1, score2]


def full_auc(model, test_triplets):
    """
    Measure AUC for model and ground truth on all items.

    Returns:
    - float AUC
    """

    grnds = []
    preds = []
    preds_before = []
    embs_anchor, embs_pos, embs_neg = test_triplets

    inter_embs_anchor = get_hidden_output(model, embs_anchor)
    inter_embs_pos = get_hidden_output(model, embs_pos)
    inter_embs_neg = get_hidden_output(model, embs_neg)
    # print(inter_embs_pos.shape)

    accs = []
    accs_before = []

    for i, e in enumerate(inter_embs_anchor):
        if i % 10000 == 0:
            print('test', i)

        emb_anchor = e
        emb_pos = inter_embs_pos[i]
        emb_neg = inter_embs_neg[i]
        test_embs = np.array([emb_pos, emb_neg])

        emb_anchor_before = embs_anchor[i]
        emb_pos_before = embs_pos[i]
        emb_neg_before = embs_neg[i]
        test_embs_before = np.array([emb_pos_before, emb_neg_before])

        predictions = predict(emb_anchor, test_embs)
        predictions_before = predict(emb_anchor_before, test_embs_before)

        acc_before = 1 if predictions_before[0] < predictions_before[1] else 0
        acc = 1 if predictions[0] < predictions[1] else 0
        accs_before.append(acc_before)
        accs.append(acc)

        grnd = [0, 1]
        grnds += grnd
        preds += predictions
        preds_before += predictions_before

    auc_before = roc_auc_score(grnds, preds_before)
    auc = roc_auc_score(grnds, preds)
    print('test accuracy before', np.mean(accs_before))
    print('test accuracy after', np.mean(accs))

    print('test AUC before', auc_before)
    print('test AUC after', auc)
    return auc
