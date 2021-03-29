"""https://www.kaggle.com/c/shopee-product-matching/discussion/228424"""
import numpy as np
from torch.nn import functional as F


# Cosine similiarity across all pairs of rows
def emb_sim(inp):
    inp = F.normalize(inp)
    return inp @ inp.T


# F1 score of prediction groups based on target_groups
def score_groups(pred_matrix, target_matrix):
    intersect = pred_matrix.logical_and(target_matrix)
    f1s = 2 * intersect.sum(dim=1) / (target_matrix.sum(dim=1) + pred_matrix.sum(dim=1))
    return f1s.mean()


# Calculates score for all values of threshold with the given range and increment
def treshold_finder(embs, y, start=0.7, end=1, step=0.01):
    sims = emb_sim(embs)
    target_matrix = y[:, None] == y[None, :]
    scores, ts = [], np.arange(start, end, step)
    for t in ts:
        scores.append(score_groups(sims > t, target_matrix))
    best = np.array(scores).argmax()
    return scores[best], ts[best]


# predictions to dataframe
def add_ground_truth(df):
    tmp = df.groupby("label_group").posting_id.agg("unique").to_dict()
    df["true_postings"] = df.label_group.map(tmp)


def add_predictions(df, sims):
    assert len(df) == len(sims)
    preds = []
    for i in range(len(df)):
        preds.append(df["posting_id"].values[sims[i]])
    df["pred_postings"] = preds


def row_wise_f1_score(labels, preds):
    scores = []
    for label, pred in zip(labels, preds):
        n = len(np.intersect1d(label, pred))
        score = 2 * n / (len(label) + len(pred))
        scores.append(score)
    return scores, np.mean(scores)


def validate_score(df, embeeds, th):
    sims = emb_sim(embeeds)
    sims = sims.cpu().numpy()
    sims = sims > th
    add_ground_truth(df)
    add_predictions(df, sims)
    scores, f1mean = row_wise_f1_score(df["true_postings"], df["pred_postings"])
    df["f1"] = scores
    return f1mean, df
