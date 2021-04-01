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


def get_sim_stats(sims):
    best25_mean, best25_var, best25_std = [], [], []
    for sim in sims:
        best25_ids = np.argsort(sim)[-10:]
        best25_mean.append(np.mean(sim[best25_ids]))
        best25_var.append(np.var(sim[best25_ids]))
        best25_std.append(np.std(sim[best25_ids]))
    return best25_mean, best25_var, best25_std


def compute_thres(mean_sim, std_sim, var_sim, var_q50):
    if var_sim < var_q50:
        th = mean_sim - 0.1 * std_sim
    else:
        th = mean_sim - 0.5 * std_sim
    return th


def validate_score(df, embeeds, th):
    sims = emb_sim(embeeds)
    sims = sims.cpu().numpy()
    # add some similarity scores statistics before thresholding
    best25_mean, best25_var, best25_std = get_sim_stats(sims)
    df["best25_mean"] = best25_mean
    df["best25_var"] = best25_var
    df["best25_std"] = best25_std
    var_q50 = np.quantile(df.best25_var, q=0.5)
    th = df.apply(
        lambda x: compute_thres(
            x["best25_mean"], x["best25_std"], x["best25_var"], var_q50
        ),
        axis=1,
    )
    th = th.values[:, None]
    sims = sims > th
    add_ground_truth(df)
    add_predictions(df, sims)
    scores, f1mean = row_wise_f1_score(df["true_postings"], df["pred_postings"])
    df["f1"] = scores
    return f1mean, df


def make_submit_df(df, embeeds, th):
    sims = emb_sim(embeeds)
    sims = sims.cpu().numpy()
    sims = sims > th

    assert len(df) == len(sims)
    preds = []
    for i in range(len(df)):
        preds.append(" ".join(df["posting_id"].values[sims[i]]))
    df["matches"] = preds
    return df
