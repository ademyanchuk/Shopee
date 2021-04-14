"""https://www.kaggle.com/c/shopee-product-matching/discussion/228424"""
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


# Cosine similiarity across all pairs of rows
def emb_sim(inp, chunk_sz=0):
    assert chunk_sz >= 0
    inp = F.normalize(inp)
    if chunk_sz == 0:
        return inp @ inp.T
    else:
        return emb_sim_chunked(inp, chunk_sz)


def emb_sim_chunked(inp, chunk_sz):
    num_chunks = (len(inp) // chunk_sz) + 1
    sims = []
    for i in range(num_chunks):
        a = i * chunk_sz
        b = (i + 1) * chunk_sz
        b = min(b, len(inp))
        print(f"compute similarities for chunks {a} to {b}")
        sim = inp[a:b] @ inp.T
        sims.append(sim.cpu())
    return torch.cat(sims)


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


def binned_threshold_f1(embs: torch.Tensor, y: torch.Tensor):
    sims = emb_sim(embs)
    target_matrix = y[:, None] == y[None, :]
    sim_stats = get_sim_stats_torch(sims)
    quants = torch.quantile(sim_stats, q=torch.tensor([0.3, 0.6, 0.9]))
    th = torch.stack([compute_thres(x, quants) for x in sim_stats])
    th = th[:, None].cuda()
    score = score_groups(sims > th, target_matrix)
    return score.item(), np.nan


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
    best25_mean = []
    for sim in sims:
        best25_ids = np.argsort(sim)[-6:]
        best25_mean.append(np.mean(sim[best25_ids]))
    return best25_mean


def get_sim_stats_torch(sims: torch.Tensor):
    best_mean = []
    for sim in sims:
        best_ids = torch.argsort(sim)[-6:]
        best_mean.append(torch.mean(sim[best_ids]))
    return torch.tensor(best_mean)


def compute_thres(mean_sim, qunts, coeff=0.9):
    if mean_sim <= qunts[0]:
        return qunts[0] * coeff
    elif mean_sim > qunts[0] and mean_sim <= qunts[1]:
        return qunts[1] * coeff
    else:
        return qunts[2] * coeff


def validate_score(df, embeeds, th, chunk_sz=0):
    sims = emb_sim(embeeds, chunk_sz)
    sims = sims.cpu().numpy()
    # add some similarity scores statistics before thresholding
    best25_mean = get_sim_stats(sims)
    df["best25_mean"] = best25_mean
    # qunts = np.quantile(df.best25_mean, q=[0.3, 0.6, 0.9])
    # th = df.apply(lambda x: compute_thres(x["best25_mean"], qunts), axis=1,)
    th = df["best25_mean"] * 0.9
    th = th.values[:, None]
    sims = sims > th
    add_ground_truth(df)
    add_predictions(df, sims)
    scores, f1mean = row_wise_f1_score(df["true_postings"], df["pred_postings"])
    df["f1"] = scores
    return f1mean, df
