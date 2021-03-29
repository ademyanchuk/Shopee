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
