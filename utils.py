import torch
from torch.autograd import Variable
import json
import numpy as np

config = json.load(open('./config.json', 'r'))


def Z_Score(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std


def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var


def CalConfusionMatrix(confusion_matrix):
    TP, FP, FN, TN, precise, recall, f1_score = 0, 0, 0, 0, 0, 0, 0
    n = confusion_matrix.shape[0]
    for i in range(n):
        TP = confusion_matrix[i][i]
        FP = (confusion_matrix[i].sum() - TP)
        FN = (confusion_matrix[:, i].sum() - TP)
        TN = (confusion_matrix.sum() - TP - FP - FN)
        if TP != 0:
            precise_temp = Precise(TP, FP)
            precise += precise_temp
            recall_temp = ReCall(TP, FN)
            recall += recall_temp
            f1_score += F1_Score(precise_temp, recall_temp)
        else:
            precise += 0.
            recall += 0.
            f1_score += 0.
    return precise / n, recall / n, f1_score / n


def Precise(TP, FP):
    return TP / (TP + FP)


def ReCall(TP, FN):
    return TP / (TP + FN)


def F1_Score(P, R):
    return 2 * P * R / (P + R)
