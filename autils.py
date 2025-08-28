import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import scipy.sparse as sp


def scaley(ymat):
    return (ymat - ymat.min()) / ymat.max()


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def normalized(wmat):
    deg = np.diag(np.sum(wmat, axis=0))
    degpow = np.power(deg, -0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow, wmat), degpow)
    return W


def norm_adj(feat):
    feat = feat.numpy()
    adj = normalized(feat + np.eye(feat.shape[0]))
    adj = torch.from_numpy(adj).float()
    return adj


def show_auc(ymat, A_label):
    ldi = A_label
    # y_true = ldi[:, i].flatten()
    y_true = ldi.flatten()
    # ymat = ymat[:, i].flatten()
    ymat = ymat.flatten()
    fpr, tpr, rocth = roc_curve(y_true, ymat)
    auroc = auc(fpr, tpr)
    # np.savetxt('./space/dataset2/only_positive'+str(i)+'_roc.txt',np.vstack((fpr,tpr)),fmt='%10.5f',delimiter=',')
    precision, recall, prth = precision_recall_curve(y_true, ymat)

    aupr = auc(recall, precision)
    # np.savetxt('./space/dataset2/only_positive'+str(i)+'_pr.txt',np.vstack((recall,precision)),fmt='%10.5f',delimiter=',')
    print('AUROC= %.4f | AUPR= %.4f' % (auroc, aupr))

    y_true = np.reshape(y_true, [-1])
    y_pred = np.reshape(ymat, [-1])
    y_pred = np.rint(y_pred)
    # y_pred = np.rint(ymat)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)  # 计算accuracy
    print('precision= %.4f | recall= %.4f | f1score= %.4f | accuracy= %.4f' % (p, r, f1score, acc))
    return auroc, aupr, p, r, f1score, acc  # 返回accuracy
    # if (math.isnan(aupr)):
    #     aupr = 0.0
    # if (math.isnan(auroc)):
    #     auroc = 0.0

    # return auroc, aupr


def cut(y):
    y = y.cpu().detach().numpy()
    row = y.shape[0]
    col = y.shape[1]
    # F1 = torch.zeros(int(row/2), int(col/2)).detach().numpy()
    F1 = np.add(y[0:int(row / 2), 0:int(col / 2)],
                y[int(row / 2):row, int(col / 2):col])
    F1 = np.add(F1, y[0:int(row / 2), int(col / 2):col])
    F1 = np.add(F1, y[int(row / 2):row, 0:int(col / 2)])
    F1 = F1 / 4
    F1 = torch.from_numpy(F1).float()
    return F1


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchors, positives, negatives):
        anchors_norm = F.normalize(anchors, dim=1)
        positives_norm = F.normalize(positives, dim=1)
        negatives_norm = F.normalize(negatives, dim=1)

        positive_similarity = torch.sum(anchors_norm * positives_norm, dim=1)

        # 初始化负样本相似度列表
        negative_similarity_list = []

        # 逐个计算负样本相似度
        for i in range(anchors.size(0)):
            # 对负样本张量进行转置，以确保维度匹配
            negative_similarity = torch.matmul(negatives_norm[i].unsqueeze(0), anchors_norm[i].unsqueeze(1)).squeeze(1)
            negative_similarity_list.append(negative_similarity.unsqueeze(0))

        # 合并结果
        negative_similarity = torch.cat(negative_similarity_list, dim=0)

        # 组合正负样本的相似度，并应用温度缩放
        logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarity], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchors.device)

        # 计算损失
        loss = F.cross_entropy(logits, labels)
        return loss
