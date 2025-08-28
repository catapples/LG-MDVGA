import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.preprocessing import minmax_scale
from torch import device
from vgaemode import VGAEModel, Parameter,DGCN
# from gcn_vgaemode import gcn_VGAEModel, GCN
from autils import *
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, # 0.001
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-7,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', '-h1', type=int, default=1000, help='Number of units in hidden layer 1.')
parser.add_argument('--alpha', type=float, default=0.45,  # 0.45
                    help='Weight between lncRNA space and protein space')
parser.add_argument('--beta', type=float, default=0.9,
                    help='Hyperparameter beta')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use.')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
set_seed(args.seed, args.cuda)

# swl.csv swp.csv interaction.txt
# lpi2.csv
t = time.time()
f = np.loadtxt('./dataset2/interaction.csv', delimiter=',')

x = np.concatenate((f, f), axis=1)
y = np.concatenate((f, f), axis=1)
mdi = np.concatenate((x, y))
# mdi = f
mdit = torch.from_numpy(mdi).float()

epl = np.loadtxt('./dataset2/ep.csv', delimiter=',')
swl = np.loadtxt('./dataset2/swl.csv', delimiter=',')
gop = np.loadtxt('./dataset2/go.csv', delimiter=',')
swp = np.loadtxt('./dataset2/swp.csv', delimiter=',')


parameter = Parameter(f)
Ilnce, Iproe = parameter()

Ilnc = np.identity(f.shape[0])
# Ilnc = np.zeros((f.shape[0], f.shape[0]))
rna1 = np.concatenate((Ilnce, epl), axis=1)
rna2 = np.concatenate((swl, Ilnce), axis=1)
rnafeat = np.concatenate((rna1, rna2))
# rnafeat = epl
# rnafeat, protfeat = get_syn_sim(mdit, 500, 500)
# rnafeat = (epl+swl)/2
# protfeat = (gop+swp)/2
rnafeat = asym_adj(rnafeat)

Ipro = np.identity(f.shape[1])
# Ipro = np.zeros((f.shape[1], f.shape[1]))
pro1 = np.concatenate((Iproe, gop), axis=1)
pro2 = np.concatenate((swp, Iproe), axis=1)
protfeat = np.concatenate((pro1, pro2))
# protfeat = swp
protfeat = asym_adj(protfeat)

gm = torch.from_numpy(np.array(rnafeat)).float()
gd = torch.from_numpy(np.array(protfeat)).float()
# gm = norm_adj(gm)
# gd = norm_adj(gd)

f_u, sf, f_v = torch.svd_lowrank(mdit, q=1)
u_mul_f = f_u @ (torch.diag(sf))
v_mul_f = f_v @ (torch.diag(sf))
mdit_svd = u_mul_f @ v_mul_f.T
mdit_svd = F.softmax(F.relu(mdit_svd), dim=1)

lnc_u, slnc, lnc_v = torch.svd_lowrank(gm, q=1)
u_mul_lnc = lnc_u @ (torch.diag(slnc))
v_mul_lnc = lnc_v @ (torch.diag(slnc))
gm_svd = u_mul_lnc @ v_mul_lnc.T


pro_u, spro, pro_v = torch.svd_lowrank(gd, q=1)
u_mul_pro = pro_u @ (torch.diag(spro))
v_mul_pro = pro_v @ (torch.diag(spro))
gd_svd = u_mul_pro @ v_mul_pro.T

# gm = norm_adj(gm)
# gd = norm_adj(gd)

# gm = torch.from_numpy(np.array(gm)).float()
# gd = torch.from_numpy(np.array(gd)).float()


if args.cuda:
    mdit = mdit.cuda()
    mdit_svd = mdit_svd.cuda()
    gm = gm.cuda()
    gd = gd.cuda()
    gm_svd = gm_svd.cuda()
    gd_svd = gd_svd.cuda()


class VGAEL(nn.Module):
    def __init__(self):
        super(VGAEL, self).__init__()
        self.vgael = VGAEModel(mdi.shape[1], args.hidden1)

    def forward(self, y0, y0_svd):
        yl = self.vgael(gm, y0)
        yl_svd = self.vgael(gm_svd, y0_svd)
        return yl, yl_svd


class VGAEP(nn.Module):
    def __init__(self):
        super(VGAEP, self).__init__()
        self.vgaep = VGAEModel(mdi.shape[0], args.hidden1)

    def forward(self, y0, y0_svd):
        yp = self.vgaep(gd, y0.t())
        yp_svd = self.vgaep(gd_svd, y0_svd.t())
        return yp, yp_svd


def train(vgael, vgaep, y0, y0_svd, epoch, alpha, parameter):
    beta = args.beta
    optl = torch.optim.Adam(vgael.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-5)
    optp = torch.optim.Adam(vgaep.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-5)
    opta = torch.optim.Adam(parameter.parameters(), lr=0.01, weight_decay=args.weight_decay)
    for e in range(epoch):
        vgael.train()
        vgaep.train()
        parameter.train()
        yl, yl_svd = vgael(y0, y0_svd)
        yp, yp_svd = vgaep(y0, y0_svd)

        # np.savetxt('yp.csv', yp.detach().numpy(), fmt='%10.5f', delimiter=',')
        KLl = F.kl_div(yl.softmax(dim=-1).log(), y0.softmax(dim=-1), reduction='batchmean')
        KLp = F.kl_div(yp.softmax(dim=-1).log(), y0.t().softmax(dim=-1), reduction='batchmean')

        loss1 = KLl
        loss2 = KLp

        z = alpha * yl + (1 - alpha) * yp.t()
        f1 = cut(z)
        # f1 = z

        nceloss = InfoNCELoss()
        # loss = alpha * loss1 + (1 - alpha) * loss2  # cl_loss =0
        cl_loss = nceloss(y0, yl, yl_svd) + nceloss(y0.T, yp, yp_svd)
        # cl_loss = nceloss(y0.T, yp, yp_svd)  # alpha=0
        # cl_loss = nceloss(y0, yl, yl_svd)  # alpha=1
        loss = beta * (alpha * loss1 + (1 - alpha) * loss2) + (1 - beta) * cl_loss


        optp.zero_grad()
        optl.zero_grad()
        opta.zero_grad()
        loss.backward()
        optp.step()
        optl.step()
        opta.zero_grad()
        vgaep.eval()
        vgael.eval()
        parameter.eval()
        if e % 20 == 0 and e != 0:
            print('Epoch %d | Loss: %.4f' % (e, loss.item()))

    return f1


def trainres(A0, A0_svd, parameter):
    vgael = VGAEL()
    vgaep = VGAEP()
    if args.cuda:
        vgael = vgael.cuda()
        vgaep = vgaep.cuda()
        parameter = parameter.cuda()

    return train(vgael, vgaep, A0, A0_svd, args.epochs, args.alpha,parameter)


def fivefoldcv(A, A_svd):
    N = int(A.shape[0] / 2)
    idx = np.arange(N)
    np.random.shuffle(idx)
    aurocl = np.zeros(5)
    auprl = np.zeros(5)
    pl = np.zeros(5)
    rl = np.zeros(5)
    f1l = np.zeros(5)
    accl = np.zeros(5)  # 添加accuracy数组

    for i in range(5):
        print("Fold {}".format(i + 1))
        A0 = A.clone()
        A0_svd = A_svd.clone()
        for j in range(i * N // 5, (i + 1) * N // 5):
            A0[idx[j], :] = torch.zeros(A.shape[1])
            A0[idx[j] + N, :] = torch.zeros(A.shape[1])
            A0_svd[idx[j], :] = torch.zeros(A_svd.shape[1])
            A0_svd[idx[j] + N, :] = torch.zeros(A_svd.shape[1])

        resi = trainres(A0, A0_svd,parameter)
        if args.cuda:
            resi = resi.cpu().detach().numpy()
        else:
            resi = resi.detach().numpy()

        auroc, aupr, p, r, f1, acc = show_auc(resi, cut(A).cpu())  # 接收accuracy
        aurocl[i] = auroc
        auprl[i] = aupr
        pl[i] = p
        rl[i] = r
        f1l[i] = f1
        accl[i] = acc  # 保存accuracy
        print('AUROC= %.4f|AUPR= %.4f|recall=%.4f|precision=%.4f|f1=%.4f|accuracy=%.4f' % (aurocl[i], auprl[i], rl[i], pl[i], f1l[i], accl[i]))

    print("===Final result===")
    print('AUROC= %.4f +- %.4f | AUPR= %.4f +- %.4f' % (aurocl.mean(), aurocl.std(), auprl.mean(), auprl.std()))
    print('recall= %.4f +- %.4f | precision= %.4f +- %.4f | f1score= %.4f +- %.4f | accuracy= %.4f +- %.4f' % (
        rl.mean(), rl.std(), pl.mean(), pl.std(), f1l.mean(), f1l.std(), accl.mean(), accl.std()))
    print("time=", "{:.5f}".format(time.time() - t))



fivefoldcv(mdit, mdit_svd)