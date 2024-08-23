import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import sys
sys.path.append("..")

class DomainLevelGragh(nn.Module):
    def __init__(self, in_dim, do_emb_size, eg_emb_size, pretrain=True):
        super(DomainLevelGragh, self).__init__()
        self.pretrain = pretrain
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.gcn_v = GCN_V(in_dim=in_dim, out_dim=in_dim)
        self.gcn_e = GCN_E(in_dim=in_dim, out_dim=eg_emb_size)

        self.hyperpred = HyperPred(in_dim + eg_emb_size)

        self.domain_learner = DomainBranch(in_dim=in_dim, out_dim=do_emb_size)

        # self.d_cls = nn.Sequential(nn.Linear(do_emb_size, do_emb_size // 2, bias=True),
        #                            nn.ReLU(),
        #                            nn.Linear(do_emb_size // 2, do_emb_size // 4, bias=True),
        #                            nn.ReLU(),
        #                            nn.Linear(do_emb_size // 4, 25, bias=True))


    def forward(self, X):
        '''
        x: (N, 2048, 7, 7) extracted feature
        N: batch size (i.e. number of domain graph nodes)
        do_emb_size                 =>  P: node embedding size in domain graph
        in_emb_size                 =>  K: node embedding size in instance graph
        '''
        # embedding --> (X, A)

        ##  如果特征是四维的，则使用这行代码
        # X = self.global_pool(X)[:, :, 0, 0]  # (N, out_dim, 7, 7) -> (N, out_dim)

        do_emb = self.gcn_v(X)  # fc --> (N, P)
        eg_emb = self.gcn_e(X)  # GCN --> (N^2, K)

        # level prediction
        if self.pretrain:
            eg_emb_eg = eg_emb.view(do_emb.size(0), do_emb.size(0), -1).mean(1)  # (N^2, K) --> (N, N, K) --> (N, K)
        else:
            eg_emb_ = eg_emb.view(do_emb.size(0), do_emb.size(0), -1)
            eg_emb_eg = (eg_emb_ * torch.eye(do_emb.size(0)).cuda().unsqueeze(-1).expand(-1,-1,eg_emb.size(-1))).sum(1)
        mean, scale = self.hyperpred(torch.cat([do_emb, eg_emb_eg], -1))
        level_pred = self._level_vae(mean, scale)  # (N^2, 1)

        # domain GCN
        eg_emb_do = eg_emb.mean(1).view(do_emb.size(0), do_emb.size(0))  # (N^2, K) -> (N^2, 1) -> (N, N)
        do_emb_1, do_A_1 = self.domain_learner(do_emb, eg_emb_do)  # (N, P), (N, N)
        # type_pred = self.d_cls(do_emb_1)

        # level GCN
        # print(do_emb.size(), eg_emb.size())

        return do_emb, eg_emb_eg, level_pred, do_emb_1

    def _level_vae(self,mean,scale):
        # (N, P)
        # noise = torch.randn(mean.size()).cuda()
        noise = torch.cuda.FloatTensor(mean.size()) if torch.cuda.is_available() else torch.FloatTensor(mean.size())
        torch.randn(mean.size(), out=noise)
        level_pred = mean + noise * scale
        return level_pred


class DomainBranch(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DomainBranch, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gcn = GCN(in_dim=in_dim, out_dim=out_dim)


    def forward(self, X, A):
        A = graph_norm(A, self_loop=True, symmetric=True)
        X = self.gcn(X, A)                              # (N, out_dim) <=> (N, P)
        return X, A




class GCN_V(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN_V, self).__init__()
        # in 2048
        self.gcn = GCN(in_dim, out_dim)
        self.fc = nn.Sequential(nn.Linear(in_dim, in_dim, bias=True),  # 1024
                                nn.ReLU(),
                                nn.Linear(in_dim, in_dim, bias=True),  # 512
                                nn.ReLU(),
                                nn.Linear(in_dim, out_dim, bias=True))  # 256

    def forward(self, X):
        # A = cal_similarity(X)  # (N, N)
        # A = graph_norm(A, self_loop=True, symmetric=True)
        # X = self.gcn(X, A)  # (N, P)
        X = self.fc(X)
        return X

class GCN_E(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN_E, self).__init__()
        # in 2048
        self.gcn = GCN(in_dim, out_dim)

    def forward(self, X):
        X = cal_edge_emb(X)  # (N^2, K)
        A = cal_similarity(X)   # (N^2, N^2)
        A = graph_norm(A, self_loop=True, symmetric=True)
        X = self.gcn(X, A)  # (N^2, P)
        return X

class HyperPred(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super(HyperPred, self).__init__()

        self.fc = nn.Sequential(nn.Linear(in_dim, in_dim // 2, bias=True),
                                  nn.ReLU(),
                                  nn.Linear(in_dim // 2, out_dim, bias=True))
    def forward(self, x):
        # input: (N, K)
        # output: (N, 2)
        x = self.fc(x)
        mean, scale = x.split(1, dim=1)  # (N, 1) * 2
        return mean, scale


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()

        # make sure out_dim >= 4*in_dim
        self.W1 = nn.Linear(in_dim, in_dim//2, bias=False)
        self.W2 = nn.Linear(in_dim//2, in_dim//4, bias=False)
        self.W3 = nn.Linear(in_dim//4, out_dim, bias=False)

    def forward(self, X, A):
        # X: (N, dim); A: (N, N)
        # d = A.mm(X)
        f = torch.isnan(A)
        m = torch.isnan(X)
        # if torch.isnan(A) == True :
        #     print('AAAA')
        # if torch.isnan(X):
        #     print('xxxxx')
        X = F.relu(self.W1(A.mm(X)))
        X = F.relu(self.W2(A.mm(X)))
        X = self.W3(A.mm(X))
        return X


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        if features.dim() == 2:
            x = torch.spmm(A, features)
        elif features.dim() == 3:
            x = torch.bmm(A, features)
        else:
            raise RuntimeError('the dimension of features should be 2 or 3')
        return x
    
    
def graph_norm(A, batch=False, self_loop=True, symmetric=True):
	# A = A + I    A: (bs, num_nodes, num_nodes)
    if self_loop:
        eye = torch.eye(A.size(1)).expand(A.size())
        eye = eye.cuda() if A.is_cuda else eye
        A = A + eye

    # Degree
    d = A.sum(-1) # (bs, num_nodes)
    if symmetric:
		# D = D^-1/2
        d = torch.pow(d, -0.5)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A).bmm(D)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A).mm(D)
    else:
		# D=D^-1
        d = torch.pow(d,-1)
        if batch:
            D = A.detach().clone()
            for i in A.size(0):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A)
        else:
            D =torch.diag(d)
            norm_A = D.mm(A)

    return norm_A


def cal_similarity(x, p=2, dim=1):
    '''
    x: (n,K)
    return: (n,n)
    '''
    x = F.normalize(x, p=p, dim=dim)
    return torch.mm(x, x.transpose(0, 1))

def cal_edge_emb(x, p=2, dim=1):
    '''
    x: (n,K)
    return: (n^2, K)
    '''
    x = F.normalize(x, p=p, dim=dim)
    x_r = torch.transpose(x, 0, 1).unsqueeze(2)  # (K, n, 1)
    x_c = torch.transpose(x, 0, 1).unsqueeze(1)  # (K, 1, n)
    A = torch.bmm(x_r, x_c).permute(1,2,0)  # (n, n, K)

    A = A.view(A.size(0) * A.size(1), A.size(2))  # (n^2, K)
    # print(A.size())
    return A.cuda()
