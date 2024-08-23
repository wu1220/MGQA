import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_vqa_attn_mask(video_length, max_len):
    batch_size = len(video_length)
    attn_mask = torch.zeros(batch_size, max_len)
    for idx in range(batch_size):
        attn_mask[idx, int(video_length[idx]):] = 1
    attn_mask = attn_mask.data.eq(1).unsqueeze(1).expand(batch_size, max_len, max_len)
    
    subsequence_mask = torch.triu(torch.ones(batch_size, max_len, max_len), diagonal=1)
    subsequence_mask = subsequence_mask.data.eq(1)
    # print(subsequence_mask)
    
    return attn_mask + subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        
    def forward(self, Q, K, V, attn_mask):
        '''
        
        :param Q: [batch_size, n_heads, max_len_q, d_k]
        :param K: [batch_size, n_heads, max_len_k, d_k]
        :param V: [batch_size, n_heads, max_len_v, d_v]
        :param attn_mask: [batch_size, n_heads, max_len_q, max_len_k]
        :return:
        '''
        d_k = K.shape[-1]
        # attn: [batch_size, n_heads, max_len_q, max_len_k]
        attn = torch.matmul(Q, K.transpose(-1, -2) / np.sqrt(d_k))
        attn.masked_fill_(attn_mask, -1e9)
        attn = F.softmax(attn, dim=-1)
        # context: [bach_size, n_heads, max_len_q, d_v]
        context = torch.matmul(attn, V)
        
        return context, attn
    
    
class MultiAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, n_heads*d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads*d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads*d_v, bias=False)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads*d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, X, attn_mask):
        '''
        
        :param X: [batch_size, max_len, d_model]
        :param attn_mask: [batch_size, max_len, max_len]
        :return:
        '''
        batck_size, max_len, _ = X.shape
        Q = self.W_Q(X).view(batck_size, max_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(X).view(batck_size, max_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(X).view(batck_size, max_len, self.n_heads, self.d_v).transpose(1, 2)
        
        # context: [bach_size, n_heads, max_len_q, d_v]
        # attn: [batch_size, n_heads, max_len_q, max_len_k]
        context, attn = self.attention(Q, K, V, attn_mask)
        # context: [bach_size, n_heads, max_len_q, d_v] -> [batch_size, max_len_q, n_heads*d_v]
        context = context.transpose(1, 2).reshape(batck_size, -1, self.n_heads*self.d_v)
        # output: [batch_size, max_len_q, d_model]
        output = self.fc(context)
        
        return self.layer_norm(output+X), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_fc):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_fc, bias=False),
            nn.ReLU(),
            nn.Linear(d_fc, d_model, bias=False),
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, X):
        '''
        
        :param X: [batch_size, max_len, d_model]
        :return:
        '''
        output = self.fc(X)
        return self.layer_norm(output + X)
        
        
class EDLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_fc):
        super(EDLayer, self).__init__()
        self.self_attn = MultiAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_fc)
        
    def forward(self, X, attn_mask):
        # ed_output: [batch_size, max_len, d_model]
        # attn: [batch_size, n_heads, max_len_q, max_len_k]
        ed_output, attn = self.self_attn(X, attn_mask)
        ed_output = self.pos_ffn(ed_output)
        
        return ed_output, attn


class EDCoder(nn.Module):
    def __init__(self, max_len, n_layers, d_model, n_heads, d_k, d_v, d_fc):
        super(EDCoder, self).__init__()
        self.max_len = max_len
        self.n_heads = n_heads
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model), requires_grad=True)
        self.layers = nn.ModuleList([EDLayer(d_model, n_heads, d_k, d_v, d_fc) for _ in range(n_layers)])
        
    def forward(self, X, video_length):
        X = X + self.pos_emb
        # attn_mask: [batch_size, n_heads, max_len_q, max_len_k]
        attn_mask = get_vqa_attn_mask(video_length, self.max_len).unsqueeze(1).repeat(1, self.n_heads, 1, 1).to(X.device)
        attns = []
        for layer in self.layers:
            X, attn = layer(X, attn_mask)
            attns.append(attn)
            
        return X, attns


class ANN(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)  #
        self.dropout = nn.Dropout(p=dropout_p)  #
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers-1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input


def TP(q, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


class TransformerVSFA(nn.Module):
    def __init__(self, input_size=8000, reduced_size=128, hidden_size=32, max_len=300, n_layers=6, n_heads=8, d_k=64, d_v=64):
        super(TransformerVSFA, self).__init__()
        '''
        max_len = 200
        n_layers = 2
        d_model = 2048
        n_heads = 3
        d_k = 64
        d_v = 64
        d_fc = 4096
        '''
        self.hidden_size = hidden_size
        self.ann = ANN(input_size, reduced_size, 1)
        # self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.attention = EDCoder(max_len, n_layers, reduced_size, n_heads, d_k, d_v, hidden_size)
        self.fc = nn.Linear(reduced_size, hidden_size)
        self.relu = nn.ReLU()
        
        self.q = nn.Linear(hidden_size, 1)

    def forward(self, input, input_length):
        input = self.ann(input)  # dimension reduction
        # outputs, _ = self.rnn(input, self._get_initial_state(input.size(0), input.device))
        outputs, attn = self.attention(input, input_length)
        outputs = self.relu(self.fc(outputs))
        q = self.q(outputs)  # frame quality
        # score = torch.zeros_like(input_length, device=q.device)  #
        # for i in range(input_length.shape[0]):  #
        #     qi = q[i, :int(input_length[i].cpu().numpy())]
        #     qi = TP(qi)
        #     score[i] = torch.mean(qi)  # video overall quality
        # return score
        return torch.mean(q,dim=1)

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


if __name__ == '__main__':
    batch_size = 5
    video_length = torch.tensor([192, 184, 200, 191, 50], dtype=torch.int)
    max_len = 200
    n_layers = 2
    d_model = 2048
    n_heads = 3
    d_k = 64
    d_v = 64
    d_fc = 4096
    # model = EDCoder(max_len, n_layers, d_model, n_heads, d_k, d_v, d_fc)
    model = TransformerVSFA(d_model)
    X = torch.randn(batch_size, max_len, d_model)
    outputs, attn = model(X, video_length)
    print(outputs.shape, len(attn))
    print(outputs)
    
    
