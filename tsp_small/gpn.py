import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.size = 0
        self.batch_size = 0
        self.dim = n_hidden

        v = torch.FloatTensor(n_hidden).cuda()
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-1 / math.sqrt(n_hidden), 1 / math.sqrt(n_hidden))  # 初始化

        # parameters for pointer attention
        self.Wref = nn.Linear(n_hidden, n_hidden)
        self.Wq = nn.Linear(n_hidden, n_hidden)

    def forward(self, q, ref):  # query and reference
        # q [batch_size, n_hidden]
        # ref [size*batch_size, n_hidden]
        self.batch_size = q.size(0)
        self.size = int(ref.size(0) / self.batch_size)  # size of TSP
        q = self.Wq(q)  # q [batch_size, n_hidden]
        ref = self.Wref(ref)  # ref [size*batch_size, n_hidden]
        ref = ref.view(self.batch_size, self.size, self.dim)  # ref [batch_size, size, n_hidden]

        q_ex = q.unsqueeze(1).repeat(1, self.size, 1)  # q_ex [batch_size, size, n_hidden]

        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)  # v_view [batch_size, n_hidden, 1]

        # [batch_size, size, n_hidden] * [batch_size, n_hidden, 1] -> [batch_size, size, 1]
        u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2)  # .squeeze(2)去掉第三维
        # u [batch_size, size]
        return u, ref


class LSTM(nn.Module):
    def __init__(self, n_hidden):
        super(LSTM, self).__init__()

        # parameters for input gate
        self.Wxi = nn.Linear(n_hidden, n_hidden)  # W(xt)
        self.Whi = nn.Linear(n_hidden, n_hidden)  # W(ht)
        self.wci = nn.Linear(n_hidden, n_hidden)  # w(ct)

        # parameters for forget gate
        self.Wxf = nn.Linear(n_hidden, n_hidden)  # W(xt)
        self.Whf = nn.Linear(n_hidden, n_hidden)  # W(ht)
        self.wcf = nn.Linear(n_hidden, n_hidden)  # w(ct)

        # parameters for cell gate
        self.Wxc = nn.Linear(n_hidden, n_hidden)  # W(xt)
        self.Whc = nn.Linear(n_hidden, n_hidden)  # W(ht)

        # parameters for forget gate
        self.Wxo = nn.Linear(n_hidden, n_hidden)  # W(xt)
        self.Who = nn.Linear(n_hidden, n_hidden)  # W(ht)
        self.wco = nn.Linear(n_hidden, n_hidden)  # w(ct)

    def forward(self, x, h, c):  # query and reference

        # input gate
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
        # forget gate
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
        # cell gate
        c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
        # output gate
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))

        h = o * torch.tanh(c)

        return h, c


class GPN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(GPN, self).__init__()
        self.city_size = 0
        self.batch_size = 0
        self.dim = n_hidden

        # lstm for first turn
        self.lstm0 = nn.LSTM(n_hidden, n_hidden)

        # pointer layer
        self.pointer = Attention(n_hidden)

        # lstm encoder
        self.encoder = LSTM(n_hidden)

        # trainable first hidden input
        h0 = torch.FloatTensor(n_hidden).cuda()
        c0 = torch.FloatTensor(n_hidden).cuda()

        # trainable latent variable coefficient
        alpha = torch.ones(1).cuda()

        self.h0 = nn.Parameter(h0)
        self.c0 = nn.Parameter(c0)

        self.alpha = nn.Parameter(alpha)
        self.h0.data.uniform_(-1 / math.sqrt(n_hidden), 1 / math.sqrt(n_hidden))
        self.c0.data.uniform_(-1 / math.sqrt(n_hidden), 1 / math.sqrt(n_hidden))

        r1 = torch.ones(1).cuda()
        r2 = torch.ones(1).cuda()
        r3 = torch.ones(1).cuda()
        self.r1 = nn.Parameter(r1)
        self.r2 = nn.Parameter(r2)
        self.r3 = nn.Parameter(r3)

        # embedding
        self.embedding_x = nn.Linear(n_feature, n_hidden)
        self.embedding_all = nn.Linear(n_feature, n_hidden)

        # weights for GNN
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_hidden)
        self.W3 = nn.Linear(n_hidden, n_hidden)

        # aggregation function for GNN
        self.agg_1 = nn.Linear(n_hidden, n_hidden)
        self.agg_2 = nn.Linear(n_hidden, n_hidden)
        self.agg_3 = nn.Linear(n_hidden, n_hidden)

    def forward(self, x, X_all, mask, h=None, c=None, latent=None):
        '''
        Inputs (B: batch size, size: city size, dim: hidden dimension)
        
        x: current city coordinate (B, 2)
        X_all: all cities' cooridnates (B, size, 2)
        mask: mask visited cities
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent: latent pointer vector from previous layer (B, size, dim)
        
        Outputs
        
        softmax: probability distribution of next city (B, size)
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent_u: latent pointer vector for next layer
        '''

        self.batch_size = X_all.size(0)
        self.city_size = X_all.size(1)

        # =============================
        # vector context
        # =============================

        # x_expand = x.unsqueeze(1).repeat(1, self.city_size, 1)   # (B, size)
        # X_all = X_all - x_expand

        # the weights share across all the cities
        x = self.embedding_x(x)  # [batch_size, n_hidden]
        context = self.embedding_all(X_all)  # [batch_size, size, n_hidden]

        # =============================
        # process hidden variable
        # =============================

        first_turn = False
        if h is None or c is None:
            first_turn = True

        if first_turn:
            # (dim) -> (B, dim)
            h0 = self.h0.unsqueeze(0).expand(self.batch_size, self.dim)  # [batch_size, n_hidden]
            c0 = self.c0.unsqueeze(0).expand(self.batch_size, self.dim)

            h0 = h0.unsqueeze(0).contiguous()  # contiguous()拷贝一份
            c0 = c0.unsqueeze(0).contiguous()  # [1, batch_size, n_hidden]

            input_context = context.permute(1, 0, 2).contiguous()  # permute()将维度换位 [size, batch_size, n_hidden]
            _, (h_enc, c_enc) = self.lstm0(input_context, (h0, c0))
            # 官方的LSTM输入由两部分组成：input、(初始的隐状态h_0，初始的单元状态c_0)
            # input [seq_len, batch_size, input_size]
            # h_0, c_0 [num_directions*num_layers, batch_size, hidden_size]
            # 输出也由两部分组成：output、(隐状态h_n，单元状态c_n)
            # output [seq_len, batch_size, num_directions*hidden_size]
            # h_n和c_n的shape保持不变

            # let h0, c0 be the hidden variable of first turn
            h = h_enc.squeeze(0)  # [batch_size, n_hidden]
            c = c_enc.squeeze(0)

        # =============================
        # graph neural network encoder
        # =============================

        context = context.view(-1, self.dim)  # [size*batch_size, n_hidden]

        context = self.r1 * self.W1(context) \
                  + (1 - self.r1) * F.relu(self.agg_1(context))

        context = self.r2 * self.W2(context) \
                  + (1 - self.r2) * F.relu(self.agg_2(context))

        context = self.r3 * self.W3(context) \
                  + (1 - self.r3) * F.relu(self.agg_3(context))

        # LSTM encoder
        h, c = self.encoder(x, h, c)

        # query vector
        q = h

        # pointer
        u, _ = self.pointer(q, context)

        latent_u = u.clone()

        u = 10 * torch.tanh(u) + mask

        if latent is not None:
            u += self.alpha * latent

        return F.softmax(u, dim=1), h, c, latent_u
