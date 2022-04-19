"""
Defining DATE model architecture and Layers
"""

import numpy as np
import torch as torch
import torch.nn as nn
from utils_lib import to_device
from torch.nn.functional import softmax
cuda = torch.device('cuda')


""" GATE GAM"""


class GAM(nn.Module):
    def __init__(self, nin, nhidden, nout, dropout):
        super(GAM, self).__init__()
        self.num_features = nin
        self.num_hidden = nhidden
        self.num_out = nout
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.W_1 = nn.Parameter(torch.empty(size=(nin, nhidden)))
        nn.init.xavier_uniform_(self.W_1.data, gain=np.sqrt(2))

        self.b_1 = nn.Parameter(torch.empty(size=(1, nhidden)))
        nn.init.xavier_uniform_(self.b_1.data, gain=np.sqrt(2))

        self.W_2 = nn.Parameter(torch.empty(size=(nhidden, nout)))
        nn.init.xavier_uniform_(self.W_2.data, gain=np.sqrt(2))

        self.b_2 = nn.Parameter(torch.empty(size=(1, nout)))
        nn.init.xavier_uniform_(self.b_2.data, gain=np.sqrt(2))

    def forward(self, hidden, adj):
        alpha = softmax(adj, dim=1).to(torch.float)
        out = alpha @ hidden @ self.W_1 + self.b_1
        out = self.relu(out)
        out = self.dropout(out)
        out = alpha @ out @ self.W_2 + self.b_2
        out = self.relu(out)
        out = self.dropout(out)
        # out = torch.mean(out, dim=0)    #   REDUCE SIZE ?
        return out


"""## TDU"""


class AttentionHead(nn.Module):
    def __init__(self, num_features, kvdim):
        super(AttentionHead, self).__init__()
        self.num_features = num_features

        # init num features in each q, k, v matrix
        self.kvdim = kvdim

        self.W_q = nn.Parameter(torch.empty(size=(self.num_features, self.kvdim)))
        nn.init.xavier_uniform_(self.W_q.data, gain=1)

        self.W_k = nn.Parameter(torch.empty(size=(self.num_features, self.kvdim)))
        nn.init.xavier_uniform_(self.W_k.data, gain=1)

        self.W_v = nn.Parameter(torch.empty(size=(self.num_features, self.kvdim)))
        nn.init.xavier_uniform_(self.W_v.data, gain=1)

    def forward(self, Ht, Ft_1, ret_attn=False):
        """Feed forward layer of MHA. Outputs weighted averages of attention
        performed using Ht as queries and Ft_1 as keys and values
        Parameters
        ----------
        Ht |ct| x d tensor
            Current output of the GAM module
        Ft_1 |ct_1| x d tensor
            Previous hidden state output of the TDU module
        Returns
        -------
        |ct| x d tensor
        Result of performing multi-head attention
        """

        queries = Ht @ self.W_q
        keys = Ft_1 @ self.W_k
        values = Ft_1 @ self.W_v

        def perform_attention(query, key, value, ret_attn):
            attn_scores = query @ key.T / np.sqrt(self.kvdim)
            softmax_attn_scores = softmax(attn_scores, dim=-1)
            if ret_attn:
                return softmax_attn_scores @ value, softmax_attn_scores
            return softmax_attn_scores @ value

        return perform_attention(queries, keys, values, ret_attn)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_features, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.kvdim = num_features // num_heads
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(self.num_features, self.num_features, bias=False)
        self.W_k = nn.Linear(self.num_features, self.num_features, bias=False)
        self.W_v = nn.Linear(self.num_features, self.num_features, bias=False)
        self.W_o = nn.Linear(self.num_features, self.num_features, bias=False)

    def forward(self, input_embeddings, history_hidden):

        queries = self.W_q(input_embeddings)
        keys = self.W_k(history_hidden)
        values = self.W_v(history_hidden)

        queries_split = torch.split(queries, self.kvdim, dim=1)
        keys_split = torch.split(keys, self.kvdim, dim=1)
        values_split = torch.split(values, self.kvdim, dim=1)

        def perform_attention(query, key, value):
            attn_scores = query @ key.T / np.sqrt(self.kvdim)
            softmax_attn_scores = softmax(attn_scores, dim=-1)
            return softmax_attn_scores @ value

        head_scores = torch.cat([perform_attention(
                                    queries_split[head],
                                    keys_split[head],
                                    values_split[head])
                                for head in range(self.num_heads)], dim=1)
        head_scores = self.dropout(head_scores)
        return self.W_o(head_scores)


class TDU(nn.Module):
    def __init__(self, num_features, num_heads, dropout, num_events):
        super(TDU, self).__init__()
        self.num_features = num_features  # should be d_model
        self.num_heads = num_heads  # desired number of attention heads
        self.num_events = num_events

        # initializing the Rt equation w xavier uniform params
        self.R = nn.Sequential(
            nn.Linear(num_features, num_features, bias=True),
            nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.R[0].weight, gain=1)

        # initializing the Zt equation w xavier uniform params
        self.Z = nn.Sequential(
            nn.Linear(num_features, num_features, bias=True),
            nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.Z[0].weight, gain=1)

        # initializing the F_tilde equation w xavier uniform params
        self.F_tilde = nn.Sequential(
            nn.Linear(num_features, num_features, bias=True),
            nn.Tanh()
        )
        nn.init.xavier_uniform_(self.F_tilde[0].weight, gain=5 / 3.)

        self.multihead_attn = MultiHeadAttention(num_features, num_heads, \
                                                 dropout)

        # self.multihead_attn = MultiHeadAttention(num_heads, num_features)
        # self.GI = nn.Parameter(torch.ones(num_events, 1) * .5)
        self.GI = nn.Parameter(torch.ones(num_events, num_features) * .5)

    def forward(self, Ht, Ft_1, F_hat_t_1, code_rows):
        """Feed forward layer of the TDU. Creates F0, F hat 0.

        Parameters
        ----------
        Ht |ct| x d tensor
            input embeddings output by the GAM
        Ft_1 |ct_1| x d tensor
            previous timestep TDU output
        F_hat_t_1 |c| x d tensor
            full history, all event code TDU node representation
        code_rows |ct|, tensor
            for each of the current embeddings (Ht) \
            gives its index in F_hat_t_1

        Returns
        -------
        F0, Fhat0 |ct| x d tensor, |c| x d tensor
            Two all zero tensors of size x num_features
        """
        # GsFt_1 = self.multihead_attn(Ht, Ft_1)
        # GsFt_1_Ht = GsFt_1 + Ht

        GsFt_1 = Ht
        GsFt_1_Ht = GsFt_1

        R_t = self.R(GsFt_1_Ht)
        Z_t = self.Z(GsFt_1_Ht)
        F_tilde_t = self.F_tilde(R_t * GsFt_1 + Ht)
        F_t = (1 - Z_t) * GsFt_1 + Z_t * F_tilde_t
        F_hat_t = to_device(torch.zeros(self.num_events, self.num_features))
        for indx, row in enumerate(code_rows):
            F_hat_t[row] = self.GI[row] * F_hat_t_1[row] + (1 - self.GI[row]) * F_t[indx]
        return F_hat_t, F_t

    def initHidden(self, num_codes):
        """Creates F0, F hat 0.
        Parameters
        ----------
        num_codes int
            total number of clinical events

        Returns
        -------
        F0, Fhat0 tensor, tensor
            Two all zero tensors of num_codes x num_features
        """
        return torch.zeros(num_codes, self.num_features), \
               torch.zeros(num_codes, self.num_features)


""" Classifier"""


class MIML(nn.Module):
    def __init__(self, num_features_concat, num_drugs, agg, threshold):
        super(MIML, self).__init__()
        self.num_features_concat = num_features_concat
        self.num_drugs = num_drugs

        self.fc = nn.Linear(num_features_concat, num_drugs)
        nn.init.xavier_uniform_(self.fc.weight, gain=np.sqrt(2))
        # self.elu = nn.ELU(alpha=alpha)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.thresh = to_device(torch.tensor(threshold, dtype=torch.float, requires_grad=True))
        self.threshold = nn.Threshold(threshold=threshold, value=0)
        self.agg = agg
        self.gelu = nn.GELU()
        # self.swish = nn.Hardswish()

    def forward(self, Ot):
        out = self.fc(Ot)
        # out = Ot
        if self.agg == 'sum':
            Ot = torch.sum(self.gelu(out), dim=0)
        if self.agg == 'mean':
            return torch.mean(self.gelu(out), dim=1)
        if self.agg == 'max':
            return torch.max(self.gelu(out), dim=1)[0]
        return Ot


""" FULL GATE  """


class GATE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=64, gam_in=64, gam_hidden=128,
                 gam_out=64, gam_dropout=0.38, tdu_features=64, tdu_heads=1, tdu_dropout=0.061,
                 num_features_concat=128, num_drugs=150, agg='sum', threshold=0.2):
        super(GATE, self).__init__()
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.gam = GAM(gam_in, gam_hidden, gam_out, gam_dropout)
        self.tdu = TDU(tdu_features, tdu_heads, tdu_dropout, num_embeddings)
        self.miml = MIML(num_features_concat, num_drugs, agg, threshold)
        self.sigmoid = nn.Sigmoid()

    def forward(self, admissions):
        Ft_1, F_hat_t_1 = self.tdu.initHidden(self.num_embeddings)
        Ft_1, F_hat_t_1 = to_device(Ft_1), to_device(F_hat_t_1)
        for timestep, admission in enumerate(admissions):
            if timestep != len(admissions) - 1:
                adj, code_indices = to_device(admission[0]), to_device(admission[1])    #.to(cuda), H0[1].to(cuda)
            else:
                adj, code_indices = to_device(admission[2]), to_device(admission[3])    #.to(cuda), H0[3].to(cuda)
            gam_in = self.embeddings(code_indices)
            gam_output = self.gam(gam_in, adj)
            F_hat_t, Ft = self.tdu(gam_output, Ft_1, F_hat_t_1, code_indices)
            F_hat_t_1, Ft_1 = F_hat_t, Ft
        Ot = torch.cat([gam_output, F_hat_t_1[code_indices]], dim=1)
        Y_hat = self.miml(Ot)
        sigged = self.sigmoid(Y_hat)
        pred = (sigged > self.miml.thresh).float()
        return pred, sigged, Y_hat


class GATE2(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, gam_in, gam_hidden, gam_out,
                 gam_dropout, tdu_features, tdu_heads, tdu_dropout, tdu_events,
                 num_features_concat, num_drugs, agg, threshold):
        super(GATE2, self).__init__()
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.gam = GAM(gam_in, gam_hidden, gam_out, gam_dropout)
        self.tdu = TDU(tdu_features, tdu_heads, tdu_dropout, tdu_events)
        self.miml = MIML(num_features_concat, num_drugs, agg, threshold)
        self.sigmoid = nn.Sigmoid()

    def forward(self, admissions):
        Ft_1, F_hat_t_1 = self.tdu.initHidden(self.num_embeddings)
        Ft_1, F_hat_t_1 = to_device(Ft_1), to_device(F_hat_t_1)
        for i, admission in enumerate(admissions):
            code_indices, adj = admission
            code_indices, adj = to_device(code_indices), to_device(adj)
            gam_in = self.embeddings(code_indices)
            gam_output = self.gam(gam_in, adj)
            F_hat_t, Ft = self.tdu(gam_output, Ft_1, F_hat_t_1, code_indices)
            F_hat_t_1, Ft_1 = F_hat_t, Ft
        Ot = torch.cat([gam_output, F_hat_t_1[code_indices]], dim=2)
        Y_hat = self.miml(Ot)
        sigged = self.sigmoid(Y_hat)
        pred = (sigged > self.miml.thresh).float()
        return pred, sigged, Y_hat
