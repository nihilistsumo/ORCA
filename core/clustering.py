import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sentence_transformers import models, SentenceTransformer
from sklearn.cluster import KMeans, kmeans_plusplus, AgglomerativeClustering
import numpy as np
import random
random.seed(42)


def put_features_in_device(input_features, device):
    for key in input_features.keys():
        if isinstance(input_features[key], Tensor):
            input_features[key] = input_features[key].to(device)


# Both gt and a are n x k matrix
def get_nmi_loss(a, c, para_labels, device, from_attn=True):
    n = len(para_labels)
    unique_labels = list(set(para_labels))
    k = len(unique_labels)
    gt = torch.zeros(n, k, device=device)
    for i in range(n):
        gt[i][unique_labels.index(para_labels[i])] = 1.0
    G = torch.sum(gt, dim=0)
    C = torch.sum(a, dim=0)
    U = torch.matmul(gt.T, a)
    n = torch.sum(a)
    GxC = torch.outer(G, C)
    mi = torch.sum((U / n) * torch.log(n * U / GxC))
    nmi = 2 * mi / (-torch.sum(G * torch.log(G / n) / n) - torch.sum(C * torch.log(C / n) / n))
    return -nmi


def get_weighted_adj_rand_loss(a, c, para_labels, device, from_attn=True):
    n = len(para_labels)
    unique_labels = list(set(para_labels))
    gt = torch.zeros(n, n, device=device)
    gt_weights = torch.ones(n, n, device=device)
    para_label_freq = {k: para_labels.count(k) for k in unique_labels}
    for i in range(n):
        for j in range(n):
            if para_labels[i] == para_labels[j]:
                gt[i][j] = 1.0
                gt_weights[i][j] = para_label_freq[para_labels[i]]
    if from_attn:
        dist_mat = torch.cdist(a, a)
    else:
        dist_mat = torch.cdist(torch.matmul(a, c), torch.matmul(a, c))
    sim_mat = 1 / (1 + dist_mat)
    loss = torch.sum(((gt - sim_mat) ** 2) * gt_weights) / gt.shape[0]
    #loss = torch.sum(gt * dist_mat) / gt.shape[0] ** 2 - torch.sum((1 - gt) * dist_mat) / gt.shape[0] ** 2
    return loss


def get_adj_rand_loss(a, c, para_labels, device, from_attn=True):
    n = len(para_labels)
    unique_labels = list(set(para_labels))
    GT = torch.zeros(n, len(unique_labels), device=device)
    for i in range(n):
        k = unique_labels.index(para_labels[i])
        GT[i][k] = 1.0
    M = torch.matmul(GT.T, a)
    g = torch.sum(GT, dim=0)
    c = torch.sum(a, dim=0)
    rand = 0.5 * torch.sum(M**2 - M)
    expected = 0.5 * torch.sum(g**2 - g) * torch.sum(c**2 - c) / (n**2 - n)
    maximum = 0.25 * (torch.sum(g**2 - g) + torch.sum(c**2 - c))
    ari = (rand - expected) / (maximum - expected)
    loss = -ari
    return loss


def get_rand_loss(a, c, para_labels, device, from_attn=True):
    n = len(para_labels)
    unique_labels = list(set(para_labels))
    gt = torch.zeros(n, n, device=device)
    for i in range(n):
        for j in range(n):
            if para_labels[i] == para_labels[j]:
                gt[i][j] = 1.0
    if from_attn:
        dist_mat = torch.cdist(a, a)
    else:
        dist_mat = torch.cdist(torch.matmul(a, c), torch.matmul(a, c))
    sim_mat = 2 / (1 + torch.exp(dist_mat))
    loss = torch.sum((gt - sim_mat) ** 2) / n ** 2
    # loss = torch.sum(gt * dist_mat) / gt.shape[0] ** 2 - torch.sum((1 - gt) * dist_mat) / gt.shape[0] ** 2
    return loss


class DKM(nn.Module):

    def __init__(self, temp=0.5, threshold=0.0001, max_iter=100, eps=1e-6):
        super(DKM, self).__init__()
        self.temp = temp
        self.threshold = threshold
        self.max_iter = max_iter
        self.softmax = nn.Softmax(dim=1)
        self.cos = nn.CosineSimilarity()
        self.eps = eps

    def cosine_sim(self, x, y):
        return self.cos(x.repeat_interleave(y.shape[0], dim=0), y.repeat((x.shape[0], 1))).reshape(x.shape[0], y.shape[0])

    def forward(self, X, C_init):
        self.emb_dim = X.shape[1]
        self.C = C_init
        self.d = -torch.cdist(X, C_init, p=2.0)
        #self.d = self.cosine_sim(X, C_init)
        self.a = self.softmax(self.d/self.temp)
        self.a_sum = torch.sum(self.a, dim=0) + self.eps
        self.C_new = torch.matmul(self.a.T, X)/self.a_sum.repeat((self.emb_dim, 1)).T
        diff = torch.norm(self.C_new - self.C, p=1).item()
        i = 0
        while diff > self.threshold and i < self.max_iter:
            self.C = self.C_new
            self.d = -torch.cdist(X, self.C, p=2.0)
            #self.d = self.cosine_sim(X, C_init)
            self.a = self.softmax(self.d / self.temp)
            self.a_sum = torch.sum(self.a, dim=0) + self.eps
            self.C_new = torch.matmul(self.a.T, X) / self.a_sum.repeat((self.emb_dim, 1)).T
            diff = torch.norm(self.C_new - self.C, p=1).item()
            i += 1
        return self.C, self.a


class DKM_param(nn.Module):

    def __init__(self, temp=0.05, threshold=0.0001, max_iter=100, eps=1e-6, emb_dim=768, l1_size=512):
        super(DKM_param, self).__init__()
        self.temp = temp
        self.threshold = threshold
        self.max_iter = max_iter
        self.emb_dim = emb_dim
        self.softmax = nn.Softmax(dim=1)
        self.l1_size = l1_size
        self.fc1 = nn.Linear(5 * self.emb_dim, self.l1_size)
        self.fc2 = nn.Linear(self.l1_size, 1)
        self.act = nn.ReLU()
        self.final_act = nn.Sigmoid()
        self.eps = eps

    def learned_dist(self, q, x, y):
        x_tr = x.repeat_interleave(y.shape[0], dim=0)
        y_tr = y.repeat((x.shape[0], 1))
        qxy_tr = torch.hstack((x_tr, y_tr, torch.abs(x_tr - y_tr), torch.abs(x_tr - q), torch.abs(y_tr - q)))
        d = self.final_act(self.fc2(self.act(self.fc1(qxy_tr)))).reshape(x.shape[0], y.shape[0])
        return d

    def forward(self, q, X, C_init):
        self.emb_dim = X.shape[1]
        self.C = C_init
        # self.d = -torch.cdist(X, C_init, p=2.0)
        self.d = -self.learned_dist(q, X, C_init)
        # self.a = self.softmax(self.cosine_sim(X, C_init) / self.temp)
        self.a = self.softmax(self.d / self.temp)
        self.a_sum = torch.sum(self.a, dim=0) + self.eps
        self.C_new = torch.matmul(self.a.T, X) / self.a_sum.repeat((self.emb_dim, 1)).T
        diff = torch.norm(self.C_new - self.C, p=1).item()
        i = 0
        while diff > self.threshold and i < self.max_iter:
            self.C = self.C_new
            # self.d = -torch.cdist(X, self.C, p=2.0)
            self.d = -self.learned_dist(q, X, self.C)
            # self.a = self.softmax(self.cosine_sim(X, self.C) / self.temp)
            self.a = self.softmax(self.d / self.temp)
            self.a_sum = torch.sum(self.a, dim=0) + self.eps
            self.C_new = torch.matmul(self.a.T, X) / self.a_sum.repeat((self.emb_dim, 1)).T
            diff = torch.norm(self.C_new - self.C, p=1).item()
            i += 1
        return self.C, self.a


class QuerySpecificDKM(nn.Module):
    def __init__(self, trans_model_name, emb_dim, device, max_len):
        super(QuerySpecificDKM, self).__init__()
        trans_model = models.Transformer(trans_model_name, max_seq_length=max_len)
        pool_model = models.Pooling(trans_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pool_model.get_sentence_embedding_dimension(), out_features=emb_dim,
                                   activation_function=nn.Tanh())
        self.emb_model = SentenceTransformer(modules=[trans_model, pool_model]).to(device)
        self.device = device
        self.dkm = DKM_param(emb_dim=self.emb_model.get_sentence_embedding_dimension())
        self.dkm.to(device)

    def forward(self, query_content, texts, k, learn_emb=False):
        n = len(texts)
        if learn_emb:
            query_fet = self.emb_model.tokenize([(query_content, '')])
            psg_fet = self.emb_model.tokenize([(query_content, t) for t in texts])
            put_features_in_device(query_fet, self.device)
            put_features_in_device(psg_fet, self.device)
            self.query_vec = self.emb_model(query_fet)['sentence_embedding']
            self.emb_vecs = self.emb_model(psg_fet)['sentence_embedding']
        else:
            self.query_vec = torch.tensor(self.emb_model.encode([query_content]), device=self.device)
            self.emb_vecs = torch.tensor(self.emb_model.encode(texts), device=self.device)
        init_indices = random.sample(range(n), k)
        init_c = self.emb_vecs[init_indices]
        self.C, self.a = self.dkm(self.query_vec, self.emb_vecs, init_c)
        return self.C, self.a

    def get_clustering(self, query_content, texts, k):
        n = len(texts)
        self.query_vec = torch.tensor(self.emb_model.encode([query_content]), device=self.device)
        self.emb_vecs = torch.tensor(self.emb_model.encode(texts), device=self.device)
        init_indices = random.sample(range(n), k)
        init_c = self.emb_vecs[init_indices]
        self.C, self.a = self.dkm(self.query_vec, self.emb_vecs, init_c)
        pred_labels = torch.argmax(self.a, dim=1).detach().cpu().numpy()
        return pred_labels


class QuerySpecificClusteringModel(nn.Module):
    def __init__(self, trans_model_name, emb_dim, device, max_len, kmeans_plus=False):
        super(QuerySpecificClusteringModel, self).__init__()
        self.device = device
        emb_model = models.Transformer(trans_model_name, max_seq_length=max_len)
        pool_model = models.Pooling(emb_model.get_word_embedding_dimension())
        if emb_dim is not None:
            dense_model = models.Dense(in_features=pool_model.get_sentence_embedding_dimension(), out_features=emb_dim, activation_function=nn.Tanh())
        self.qp_model = SentenceTransformer(modules=[emb_model, pool_model]).to(device)
        self.dkm = DKM()
        self.use_kmeans_plus = kmeans_plus

    def forward(self, input_features, k):
        self.qp = self.qp_model(input_features)['sentence_embedding']
        if self.use_kmeans_plus:
            qp = self.qp.detach().clone().cpu().numpy()
            init_c, _ = kmeans_plusplus(qp, k)
            init_c = torch.tensor(init_c, dtype=torch.float32, device=self.device)
        else:
            init_c = self.qp[random.sample(range(self.qp.shape[0]), k)].detach().clone()
        self.C, self.a = self.dkm(self.qp, init_c)
        return self.C, self.a

    def get_embedding(self, input_features):
        self.qp = self.qp_model(input_features)['sentence_embedding']
        return self.qp

    def get_clustering(self, embeddings, k, debug_switch=False):
        #self.qp = self.qp_model(input_features)['sentence_embedding']
        if self.use_kmeans_plus:
            embeddings = embeddings.detach().clone().cpu().numpy()
            init_c, _ = kmeans_plusplus(embeddings, k)
            init_c = torch.tensor(init_c, dtype=torch.float32, device=self.device)
        else:
            init_c = embeddings[random.sample(range(embeddings.shape[0]), k)].detach().clone()
        self.C, self.a = self.dkm(embeddings, init_c)
        if debug_switch:
            c_np = self.C.clone().cpu().numpy()
            a_np = self.a.clone().cpu().numpy()
            init_c_np = init_c.clone().cpu().numpy()
            embeddings_np = embeddings.clone().cpu().numpy()
            if torch.std(self.a).item() < 0.01:
                print('Low std in attention matrix')
        pred_labels = torch.argmax(self.a, dim=1).detach().cpu().numpy()
        return pred_labels


class QuerySpecificHACModel(nn.Module):
    def __init__(self, trans_model_name, emb_dim, device, max_len):
        super(QuerySpecificHACModel, self).__init__()
        self.device = device
        emb_model = models.Transformer(trans_model_name, max_seq_length=max_len)
        pool_model = models.Pooling(emb_model.get_word_embedding_dimension())
        if emb_dim is not None:
            dense_model = models.Dense(in_features=pool_model.get_sentence_embedding_dimension(), out_features=emb_dim, activation_function=nn.Tanh())
        self.qp_model = SentenceTransformer(modules=[emb_model, pool_model]).to(device)

    def euclid_dist(self, x):
        dist_mat = torch.norm(x[:, None] - x, dim=2, p=2)
        return dist_mat

    def forward(self, input_features):
        self.qp = self.qp_model(input_features)['sentence_embedding']
        self.qp_dist_mat = self.euclid_dist(self.qp)
        return self.qp_dist_mat

    def get_embedding(self, input_features):
        self.qp = self.qp_model(input_features)['sentence_embedding']
        return self.qp

    def get_clustering(self, embeddings, k):
        #self.qp = self.qp_model(input_features)['sentence_embedding']
        dist_mat = self.euclid_dist(embeddings)
        cl = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
        pred_labels = cl.fit_predict(dist_mat.detach().cpu().numpy())
        return pred_labels


class QS3M_HACModel(nn.Module):
    def __init__(self, trans_model_name, emb_dim, device, max_len, train_emb=False):
        super(QS3M_HACModel, self).__init__()
        self.device = device
        self.train_emb = train_emb
        emb_model = models.Transformer(trans_model_name, max_seq_length=max_len)
        pool_model = models.Pooling(emb_model.get_word_embedding_dimension())
        if emb_dim is not None:
            dense_model = models.Dense(in_features=pool_model.get_sentence_embedding_dimension(), out_features=emb_dim, activation_function=nn.Tanh())
        self.qp_model = SentenceTransformer(modules=[emb_model, pool_model]).to(device)
        self.LL1 = nn.Linear(self.qp_model.get_sentence_embedding_dimension(), self.qp_model.get_sentence_embedding_dimension())
        self.LL2 = nn.Linear(self.qp_model.get_sentence_embedding_dimension(), self.qp_model.get_sentence_embedding_dimension())
        self.LL3 = nn.Linear(5 * self.qp_model.get_sentence_embedding_dimension(), 1)
        self.act = nn.ReLU()
        self.final_act = nn.Sigmoid()

    def get_similarity_mat(self, query_vec, psg_vecs):
        n = psg_vecs.shape[0]
        q_rep = query_vec.tile(n ** 2, 1)
        p1_rep = psg_vecs.repeat_interleave(n, 0)
        p2_rep = psg_vecs.tile(n, 1)
        q_rep_tr = self.act(self.LL2(self.act(self.LL1(q_rep))))
        p1_rep_tr = self.act(self.LL2(self.act(self.LL1(p1_rep))))
        p2_rep_tr = self.act(self.LL2(self.act(self.LL1(p2_rep))))
        pd = torch.abs(p1_rep_tr - p2_rep_tr)
        p1q = torch.abs(p1_rep_tr - q_rep_tr)
        p2q = torch.abs(p2_rep_tr - q_rep_tr)
        z = torch.hstack((p1_rep_tr, p2_rep_tr, pd, p1q, p2q))
        sim_scores = self.final_act(self.LL3(z)).reshape(n, n)
        return sim_scores

    def forward(self, query, passages):
        if self.train_emb:
            query_fet = self.qp_model.tokenize([(query, "")])
            psg_fet = self.qp_model.tokenize([(query, p) for p in passages])
            put_features_in_device(query_fet, self.device)
            put_features_in_device(psg_fet, self.device)
            query_vec = self.qp_model(query_fet)['sentence_embedding']
            psg_vecs = self.qp_model(psg_fet)['sentence_embedding']
        else:
            query_vec = self.qp_model.encode([query], convert_to_tensor=True)
            psg_vecs = self.qp_model.encode(passages, convert_to_tensor=True)
        sim_scores = self.get_similarity_mat(query_vec, psg_vecs)
        return sim_scores

    def get_clustering(self, query, passages, k):
        q_vec = self.qp_model.encode([(query, "")], convert_to_tensor=True)
        psg_vecs = self.qp_model.encode([(query, p) for p in passages], convert_to_tensor=True)
        sim_scores = self.get_similarity_mat(q_vec, psg_vecs)
        dist_mat = 1 - sim_scores
        cl = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
        pred_labels = cl.fit_predict(dist_mat.detach().cpu().numpy())
        return pred_labels



class QuerySpecificAttentionClusteringModel(nn.Module):

    def __init__(self, trans_model_name, emb_dim, attn_dim, num_attn_head, device, max_len, kmeans_plus=False):
        """
        :param trans_model_name: The sbert model name or a pre-trained sbert model.
        :param emb_dim: Embedding dimension; if a pre-trained sbert model is given, this parameter will get the correct
        value from the model. If it is None, then no dense layer will be added to the sbert model and it will get the
        value of the pooling layer of the sbert model. Else, a dense layer with emb_dim output shape will be added.
        :param attn_dim: Dimension of the self-attention module. If None is given, then it will take the same value as
        emb_dim.
        :param num_attn_head: No. of attention heads.
        :param device: CUDA device.
        :param max_len: Max length of tokens accepted by the sbert model.
        :param kmeans_plus: Whether to use k-means++ for initialization of the DKM module.
        """
        super(QuerySpecificAttentionClusteringModel, self).__init__()
        self.device = device
        if isinstance(trans_model_name, nn.Module):
            self.qp_model = trans_model_name
            self.emb_dim = self.qp_model.get_sentence_embedding_dimension()
        else:
            emb_model = models.Transformer(trans_model_name, max_seq_length=max_len)
            pool_model = models.Pooling(emb_model.get_word_embedding_dimension())
            if emb_dim is not None:
                dense_model = models.Dense(in_features=pool_model.get_sentence_embedding_dimension(), out_features=emb_dim,
                                       activation_function=nn.Tanh())
                self.emb_dim = emb_dim
            else:
                self.emb_dim = emb_model.get_word_embedding_dimension()
            self.qp_model = SentenceTransformer(modules=[emb_model, pool_model])
        if attn_dim is not None:
            self.attn_dim = attn_dim
        else:
            self.attn_dim = self.emb_dim
        self.num_attn_head = num_attn_head
        self.self_attn = torch.nn.MultiheadAttention(self.attn_dim, self.num_attn_head, batch_first=True)
        self.fc1 = nn.Linear(self.emb_dim, self.attn_dim)
        self.fc2 = nn.Linear(self.attn_dim, self.emb_dim)
        self.act = nn.ReLU()
        self.dkm = DKM()
        self.layer_norm = nn.LayerNorm(2*self.emb_dim)
        self.use_kmeans_plus = kmeans_plus

    def forward(self, input_features, k_cl):
        self.qp_orig = self.qp_model(input_features)['sentence_embedding']
        self.qp = self.act(self.fc1(self.qp_orig))
        self.qp_tr, self.attn_wt = self.self_attn(self.qp.unsqueeze(0), self.qp.unsqueeze(0), self.qp.unsqueeze(0))
        self.qp_tr = torch.squeeze(self.qp_tr, 0)
        self.qp_tr = self.act(self.fc2(self.qp_tr))
        #self.qp_tr = self.layer_norm(self.qp_orig + self.qp_tr)
        self.qp_tr = self.layer_norm(torch.hstack((self.qp_orig, self.qp_tr)))
        if self.use_kmeans_plus:
            qp_tr = self.qp_tr.detach().clone().cpu().numpy()
            init_c, _ = kmeans_plusplus(qp_tr, k_cl)
            init_c = torch.tensor(init_c, dtype=torch.float32, device=self.device)
        else:
            init_c = self.qp_tr[random.sample(range(self.qp_tr.shape[0]), k_cl)].detach().clone()
        self.C, self.a = self.dkm(self.qp_tr, init_c)
        return self.C, self.a

    def get_embedding(self, input_texts):
        self.qp_orig = self.qp_model.encode(input_texts, convert_to_tensor=True)
        self.qp = self.act(self.fc1(self.qp_orig))
        #self.qp = self.qp_model.encode(input_texts, convert_to_tensor=True)
        self.qp_tr, self.attn_wt = self.self_attn(self.qp.unsqueeze(0), self.qp.unsqueeze(0), self.qp.unsqueeze(0))
        self.qp_tr = torch.squeeze(self.qp_tr, 0)
        self.qp_tr = self.act(self.fc2(self.qp_tr))
        #self.qp_tr = self.layer_norm(self.qp_orig + self.qp_tr)
        self.qp_tr = self.layer_norm(torch.hstack((self.qp_orig, self.qp_tr)))
        return self.qp_tr

    def get_embedding_without_attn(self, input_texts):
        self.qp = self.qp_model.encode(input_texts, convert_to_tensor=True)
        return self.qp

    def get_clustering(self, embeddings_orig, k_cl, debug_switch=False):
        #self.qp = self.qp_model(input_features)['sentence_embedding']
        embeddings = self.act(self.fc1(embeddings_orig))
        embeddings_tr, attn_wt = self.self_attn(embeddings.unsqueeze(0), embeddings.unsqueeze(0), embeddings.unsqueeze(0))
        embeddings_tr = torch.squeeze(embeddings_tr, 0)
        embeddings_tr = self.act(self.fc2(embeddings_tr))
        #embeddings_tr = self.layer_norm(embeddings_orig + embeddings_tr)
        embeddings_tr = self.layer_norm(torch.hstack((embeddings_orig, embeddings_tr)))
        if self.use_kmeans_plus:
            embeddings_tr = embeddings_tr.detach().clone().cpu().numpy()
            init_c, _ = kmeans_plusplus(embeddings_tr, k_cl)
            init_c = torch.tensor(init_c, dtype=torch.float32, device=self.device)
        else:
            init_c = embeddings_tr[random.sample(range(embeddings_tr.shape[0]), k_cl)].detach().clone()
        self.C, self.a = self.dkm(embeddings_tr, init_c)
        if debug_switch:
            c_np = self.C.clone().cpu().numpy()
            a_np = self.a.clone().cpu().numpy()
            init_c_np = init_c.clone().cpu().numpy()
            embeddings_np = embeddings_tr.clone().cpu().numpy()
            if torch.std(self.a).item() < 0.01:
                print('Low std in attention matrix')
        pred_labels = torch.argmax(self.a, dim=1).detach().cpu().numpy()
        return pred_labels


class QuerySpecificAttentionFixedEmbedClusteringModel(nn.Module):
    def __init__(self, emb_dim, attn_dim, num_attn_head, device, kmeans_plus=False, debug_mode=False):
        super(QuerySpecificAttentionFixedEmbedClusteringModel, self).__init__()
        assert attn_dim % num_attn_head == 0
        self.alpha = 0.5
        self.device = device
        self.emb_dim = emb_dim
        self.attn_dim = attn_dim
        self.num_attn_head = num_attn_head
        self.self_attn = torch.nn.MultiheadAttention(self.attn_dim, self.num_attn_head, batch_first=True,
                                                     dropout=0.1).to(device)
        self.fc1 = nn.Linear(self.emb_dim, self.attn_dim)
        self.fc2 = nn.Linear(self.attn_dim, self.emb_dim)
        self.act = nn.ReLU()
        self.dkm = DKM()
        self.layer_norm = nn.LayerNorm(self.emb_dim).to(device)
        self.use_kmeans_plus = kmeans_plus
        self.debug = debug_mode

    def forward(self, original_embeddings, k_cl):
        input_embeddings = self.act(self.fc1(original_embeddings))
        self.input_emb_tr, self.attn_wt = self.self_attn(input_embeddings.unsqueeze(0), input_embeddings.unsqueeze(0),
                                                         input_embeddings.unsqueeze(0))
        self.input_emb_tr = torch.squeeze(self.input_emb_tr, 0)
        self.input_emb_tr = self.act(self.fc2(self.input_emb_tr))
        self.input_emb_tr = self.layer_norm((1 - self.alpha) * original_embeddings + self.alpha * self.input_emb_tr)
        if self.debug:
            attn_wt_np = self.attn_wt.detach().clone().cpu().numpy()
        if self.use_kmeans_plus:
            input_emb_tr = self.input_emb_tr.detach().clone().cpu().numpy()
            init_c, _ = kmeans_plusplus(input_emb_tr, k_cl)
            init_c = torch.tensor(init_c, dtype=torch.float32, device=self.device)
        else:
            init_c = self.input_emb_tr[random.sample(range(self.input_emb_tr.shape[0]), k_cl)].detach().clone()
        self.C, self.a = self.dkm(self.input_emb_tr, init_c)
        return self.C, self.a

    def get_transformed_embedding(self, original_embeddings):
        input_embeddings = self.act(self.fc1(original_embeddings))
        self.input_emb_tr, self.attn_wt = self.self_attn(input_embeddings.unsqueeze(0), input_embeddings.unsqueeze(0),
                                                         input_embeddings.unsqueeze(0))
        self.input_emb_tr = torch.squeeze(self.input_emb_tr, 0)
        self.input_emb_tr = self.act(self.fc2(self.input_emb_tr))
        self.input_emb_tr = self.layer_norm((1 - self.alpha) * original_embeddings + self.alpha * self.input_emb_tr)
        return self.input_emb_tr

    def get_clustering(self, original_embeddings, k_cl, debug_switch=False):
        embeddings = self.act(self.fc1(original_embeddings))
        embeddings_tr, attn_wt = self.self_attn(embeddings.unsqueeze(0), embeddings.unsqueeze(0), embeddings.unsqueeze(0))
        embeddings_tr = torch.squeeze(embeddings_tr, 0)
        embeddings_tr = self.act(self.fc2(embeddings_tr))
        embeddings_tr = self.layer_norm((1 - self.alpha) * original_embeddings + self.alpha * embeddings_tr)
        if self.use_kmeans_plus:
            embeddings_tr = embeddings_tr.detach().clone().cpu().numpy()
            init_c, _ = kmeans_plusplus(embeddings_tr, k_cl)
            init_c = torch.tensor(init_c, dtype=torch.float32, device=self.device)
        else:
            init_c = embeddings_tr[random.sample(range(embeddings_tr.shape[0]), k_cl)].detach().clone()
        self.C, self.a = self.dkm(embeddings_tr, init_c)
        if debug_switch:
            c_np = self.C.clone().cpu().numpy()
            a_np = self.a.clone().cpu().numpy()
            init_c_np = init_c.clone().cpu().numpy()
            embeddings_np = embeddings_tr.clone().cpu().numpy()
            if torch.std(self.a).item() < 0.01:
                print('Low std in attention matrix')
        pred_labels = torch.argmax(self.a, dim=1).detach().cpu().numpy()
        return pred_labels


class QuerySpecificClusteringModelWithSection(nn.Module):
    def __init__(self, trans_model_name, emb_dim, device, max_len):
        super(QuerySpecificClusteringModelWithSection, self).__init__()
        emb_model = models.Transformer(trans_model_name, max_seq_length=max_len)
        pool_model = models.Pooling(emb_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pool_model.get_sentence_embedding_dimension(), out_features=emb_dim, activation_function=nn.Tanh())
        self.qp_model = SentenceTransformer(modules=[emb_model, pool_model]).to(device)
        self.sec_model = SentenceTransformer(modules=[emb_model, pool_model]).to(device)
        self.dkm = DKM()

    def forward(self, input_features, section_texts, k):
        self.qp = self.qp_model(input_features)['sentence_embedding']
        self.qs = self.sec_model.encode(section_texts, convert_to_tensor=True)
        init_c = self.qp[random.sample(range(self.qp.shape[0]), k)].detach().clone()
        self.C, self.a = self.dkm(self.qp, init_c)
        return self.C, self.a, self.qp, self.qs

    def get_embedding(self, input_features):
        self.qp = self.qp_model(input_features)['sentence_embedding']
        return self.qp

    def get_clustering(self, embeddings, k):
        #self.qp = self.qp_model(input_features)['sentence_embedding']
        init_c = embeddings[random.sample(range(embeddings.shape[0]), k)].detach().clone()
        self.C, self.a = self.dkm(embeddings, init_c)
        pred_labels = torch.argmax(self.a, dim=1).detach().cpu().numpy()
        return pred_labels


class SBERTTripletLossModel(nn.Module):
    def __init__(self, trans_model_name, device, max_len, triplet_margin):
        super(SBERTTripletLossModel, self).__init__()
        emb_model = models.Transformer(trans_model_name, max_seq_length=max_len)
        pool_model = models.Pooling(emb_model.get_word_embedding_dimension())
        self.emb_model = SentenceTransformer(modules=[emb_model, pool_model]).to(device)
        self.triplet_margin = triplet_margin
        self.dkm = DKM()

    def forward(self, input_features_anchor, input_features_pos, input_features_neg):
        anchor_emb = self.emb_model(input_features_anchor)['sentence_embedding']
        pos_emb = self.emb_model(input_features_pos)['sentence_embedding']
        neg_emb = self.emb_model(input_features_neg)['sentence_embedding']
        dist_pos = F.pairwise_distance(anchor_emb, pos_emb, p=2)
        dist_neg = F.pairwise_distance(anchor_emb, neg_emb, p=2)
        losses = F.relu(dist_pos - dist_neg + self.triplet_margin)
        return losses.mean()

    def get_embedding(self, input_texts):
        qp = torch.from_numpy(self.emb_model.encode(input_texts))
        return qp

    def get_clustering(self, embeddings, k):
        #x = self.get_embedding(input_texts)
        init_c = embeddings[random.sample(range(embeddings.shape[0]), k)].detach().clone()
        self.C, self.a = self.dkm(embeddings, init_c)
        pred_labels = torch.argmax(self.a, dim=1).detach().cpu().numpy()
        return pred_labels


class QuerySpecificClusteringModelDual(nn.Module):
    def __init__(self, q_max_len, psg_max_len, trans_model_name, emb_dim, device):
        super(QuerySpecificClusteringModelDual, self).__init__()
        q_emb_model = models.Transformer(trans_model_name, max_seq_length=q_max_len)
        q_pool_model = models.Pooling(q_emb_model.get_word_embedding_dimension())
        q_dense_model = models.Dense(in_features=q_pool_model.get_sentence_embedding_dimension(), out_features=emb_dim,
                                     activation_function=nn.Sigmoid())
        self.q_model = SentenceTransformer(modules=[q_emb_model, q_pool_model, q_dense_model]).to(device)
        emb_model = models.Transformer(trans_model_name, max_seq_length=psg_max_len)
        pool_model = models.Pooling(emb_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pool_model.get_sentence_embedding_dimension(), out_features=emb_dim,
                                     activation_function=nn.Tanh())
        self.p_model = SentenceTransformer(modules=[emb_model, pool_model, dense_model]).to(device)
        self.dkm = DKM()

    def forward(self, q_features, psg_features, k):
        self.qw = self.q_model(q_features)['sentence_embedding']
        self.p = self.p_model(psg_features)['sentence_embedding']
        self.qp = self.qw * self.p
        self.C, self.a = self.dkm(self.qp, self.qp[:k])
        self.sim_mat = 1 / (1 + torch.cdist(self.a, self.a))
        return self.sim_mat