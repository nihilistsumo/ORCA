import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AdamW
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sentence_transformers import models, SentenceTransformer
import ir_measures
from ir_measures import MAP, Rprec, nDCG
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from util.data import get_article_qrels, get_page_sec_para_dict, TRECCAR_Datset
from util.data import put_features_in_device
from core.clustering import DKM, get_nmi_loss, get_weighted_adj_rand_loss, get_adj_rand_loss, get_rand_loss
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def prepare_data(art_qrels, qrels, paratext_tsv):
    page_paras = get_article_qrels(art_qrels)
    page_sec_paras = get_page_sec_para_dict(qrels)
    paratext = {}
    with open(paratext_tsv, 'r', encoding='utf-8') as f:
        for l in f:
            p = l.split('\t')[0]
            paratext[p] = l.split('\t')[1].strip()
    return page_paras, page_sec_paras, paratext


def eval_clustering(test_samples, model):
    model.eval()
    rand_dict, nmi_dict = {}, {}
    for s in test_samples:
        true_labels = s.para_labels
        k = len(set(true_labels))
        texts = s.para_texts
        query_content = s.q.split('enwiki:')[1].replace('%20', ' ')
        input_texts = [(query_content, t, '') for t in texts]
        embeddings = model.emb_model.encode(input_texts, convert_to_tensor=True)
        pred_labels = model.get_clustering(embeddings, k)
        rand = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        rand_dict[s.q] = rand
        nmi_dict[s.q] = nmi
    return rand_dict, nmi_dict


def eval_duo_sbert(duo_sbert, page_paras, page_sec_paras, paratext, qrels):
    duo_sbert.eval()
    with open('temp.val.run', 'w') as f:
        pages = list(page_sec_paras.keys())
        for p in tqdm(range(len(pages))):
            page = pages[p]
            cand_set = page_paras[page]
            n = len(cand_set)
            texts_for_pi = []
            for sec in page_sec_paras[page].keys():
                if '/' in sec:
                    query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
                else:
                    query = sec.replace('enwiki:', '').replace('%20', ' ')
                for i in range(len(cand_set)):
                    pi = cand_set[i]
                    for j in range(len(cand_set)):
                        if i == j:
                            continue
                        pj = cand_set[j]
                        texts_for_pi.append((query, paratext[pi], paratext[pj]))
            input_embs = duo_sbert.emb_model.encode(texts_for_pi, convert_to_tensor=True)
            num_q = len(page_sec_paras[page].keys())
            assert input_embs.shape[0] == num_q * n * (n-1)
            for q in range(num_q):
                rel_score = duo_sbert.get_preds(input_embs[q * n * (n-1): (q+1) * n * (n-1)], n).flatten()
                for i in range(len(cand_set)):
                    f.write(sec + ' 0 ' + cand_set[i] + ' 0 ' + str(rel_score[i]) + ' val_runid\n')
    qrels_dat = ir_measures.read_trec_qrels(qrels)
    run_dat = ir_measures.read_trec_run('temp.val.run')
    return ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)


def random_ranking(cand_set):
    n = len(cand_set)
    return torch.rand(n).tolist()


def bm25_ranking(sec, cand_set_texts):
    if '/' in sec:
        query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
    else:
        query = sec.replace('enwiki:', '').replace('%20', ' ')
    tokenized_query = query.split(' ')
    tokenized_corpus = [t.split(' ') for t in cand_set_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25.get_scores(tokenized_query)


def eval_mono_bert_bin_clustering_full(model, page_paras, page_sec_paras, paratext, qrels):
    model.eval()
    rand_dict, nmi_dict = {}, {}
    with open('temp.val.run', 'w') as f:
        pages = list(page_sec_paras.keys())
        for p in tqdm(range(len(pages))):
            page = pages[p]
            cand_set = page_paras[page]
            n = len(cand_set)
            for sec in page_sec_paras[page].keys():
                cand_set_texts = [paratext[p] for p in cand_set]
                pred_score = model.get_rank_scores(sec, cand_set_texts)
                for i in range(n):
                    f.write(sec + ' 0 ' + cand_set[i] + ' 0 ' + str(pred_score[i]) + ' val_runid\n')
                binary_cluster_labels = [1 if cand_set[i] in page_sec_paras[page][sec] else 0 for i in range(n)]
                pred_cluster_labels = model.get_binary_clustering(sec, cand_set_texts)
                rand = adjusted_rand_score(binary_cluster_labels, pred_cluster_labels)
                nmi = normalized_mutual_info_score(binary_cluster_labels, pred_cluster_labels)
                rand_dict[page + sec] = rand
                nmi_dict[page + sec] = nmi
    qrels_dat = ir_measures.read_trec_qrels(qrels)
    run_dat = ir_measures.read_trec_run('temp.val.run')
    rank_evals = ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)
    return rank_evals, rand_dict, nmi_dict


def eval_mono_bert_bin_clustering(model, samples):
    model.eval()
    rand_dict, nmi_dict = {}, {}
    with open('temp.val.qrels', 'w') as f:
        for s in samples:
            for i in range(len(s.paras)):
                f.write(s.para_labels[i] + ' 0 ' + s.paras[i] + ' 1\n')
    with open('temp.val.run', 'w') as f:
        for si in range(len(samples)):
            s = samples[si]
            cand_set = s.paras
            n = len(cand_set)
            sections = list(set(s.para_labels))
            for sec in sections:
                pred_score = model.get_rank_scores(sec, s.para_texts)
                for p in range(n):
                    f.write(sec + ' 0 ' + cand_set[p] + ' 0 ' + str(pred_score[p]) + ' val_runid\n')
                binary_cluster_labels = [1 if sec == s.para_labels[i] else 0 for i in range(n)]
                pred_cluster_labels = model.get_binary_clustering(sec, s.para_texts)
                rand = adjusted_rand_score(binary_cluster_labels, pred_cluster_labels)
                nmi = normalized_mutual_info_score(binary_cluster_labels, pred_cluster_labels)
                rand_dict[s.q + sec] = rand
                nmi_dict[s.q + sec] = nmi
    qrels_dat = ir_measures.read_trec_qrels('temp.val.qrels')
    run_dat = ir_measures.read_trec_run('temp.val.run')
    rank_evals = ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)
    return rank_evals, rand_dict, nmi_dict


def eval_mono_bert_ranking(model, samples):
    model.eval()
    with open('temp.val.qrels', 'w') as f:
        for s in samples:
            for i in range(len(s.paras)):
                f.write(s.para_labels[i] + ' 0 ' + s.paras[i] + ' 1\n')
    with open('temp.val.run', 'w') as f:
        for si in range(len(samples)):
            s = samples[si]
            cand_set = s.paras
            n = len(cand_set)
            sections = list(set(s.para_labels))
            for sec in sections:
                pred_score = model.get_rank_scores(sec, s.para_texts)
                for p in range(n):
                    f.write(sec + ' 0 ' + cand_set[p] + ' 0 ' + str(pred_score[p]) + ' val_runid\n')
    qrels_dat = ir_measures.read_trec_qrels('temp.val.qrels')
    run_dat = ir_measures.read_trec_run('temp.val.run')
    rank_evals = ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)
    return rank_evals


def eval_mono_bert_ranking_full(model, page_paras, page_sec_paras, paratext, qrels, per_query=False):
    model.eval()
    with open('temp.val.run', 'w') as f:
        pages = list(page_sec_paras.keys())
        for p in tqdm(range(len(pages))):
            page = pages[p]
            cand_set = page_paras[page]
            n = len(cand_set)
            for sec in page_sec_paras[page].keys():
                cand_set_texts = [paratext[p] for p in cand_set]
                pred_score = model.get_rank_scores(sec, cand_set_texts)
                for i in range(n):
                    f.write(sec + ' 0 ' + cand_set[i] + ' 0 ' + str(pred_score[i]) + ' val_runid\n')
    qrels_dat = ir_measures.read_trec_qrels(qrels)
    run_dat = ir_measures.read_trec_run('temp.val.run')
    if per_query:
        rank_evals = ir_measures.iter_calc([MAP, Rprec, nDCG], qrels_dat, run_dat)
    else:
        rank_evals = ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)
    return rank_evals


def eval_random_ranker(samples):
    rand_dict, nmi_dict = {}, {}
    with open('temp.val.qrels', 'w') as f:
        for s in samples:
            for i in range(len(s.paras)):
                f.write(s.para_labels[i] + ' 0 ' + s.paras[i] + ' 1\n')
    with open('temp.val.run', 'w') as f:
        for si in range(len(samples)):
            s = samples[si]
            cand_set = s.paras
            n = len(cand_set)
            sections = list(set(s.para_labels))
            for sec in sections:
                pred_score = random_ranking(cand_set)
                for p in range(n):
                    f.write(sec + ' 0 ' + cand_set[p] + ' 0 ' + str(pred_score[p]) + ' val_runid\n')
                binary_cluster_labels = [1 if sec == s.para_labels[i] else 0 for i in range(n)]
                pred_cluster_labels = np.random.randint(0, 2, n).tolist()
                rand = adjusted_rand_score(binary_cluster_labels, pred_cluster_labels)
                nmi = normalized_mutual_info_score(binary_cluster_labels, pred_cluster_labels)
                rand_dict[s.q + sec] = rand
                nmi_dict[s.q + sec] = nmi
        qrels_dat = ir_measures.read_trec_qrels('temp.val.qrels')
        run_dat = ir_measures.read_trec_run('temp.val.run')
        rank_evals = ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)
        return rank_evals, rand_dict, nmi_dict


def eval_bm25_ranker(samples):
    with open('temp.val.qrels', 'w') as f:
        for s in samples:
            for i in range(len(s.paras)):
                f.write(s.para_labels[i] + ' 0 ' + s.paras[i] + ' 1\n')
    with open('temp.val.run', 'w') as f:
        for si in range(len(samples)):
            s = samples[si]
            cand_set = s.paras
            n = len(cand_set)
            sections = list(set(s.para_labels))
            for sec in sections:
                pred_score = bm25_ranking(sec, s.para_texts)
                for p in range(n):
                    f.write(sec + ' 0 ' + cand_set[p] + ' 0 ' + str(pred_score[p]) + ' val_runid\n')
        qrels_dat = ir_measures.read_trec_qrels('temp.val.qrels')
        run_dat = ir_measures.read_trec_run('temp.val.run')
        rank_evals = ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)
        return rank_evals


def eval_duo_sbert_treccar_dataset(duo_sbert, samples):
    duo_sbert.eval()
    with open('temp.val.qrels', 'w') as f:
        for s in samples:
            for i in range(len(s.paras)):
                f.write(s.para_labels[i] + ' 0 ' + s.paras[i] + ' 1\n')
    with open('temp.val.run', 'w') as f:
        for si in tqdm(range(len(samples))):
            s = samples[si]
            cand_set = s.paras
            n = len(cand_set)
            texts_for_pi = []
            sections = list(set(s.para_labels))
            for sec in sections:
                if '/' in sec:
                    query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
                else:
                    query = sec.replace('enwiki:', '').replace('%20', ' ')

                for i in range(len(cand_set)):
                    for j in range(len(cand_set)):
                        if i == j:
                            continue
                        texts_for_pi.append((query, s.para_texts[i], s.para_texts[j]))
            input_embs = duo_sbert.emb_model.encode(texts_for_pi, convert_to_tensor=True)
            num_q = len(sections)
            assert input_embs.shape[0] == num_q * n * (n - 1)
            for q in range(num_q):
                rel_score = duo_sbert.get_preds(input_embs[q * n * (n-1): (q+1) * n * (n-1)], n).flatten()
                for i in range(len(cand_set)):
                    f.write(sec + ' 0 ' + cand_set[i] + ' 0 ' + str(rel_score[i]) + ' val_runid\n')
    qrels_dat = ir_measures.read_trec_qrels('temp.val.qrels')
    run_dat = ir_measures.read_trec_run('temp.val.run')
    return ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)


def eval_clustering_duo_sbert(duo_sbert, page_paras, page_sec_paras, paratext, qrels, device):
    duo_sbert.eval()
    pages = list(page_sec_paras.keys())
    for p in tqdm(range(len(pages))):
        page = pages[p]
        paras = []
        true_labels = []
        for s in page_sec_paras[page].keys():
            for para in page_sec_paras[page][s]:
                paras.append(para)
                true_labels.append(s)
        texts = [('', paratext[pi], '') for pi in paras]




def eval_duo_sbert_emb(duo_sbert_emb, page_paras, page_sec_paras, paratext, qrels):
    duo_sbert_emb.eval()
    with open('temp.val.run', 'w') as f:
        pages = list(page_sec_paras.keys())
        for p in tqdm(range(len(pages))):
            page = pages[p]
            cand_set = page_paras[page]
            for sec in page_sec_paras[page].keys():
                if '/' in sec:
                    query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
                else:
                    query = sec.replace('enwiki:', '').replace('%20', ' ')
                for i in range(len(cand_set)):
                    pi = cand_set[i]
                    texts_for_pi = {'q': [], 'p1': [], 'p2': []}
                    for j in range(len(cand_set)):
                        if i == j:
                            continue
                        pj = cand_set[j]
                        texts_for_pi['q'].append(query)
                        texts_for_pi['p1'].append(paratext[pi])
                        texts_for_pi['p2'].append(paratext[pj])
                    rel_score = np.sum(duo_sbert_emb.get_preds(texts_for_pi['q'], texts_for_pi['p1'], texts_for_pi['p2']))
                    f.write(sec + ' 0 ' + pi + ' 0 ' + str(rel_score) + ' val_runid\n')
    qrels_dat = ir_measures.read_trec_qrels(qrels)
    run_dat = ir_measures.read_trec_run('temp.val.run')
    return ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)


def is_fit_for_training(paras, sec_paras):
    return 10 <= len(paras) <= 200 and len(sec_paras.keys()) > 2


class Duo_SBERT(nn.Module):
    def __init__(self, sbert_emb_model, device):
        super(Duo_SBERT, self).__init__()
        self.emb_model = sbert_emb_model
        self.fc1 = nn.Linear(in_features=sbert_emb_model.get_sentence_embedding_dimension(), out_features=1).to(device)
        self.act = nn.Sigmoid()

    def forward(self, input_fet):
        output_emb = self.emb_model(input_fet)['sentence_embedding']
        pred_score = self.act(self.fc1(output_emb))
        return pred_score

    def get_preds(self, input_emb, n):
        pred_score = self.act(self.fc1(input_emb)).detach().cpu().numpy().flatten()
        pred_score = pred_score.reshape((n, n - 1))
        rank_scores = np.sum(pred_score, axis=1)
        return rank_scores


class QSClustering_Model(nn.Module):
    def __init__(self, sbert_emb_model, device, kmeans_plus=False):
        super(QSClustering_Model, self).__init__()
        self.emb_model = sbert_emb_model
        self.device = device
        self.dkm = DKM()
        self.use_kmeans_plus = kmeans_plus

    def forward(self, input_fet, k=2):
        self.qp = self.emb_model(input_fet)['sentence_embedding']
        if self.use_kmeans_plus:
            qp = self.qp.detach().clone().cpu().numpy()
            init_c, _ = kmeans_plusplus(qp, k)
            init_c = torch.tensor(init_c, dtype=torch.float32, device=self.device)
        else:
            init_c = self.qp[random.sample(range(self.qp.shape[0]), k)].detach().clone()
        self.C, self.a = self.dkm(self.qp, init_c)
        return self.C, self.a

    def get_clustering(self, embeddings, k=2, debug_switch=False):
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


class Mono_SBERT_Binary_QSC_Model(nn.Module):
    def __init__(self, sbert_emb_model, device, kmeans_plus=False):
        super(Mono_SBERT_Binary_QSC_Model, self).__init__()
        self.emb_model = sbert_emb_model
        '''
        self.bl1 = nn.Bilinear(in1_features=sbert_emb_model.get_sentence_embedding_dimension(),
                               in2_features=sbert_emb_model.get_sentence_embedding_dimension(),
                               out_features=1).to(device)
        '''
        self.fc1 = nn.Linear(in_features=sbert_emb_model.get_sentence_embedding_dimension(), out_features=1).to(device)
        self.act = nn.Sigmoid()
        self.device = device
        self.dkm = DKM()
        self.use_kmeans_plus = kmeans_plus

    def forward(self, sec, para_texts):
        input_sec_para_texts = []
        for pi in range(len(para_texts)):
            if '/' in sec:
                query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
            else:
                query = sec.replace('enwiki:', '').replace('%20', ' ')
            input_sec_para_texts.append((query, para_texts[pi]))
        input_fet = self.emb_model.tokenize(input_sec_para_texts)
        put_features_in_device(input_fet, self.device)
        output_emb = self.emb_model(input_fet)['sentence_embedding']
        self.pred_score = self.act(self.fc1(output_emb)).flatten()
        if self.use_kmeans_plus:
            output_emb_init = output_emb.detach().clone().cpu().numpy()
            init_c, _ = kmeans_plusplus(output_emb_init, 2)
            init_c = torch.tensor(init_c, dtype=torch.float32, device=self.device)
        else:
            init_c = output_emb[random.sample(range(self.qp.shape[0]), 2)].detach().clone()
        self.C, self.a = self.dkm(output_emb, init_c)
        return self.pred_score, self.C, self.a

    def get_qs_embeddings(self, sec, para_texts):
        input_sec_para_texts = []
        for pi in range(len(para_texts)):
            if '/' in sec:
                query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
            else:
                query = sec.replace('enwiki:', '').replace('%20', ' ')
            input_sec_para_texts.append((query, para_texts[pi]))
        output_emb = self.emb_model.encode(input_sec_para_texts, convert_to_tensor=True)
        return output_emb

    def get_rank_scores(self, sec, para_texts):
        output_emb = self.get_qs_embeddings(sec, para_texts)
        pred_score = self.act(self.fc1(output_emb)).flatten().detach().cpu().numpy()
        return pred_score

    def get_binary_clustering(self, sec, para_texts):
        embeddings = self.get_qs_embeddings(sec, para_texts)
        if self.use_kmeans_plus:
            embeddings_copy = embeddings.detach().clone().cpu().numpy()
            init_c, _ = kmeans_plusplus(embeddings_copy, 2)
            init_c = torch.tensor(init_c, dtype=torch.float32, device=self.device)
        else:
            init_c = embeddings[random.sample(range(embeddings.shape[0]), 2)].detach().clone()
        self.C, self.a = self.dkm(embeddings, init_c)
        pred_labels = torch.argmax(self.a, dim=1).detach().cpu().numpy()
        return pred_labels


class Mono_SBERT_Clustering_Reg_Model(nn.Module):
    def __init__(self, sbert_emb_model, device, kmeans_plus=False):
        super(Mono_SBERT_Clustering_Reg_Model, self).__init__()
        self.emb_model = sbert_emb_model
        self.fc1 = nn.Linear(in_features=sbert_emb_model.get_sentence_embedding_dimension(), out_features=1).to(device)
        self.act = nn.Sigmoid()
        self.device = device

    def forward(self, sec, para_texts):
        input_sec_para_texts = []
        if '/' in sec:
            query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
        else:
            query = sec.replace('enwiki:', '').replace('%20', ' ')
        for pi in range(len(para_texts)):
            input_sec_para_texts.append((query, para_texts[pi]))
        input_fet = self.emb_model.tokenize(input_sec_para_texts)
        put_features_in_device(input_fet, self.device)
        output_emb = self.emb_model(input_fet)['sentence_embedding']
        self.pred_score = self.act(self.fc1(output_emb)).flatten()
        dist_mat = torch.cdist(output_emb, output_emb)
        self.sim_mat = 2 / (1 + torch.exp(dist_mat))
        return self.pred_score, self.sim_mat

    def get_qs_embeddings(self, sec, para_texts):
        input_sec_para_texts = []
        if '/' in sec:
            query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
        else:
            query = sec.replace('enwiki:', '').replace('%20', ' ')
        for pi in range(len(para_texts)):
            input_sec_para_texts.append((query, para_texts[pi]))
        output_emb = self.emb_model.encode(input_sec_para_texts, convert_to_tensor=True)
        return output_emb

    def get_rank_scores(self, sec, para_texts):
        output_emb = self.get_qs_embeddings(sec, para_texts)
        pred_score = self.act(self.fc1(output_emb)).flatten().detach().cpu().numpy()
        return pred_score


def train_duo_sbert(art_qrels,
                    qrels,
                    paratext_tsv,
                    val_art_qrels,
                    val_qrels,
                    val_paratext_tsv,
                    test_art_qrels,
                    test_qrels,
                    test_paratext_tsv,
                    device,
                    model_out,
                    trans_model_name='sentence-transformers/all-MiniLM-L6-v2',
                    max_len=512,
                    max_grad_norm=1.0,
                    weight_decay=0.01,
                    warmup=10000,
                    lrate=2e-5,
                    num_epochs=3,
                    batch_size=8,
                    val_step=1000):
    page_paras, page_sec_paras, paratext = prepare_data(art_qrels, qrels, paratext_tsv)
    val_page_paras, val_page_sec_paras, val_paratext = prepare_data(val_art_qrels, val_qrels, val_paratext_tsv)
    test_page_paras, test_page_sec_paras, test_paratext = prepare_data(test_art_qrels, test_qrels, test_paratext_tsv)
    duo_sbert = Duo_SBERT(trans_model_name, max_len, device)
    model_params = list(duo_sbert.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    train_data_len = 0
    for p in page_sec_paras.keys():
        train_data_len += len(page_sec_paras[p].keys())
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * train_data_len)
    val_eval_score = 0
    train_pages = list(page_sec_paras.keys())
    for p in tqdm(range(len(train_pages)), desc='Training loop'):
        page = train_pages[p]
        if page not in page_paras.keys() or not is_fit_for_training(page_paras[page], page_sec_paras[page]):
            continue
        paras = page_paras[page]
        duo_sbert.train()
        for sec in page_sec_paras[page].keys():
            if '/' in sec:
                query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
            else:
                query = sec.replace('enwiki:', '').replace('%20', ' ')
            rel_paras = page_sec_paras[page][sec]
            nonrel_paras = [p for p in paras if p not in rel_paras]
            random.shuffle(rel_paras)
            random.shuffle(nonrel_paras)
            rel_first_texts, nonrel_first_texts = [], []
            for p1 in rel_paras:
                for p2 in nonrel_paras:
                    rel_first_texts.append((query, paratext[p1], paratext[p2]))
                    nonrel_first_texts.append((query, paratext[p2], paratext[p1]))
            for b in range(len(rel_first_texts) // batch_size):
                rel_first_fet_batch = duo_sbert.emb_model.tokenize(rel_first_texts[b * batch_size: (b+1) * batch_size])
                nonrel_first_fet_batch = duo_sbert.emb_model.tokenize(nonrel_first_texts[b * batch_size: (b+1) * batch_size])
                put_features_in_device(rel_first_fet_batch, device)
                put_features_in_device(nonrel_first_fet_batch, device)
                rel_first_preds = duo_sbert(rel_first_fet_batch)
                nonrel_first_preds = duo_sbert(nonrel_first_fet_batch)
                loss = - torch.sum(torch.log(rel_first_preds)) - torch.sum(torch.log(1 - nonrel_first_preds))
                loss.backward()
                #print('\rLoss: %.4f' % loss.item(), end='')
                nn.utils.clip_grad_norm_(duo_sbert.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
        if (p+1) % val_step == 0:
            print('\nEvaluating on validation set...')
            val_measures = eval_duo_sbert(duo_sbert, val_page_paras, val_page_sec_paras, val_paratext, val_qrels)
            print('\nValidation eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (val_measures[MAP], val_measures[Rprec],
                                                                      val_measures[nDCG]))
            if val_measures[MAP] > val_eval_score:
                torch.save(duo_sbert, model_out)
                val_eval_score = val_measures[MAP]
    print('\nTraining complete. Evaluating on test set...')
    test_measures = eval_duo_sbert(duo_sbert, test_page_paras, test_page_sec_paras, test_paratext, test_qrels)
    print('\nTest eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (test_measures[MAP], test_measures[Rprec],
                                                                    test_measures[nDCG]))


def train_duo_sbert_with_clustering(treccar_data,
                    val_art_qrels,
                    val_qrels,
                    val_paratext_tsv,
                    test_art_qrels,
                    test_qrels,
                    test_paratext_tsv,
                    device,
                    model_out,
                    trans_model_name,
                    max_len,
                    max_grad_norm,
                    weight_decay,
                    warmup,
                    lrate,
                    num_epochs,
                    batch_size,
                    switch_step,
                    binary_clustering,
                    training_mode):

    val_page_paras, val_page_sec_paras, val_paratext = prepare_data(val_art_qrels, val_qrels, val_paratext_tsv)
    test_page_paras, test_page_sec_paras, test_paratext = prepare_data(test_art_qrels, test_qrels, test_paratext_tsv)
    dataset = np.load(treccar_data, allow_pickle=True)[()]['data']
    train_samples = dataset.samples
    val_samples = dataset.val_samples
    test_samples = dataset.test_samples
    #### Smaller experiment ####
    val_samples = val_samples[:10]
    ############################
    trans_model = models.Transformer(trans_model_name, max_seq_length=max_len)
    pool_model = models.Pooling(trans_model.get_word_embedding_dimension())
    emb_model = SentenceTransformer(modules=[trans_model, pool_model]).to(device)
    duo_sbert = Duo_SBERT(emb_model, device)
    clustering_model = QSClustering_Model(emb_model, device)
    model_params = list(set(list(duo_sbert.named_parameters()) + list(clustering_model.named_parameters())))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    train_data_len = len(train_samples)
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * train_data_len)
    val_eval_score = 0
    for i in range((train_data_len // switch_step) + 1):
        print('\nBatch %d/%d' % (i+1, train_data_len // switch_step + 1))
        batch_samples = train_samples[i * switch_step: (i + 1) * switch_step]
        if training_mode == 1 or training_mode == 2:
            for bp in tqdm(range(len(batch_samples)), desc='Retrieval training'):
                paras = batch_samples[bp].paras
                para_texts = batch_samples[bp].para_texts
                para_labels = batch_samples[bp].para_labels
                duo_sbert.train()
                for sec in set(para_labels):
                    if '/' in sec:
                        query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
                    else:
                        query = sec.replace('enwiki:', '').replace('%20', ' ')
                    rel_paras = [i for i in range(len(paras)) if sec == para_labels[i]]
                    nonrel_paras = [i for i in range(len(paras)) if sec is not para_labels[i]]
                    rel_first_texts, nonrel_first_texts = [], []
                    for p1 in rel_paras:
                        for p2 in nonrel_paras:
                            rel_first_texts.append((query, para_texts[p1], para_texts[p2]))
                            nonrel_first_texts.append((query, para_texts[p2], para_texts[p1]))
                    for b in range(len(rel_first_texts) // batch_size):
                        rel_first_fet_batch = duo_sbert.emb_model.tokenize(
                            rel_first_texts[b * batch_size: (b + 1) * batch_size])
                        nonrel_first_fet_batch = duo_sbert.emb_model.tokenize(
                            nonrel_first_texts[b * batch_size: (b + 1) * batch_size])
                        put_features_in_device(rel_first_fet_batch, device)
                        put_features_in_device(nonrel_first_fet_batch, device)
                        rel_first_preds = duo_sbert(rel_first_fet_batch)
                        nonrel_first_preds = duo_sbert(nonrel_first_fet_batch)
                        loss = - torch.sum(torch.log(rel_first_preds)) - torch.sum(torch.log(1 - nonrel_first_preds))
                        loss.backward()
                        nn.utils.clip_grad_norm_(duo_sbert.parameters(), max_grad_norm)
                        opt.step()
                        opt.zero_grad()
                        schd.step()
            #print('\nEvaluating on validation set...')
            val_measures = eval_duo_sbert_treccar_dataset(duo_sbert, val_samples)
            val_rand, val_nmi = eval_clustering(val_samples, clustering_model)
            # print('\nValidation eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (val_measures[MAP], val_measures[Rprec], val_measures[nDCG]))
            print('\nAfter retrieval training, val MAP: %.4f, val RAND: %.4f +- %.4f' % (val_measures[MAP], np.mean(list(val_rand.values())),
                            np.std(list(val_rand.values()), ddof=1) / np.sqrt(len(val_rand.keys()))))
            if val_measures[MAP] > val_eval_score and model_out is not None:
                torch.save(duo_sbert, model_out)
                val_eval_score = val_measures[MAP]
        if training_mode == 1 or training_mode == 3:
            for bp in tqdm(range(len(batch_samples)), desc='Clustering training'):
                paras = batch_samples[bp].paras
                para_texts = batch_samples[bp].para_texts
                para_labels = batch_samples[bp].para_labels
                clustering_model.train()
                if binary_clustering:
                    for sec in set(para_labels):
                        pos_paras = [paras[i] for i in range(len(paras)) if sec == para_labels[i]]
                        sample_labels = [1 if p in pos_paras else 0 for p in paras]
                        if len(set(sample_labels)) < 2:
                            continue
                        if '/' in sec:
                            query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
                        else:
                            query = sec.replace('enwiki:', '').replace('%20', ' ')
                        input_features = clustering_model.emb_model.tokenize([(query, p, '') for p in para_texts])
                        put_features_in_device(input_features, device)
                        mc, ma = clustering_model(input_features)
                        loss = get_weighted_adj_rand_loss(ma, mc, sample_labels, device, False)
                        loss.backward()
                        nn.utils.clip_grad_norm_(clustering_model.parameters(), max_grad_norm)
                        opt.step()
                        opt.zero_grad()
                        schd.step()
                else:
                    query = batch_samples[bp].q.split('enwiki:')[1].replace('%20', ' ')
                    input_features = clustering_model.emb_model.tokenize([(query, p, '') for p in para_texts])
                    put_features_in_device(input_features, device)
                    k = len(set(para_labels))
                    mc, ma = clustering_model(input_features, k)
                    loss = get_weighted_adj_rand_loss(ma, mc, para_labels, device, False)
                    loss.backward()
                    nn.utils.clip_grad_norm_(clustering_model.parameters(), max_grad_norm)
                    opt.step()
                    opt.zero_grad()
                    schd.step()
            #print('\nEvaluating on validation set...')
            val_measures = eval_duo_sbert_treccar_dataset(duo_sbert, val_samples)
            val_rand, val_nmi = eval_clustering(val_samples, clustering_model)
            #print('\nValidation eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (val_measures[MAP], val_measures[Rprec], val_measures[nDCG]))
            print('\nAfter clustering training, val MAP: %.4f, val RAND: %.4f +- %.4f' % (
            val_measures[MAP], np.mean(list(val_rand.values())),
            np.std(list(val_rand.values()), ddof=1) / np.sqrt(len(val_rand.keys()))))
            if val_measures[MAP] > val_eval_score and model_out is not None:
                torch.save(duo_sbert, model_out)
                val_eval_score = val_measures[MAP]
    print('\nTraining complete. Evaluating on test set...')
    test_measures = eval_duo_sbert_treccar_dataset(duo_sbert, test_samples)
    print('\nTest eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (test_measures[MAP], test_measures[Rprec],
                                                              test_measures[nDCG]))
    test_measures_full = eval_duo_sbert(duo_sbert, test_page_paras, test_page_sec_paras, test_paratext, test_qrels)
    print('\nFull Test eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (test_measures_full[MAP], test_measures_full[Rprec],
                                                                              test_measures_full[nDCG]))
    val_measures = eval_duo_sbert_treccar_dataset(duo_sbert, val_samples)
    print('\nValidation eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (val_measures[MAP], val_measures[Rprec],
                                                                    val_measures[nDCG]))
    val_measures_full = eval_duo_sbert(duo_sbert, val_page_paras, val_page_sec_paras, val_paratext, val_qrels)
    print('\nFull Validation eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (val_measures_full[MAP], val_measures_full[Rprec],
                                                                    val_measures_full[nDCG]))


def train_mono_sbert_with_binary_clustering(treccar_data,
                    val_art_qrels,
                    val_qrels,
                    val_paratext_tsv,
                    test_art_qrels,
                    test_qrels,
                    test_paratext_tsv,
                    device,
                    model_out,
                    trans_model_name,
                    max_len,
                    max_grad_norm,
                    weight_decay,
                    warmup,
                    lrate,
                    num_epochs,
                    val_step,
                    lambda_val):

    val_page_paras, val_page_sec_paras, val_paratext = prepare_data(val_art_qrels, val_qrels, val_paratext_tsv)
    test_page_paras, test_page_sec_paras, test_paratext = prepare_data(test_art_qrels, test_qrels, test_paratext_tsv)
    dataset = np.load(treccar_data, allow_pickle=True)[()]['data']
    train_samples = dataset.samples
    val_samples = dataset.val_samples
    test_samples = dataset.test_samples
    #### Smaller experiment ####
    train_samples = train_samples[:10000]
    ############################
    trans_model = models.Transformer(trans_model_name, max_seq_length=max_len)
    pool_model = models.Pooling(trans_model.get_word_embedding_dimension())
    emb_model = SentenceTransformer(modules=[trans_model, pool_model]).to(device)
    model = Mono_SBERT_Binary_QSC_Model(emb_model, device, True)
    model_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    train_data_len = len(train_samples)
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * train_data_len)
    mse = nn.MSELoss()
    val_rank_eval, val_rand_dict, val_nmi_dict = eval_mono_bert_bin_clustering(model, val_samples)
    print('\nInitial val MAP: %.4f, val RAND: %.4f +- %.4f' % (
        val_rank_eval[MAP], np.mean(list(val_rand_dict.values())),
        np.std(list(val_rand_dict.values()), ddof=1) / np.sqrt(len(val_rand_dict.keys()))))
    rand_rank_eval, rand_rand_dict, rand_nmi_dict = eval_random_ranker(val_samples)
    print('\nRandom ranker performance val MAP: %.4f, val RAND: %.4f +- %.4f' % (
        rand_rank_eval[MAP], np.mean(list(rand_rand_dict.values())),
        np.std(list(rand_rand_dict.values()), ddof=1) / np.sqrt(len(rand_rand_dict.keys()))))
    val_eval_score = val_rank_eval[MAP]
    bm25_rank_eval = eval_bm25_ranker(val_samples)
    print('\nBM25 ranker performance val MAP: %.4f' % bm25_rank_eval[MAP])
    for epoch in range(num_epochs):
        print('Epoch %3d' % (epoch + 1))
        for i in tqdm(range(train_data_len)):
            sample = train_samples[i]
            k = len(set(sample.para_labels))
            '''
            if k > 4:
                continue
            '''
            n = len(sample.paras)
            model.train()
            '''
            rk_loss = torch.tensor([0.], requires_grad=True).to(device)
            pred_adj_mat = torch.zeros((n, n), requires_grad=True).to(device)
            true_adj_mat = torch.zeros((n, n), requires_grad=True).to(device)
            
            for p in range(n):
                for q in range(n):
                    if sample.para_labels[p] == sample.para_labels[q]:
                        true_adj_mat[p][q] = 1.0
            '''
            for sec in set(sample.para_labels):
                pred_score, mc, ma = model(sec, sample.para_texts)
                true_labels = [1.0 if sec == sample.para_labels[p] else 0 for p in range(len(sample.para_labels))]
                true_pos_labels_tensor = torch.tensor(true_labels, requires_grad=True).to(device)
                true_neg_labels_tensor = 1 - true_pos_labels_tensor
                true_adj_mat = torch.outer(true_pos_labels_tensor, true_pos_labels_tensor) + torch.outer(
                    true_neg_labels_tensor, true_neg_labels_tensor)
                # cl_loss = get_adj_rand_loss(ma, mc, true_labels, device, False)
                # cl_loss = get_weighted_adj_rand_loss(ma, mc, true_labels, device, False)
                # cl_loss = get_rand_loss(ma, mc, true_labels, device, False)
                pred_adj_mat = torch.outer(pred_score, pred_score)
                rk_loss = mse(pred_score, true_pos_labels_tensor)
                cl_loss = mse(pred_adj_mat, true_adj_mat)
                loss = lambda_val * rk_loss + (1 - lambda_val) * cl_loss
                loss.backward()
                # print('Rank loss: %.4f, Cluster loss: %.4f, Loss: %.4f' % (rk_loss.item(), cl_loss.item(), loss.item()))
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
            if (i + 1) % val_step == 0:
                val_rank_eval, val_rand_dict, val_nmi_dict = eval_mono_bert_bin_clustering(model, val_samples)
                print('\nval MAP: %.4f, val RAND: %.4f +- %.4f' % (
                    val_rank_eval[MAP], np.mean(list(val_rand_dict.values())),
                    np.std(list(val_rand_dict.values()), ddof=1) / np.sqrt(len(val_rand_dict.keys()))))
                if val_rank_eval[MAP] > val_eval_score and model_out is not None:
                    torch.save(model, model_out)
                    val_eval_score = val_rank_eval[MAP]

    val_rank_eval, val_rand_dict, val_nmi_dict = eval_mono_bert_bin_clustering(model, val_samples)
    print('\nval MAP: %.4f, val RAND: %.4f +- %.4f' % (
        val_rank_eval[MAP], np.mean(list(val_rand_dict.values())),
        np.std(list(val_rand_dict.values()), ddof=1) / np.sqrt(len(val_rand_dict.keys()))))
    if val_rank_eval[MAP] > val_eval_score and model_out is not None:
        torch.save(model, model_out)
    print('\nTraining complete. Evaluating on full val and test sets...')
    val_rank_eval, val_rand, val_nmi = eval_mono_bert_bin_clustering_full(model, val_page_paras, val_page_sec_paras,
                                                                          val_paratext, val_qrels)
    print('\nFull val eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f, bin RAND: %.4f +- %.4f' % (
        val_rank_eval[MAP], val_rank_eval[Rprec], val_rank_eval[nDCG], np.mean(list(val_rand.values())),
        np.std(list(val_rand.values()), ddof=1) / np.sqrt(len(val_rand.keys()))))
    test_rank_eval, test_rand, test_nmi = eval_mono_bert_bin_clustering_full(model, test_page_paras,
                                                                        test_page_sec_paras, test_paratext, test_qrels)
    print('\nFull test eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f, bin RAND: %.4f +- %.4f' % (
        test_rank_eval[MAP], test_rank_eval[Rprec], test_rank_eval[nDCG], np.mean(list(test_rand.values())),
        np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys()))))


def train_mono_sbert_with_clustering_reg(treccar_data,
                                            val_art_qrels,
                                            val_qrels,
                                            val_paratext_tsv,
                                            test_art_qrels,
                                            test_qrels,
                                            test_paratext_tsv,
                                            device,
                                            model_out,
                                            trans_model_name,
                                            max_len,
                                            max_grad_norm,
                                            weight_decay,
                                            warmup,
                                            lrate,
                                            num_epochs,
                                            val_step,
                                            lambda_val):
    val_page_paras, val_page_sec_paras, val_paratext = prepare_data(val_art_qrels, val_qrels, val_paratext_tsv)
    test_page_paras, test_page_sec_paras, test_paratext = prepare_data(test_art_qrels, test_qrels, test_paratext_tsv)
    dataset = np.load(treccar_data, allow_pickle=True)[()]['data']
    train_samples = dataset.samples
    val_samples = dataset.val_samples
    test_samples = dataset.test_samples
    #### Smaller experiment ####
    train_samples = train_samples[:10000]
    ############################
    trans_model = models.Transformer(trans_model_name, max_seq_length=max_len)
    pool_model = models.Pooling(trans_model.get_word_embedding_dimension())
    emb_model = SentenceTransformer(modules=[trans_model, pool_model]).to(device)
    model = Mono_SBERT_Clustering_Reg_Model(emb_model, device)
    model_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    train_data_len = len(train_samples)
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * train_data_len)
    mse = nn.MSELoss()
    val_rank_eval = eval_mono_bert_ranking(model, val_samples)
    print('\nInitial val MAP: %.4f' % val_rank_eval[MAP])
    rand_rank_eval, rand_rand_dict, rand_nmi_dict = eval_random_ranker(val_samples)
    print('\nRandom ranker performance val MAP: %.4f' % rand_rank_eval[MAP])
    val_eval_score = val_rank_eval[MAP]
    bm25_rank_eval = eval_bm25_ranker(val_samples)
    print('\nBM25 ranker performance val MAP: %.4f' % bm25_rank_eval[MAP])
    for epoch in range(num_epochs):
        print('Epoch %3d' % (epoch + 1))
        for i in tqdm(range(train_data_len)):
            sample = train_samples[i]
            k = len(set(sample.para_labels))
            '''
            if k > 4:
                continue
            '''
            n = len(sample.paras)
            model.train()
            true_sim_mat = torch.zeros((n, n)).to(device)
            for p in range(n):
                for q in range(n):
                    if sample.para_labels[p] == sample.para_labels[q]:
                        true_sim_mat[p][q] = 1.0
            for sec in set(sample.para_labels):
                pred_score, sim_mat = model(sec, sample.para_texts)
                true_labels = [1.0 if sec == sample.para_labels[p] else 0 for p in range(len(sample.para_labels))]
                true_labels_tensor = torch.tensor(true_labels).to(device)
                rk_loss = mse(pred_score, true_labels_tensor)
                cl_loss = mse(sim_mat, true_sim_mat)
                loss = lambda_val * rk_loss + (1 - lambda_val) * cl_loss
                loss.backward()
                # print('Rank loss: %.4f, Cluster loss: %.4f, Loss: %.4f' % (rk_loss.item(), cl_loss.item(), loss.item()))
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
            if (i + 1) % val_step == 0:
                val_rank_eval = eval_mono_bert_ranking(model, val_samples)
                print('\nval MAP: %.4f' % val_rank_eval[MAP])
                if val_rank_eval[MAP] > val_eval_score and model_out is not None:
                    torch.save(model, model_out)
                    val_eval_score = val_rank_eval[MAP]

    val_rank_eval = eval_mono_bert_ranking(model, val_samples)
    print('\nval MAP: %.4f' % val_rank_eval[MAP])
    if val_rank_eval[MAP] > val_eval_score and model_out is not None:
        torch.save(model, model_out)
    print('\nTraining complete. Evaluating on full val and test sets...')
    val_rank_eval = eval_mono_bert_ranking_full(model, val_page_paras, val_page_sec_paras, val_paratext, val_qrels)
    print('\nFull val eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (
        val_rank_eval[MAP], val_rank_eval[Rprec], val_rank_eval[nDCG]))
    test_rank_eval = eval_mono_bert_ranking_full(model, test_page_paras, test_page_sec_paras, test_paratext, test_qrels)
    print('\nFull test eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (
        test_rank_eval[MAP], test_rank_eval[Rprec], test_rank_eval[nDCG]))


def train_mono_sbert_with_bin_clustering_reg(treccar_data,
                                            val_art_qrels,
                                            val_qrels,
                                            val_paratext_tsv,
                                            test_art_qrels,
                                            test_qrels,
                                            test_paratext_tsv,
                                            device,
                                            model_out,
                                            trans_model_name,
                                            max_len,
                                            max_grad_norm,
                                            weight_decay,
                                            warmup,
                                            lrate,
                                            num_epochs,
                                            val_step,
                                            lambda_val):
    val_page_paras, val_page_sec_paras, val_paratext = prepare_data(val_art_qrels, val_qrels, val_paratext_tsv)
    test_page_paras, test_page_sec_paras, test_paratext = prepare_data(test_art_qrels, test_qrels, test_paratext_tsv)
    dataset = np.load(treccar_data, allow_pickle=True)[()]['data']
    train_samples = dataset.samples
    val_samples = dataset.val_samples
    test_samples = dataset.test_samples
    #### Smaller experiment ####
    train_samples = train_samples[:10000]
    ############################
    trans_model = models.Transformer(trans_model_name, max_seq_length=max_len)
    pool_model = models.Pooling(trans_model.get_word_embedding_dimension())
    emb_model = SentenceTransformer(modules=[trans_model, pool_model]).to(device)
    model = Mono_SBERT_Clustering_Reg_Model(emb_model, device)
    model_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    train_data_len = len(train_samples)
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * train_data_len)
    mse = nn.MSELoss()
    val_rank_eval = eval_mono_bert_ranking(model, val_samples)
    print('\nInitial val MAP: %.4f' % val_rank_eval[MAP])
    rand_rank_eval, rand_rand_dict, rand_nmi_dict = eval_random_ranker(val_samples)
    print('\nRandom ranker performance val MAP: %.4f' % rand_rank_eval[MAP])
    val_eval_score = val_rank_eval[MAP]
    bm25_rank_eval = eval_bm25_ranker(val_samples)
    print('\nBM25 ranker performance val MAP: %.4f' % bm25_rank_eval[MAP])
    for epoch in range(num_epochs):
        print('Epoch %3d' % (epoch + 1))
        for i in tqdm(range(train_data_len)):
            sample = train_samples[i]
            k = len(set(sample.para_labels))
            '''
            if k > 4:
                continue
            '''
            n = len(sample.paras)
            model.train()
            for sec in set(sample.para_labels):
                true_sim_mat = torch.zeros((n, n)).to(device)
                for p in range(n):
                    for q in range(n):
                        if sample.para_labels[p] == sample.para_labels[q] == sec:
                            true_sim_mat[p][q] = 1.0
                        elif sample.para_labels[p] != sec and sample.para_labels[q] != sec:
                            true_sim_mat[p][q] = 1.0
                pred_score, sim_mat = model(sec, sample.para_texts)
                true_labels = [1.0 if sec == sample.para_labels[p] else 0 for p in range(len(sample.para_labels))]
                true_labels_tensor = torch.tensor(true_labels).to(device)
                rk_loss = mse(pred_score, true_labels_tensor)
                cl_loss = mse(sim_mat, true_sim_mat)
                loss = lambda_val * rk_loss + (1 - lambda_val) * cl_loss
                loss.backward()
                # print('Rank loss: %.4f, Cluster loss: %.4f, Loss: %.4f' % (rk_loss.item(), cl_loss.item(), loss.item()))
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
            if (i + 1) % val_step == 0:
                val_rank_eval = eval_mono_bert_ranking(model, val_samples)
                print('\nval MAP: %.4f' % val_rank_eval[MAP])
                if val_rank_eval[MAP] > val_eval_score and model_out is not None:
                    torch.save(model, model_out)
                    val_eval_score = val_rank_eval[MAP]

    val_rank_eval = eval_mono_bert_ranking(model, val_samples)
    print('\nval MAP: %.4f' % val_rank_eval[MAP])
    if val_rank_eval[MAP] > val_eval_score and model_out is not None:
        torch.save(model, model_out)
    print('\nTraining complete. Evaluating on full val and test sets...')
    val_rank_eval = eval_mono_bert_ranking_full(model, val_page_paras, val_page_sec_paras, val_paratext, val_qrels)
    print('\nFull val eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (
        val_rank_eval[MAP], val_rank_eval[Rprec], val_rank_eval[nDCG]))
    test_rank_eval = eval_mono_bert_ranking_full(model, test_page_paras, test_page_sec_paras, test_paratext, test_qrels)
    print('\nFull test eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (
        test_rank_eval[MAP], test_rank_eval[Rprec], test_rank_eval[nDCG]))


class Duo_Siamese_SBERT(nn.Module):
    def __init__(self, sbert_emb_model_name, query_max_len, psg_max_len, device):
        super(Duo_Siamese_SBERT, self).__init__()
        psg_trans_model = models.Transformer(sbert_emb_model_name, max_seq_length=psg_max_len)
        query_trans_model = models.Transformer(sbert_emb_model_name, max_seq_length=query_max_len)
        pool_model = models.Pooling(psg_trans_model.get_word_embedding_dimension())
        self.psg_sbert = SentenceTransformer(modules=[psg_trans_model, pool_model]).to(device)
        self.query_sbert = SentenceTransformer(modules=[query_trans_model, pool_model]).to(device)
        self.fc1 = nn.Linear(in_features=3 * pool_model.get_sentence_embedding_dimension(), out_features=1).to(device)
        self.act = nn.Sigmoid()

    def forward(self, queries_fet, psg1_fet, psg2_fet):
        queries_emb = self.query_sbert(queries_fet)['sentence_embedding']
        psg1_emb = self.psg_sbert(psg1_fet)['sentence_embedding']
        psg2_emb = self.psg_sbert(psg2_fet)['sentence_embedding']
        concat_emb = torch.hstack((queries_emb, psg1_emb, psg2_emb))
        pred_score = self.act(self.fc1(concat_emb))
        return pred_score

    def get_preds(self, query, psg1, psg2):
        queries_emb = self.query_sbert.encode(query, convert_to_tensor=True)
        psg1_emb = self.psg_sbert.encode(psg1, convert_to_tensor=True)
        psg2_emb = self.psg_sbert.encode(psg2, convert_to_tensor=True)
        concat_emb = torch.hstack((queries_emb, psg1_emb, psg2_emb))
        pred_score = self.act(self.fc1(concat_emb)).detach().cpu().numpy().flatten()
        return pred_score


def train_duo_sbert_emb(art_qrels,
                    qrels,
                    paratext_tsv,
                    val_art_qrels,
                    val_qrels,
                    val_paratext_tsv,
                    test_art_qrels,
                    test_qrels,
                    test_paratext_tsv,
                    device,
                    model_out,
                    trans_model_name='sentence-transformers/all-MiniLM-L6-v2',
                    query_max_len=64,
                    psg_max_len=256,
                    max_grad_norm=1.0,
                    weight_decay=0.01,
                    warmup=10000,
                    lrate=2e-5,
                    num_epochs=3,
                    batch_size=8,
                    val_step=1000):
    page_paras, page_sec_paras, paratext = prepare_data(art_qrels, qrels, paratext_tsv)
    val_page_paras, val_page_sec_paras, val_paratext = prepare_data(val_art_qrels, val_qrels, val_paratext_tsv)
    test_page_paras, test_page_sec_paras, test_paratext = prepare_data(test_art_qrels, test_qrels, test_paratext_tsv)
    duo_sbert_emb = Duo_Siamese_SBERT(trans_model_name, query_max_len, psg_max_len, device)
    model_params = list(duo_sbert_emb.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    train_data_len = 0
    for p in page_sec_paras.keys():
        train_data_len += len(page_sec_paras[p].keys())
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * train_data_len)
    val_eval_score = 0
    train_pages = list(page_sec_paras.keys())
    for p in tqdm(range(len(train_pages)), desc='Training loop'):
        page = train_pages[p]
        if page not in page_paras.keys() or not is_fit_for_training(page_paras[page], page_sec_paras[page]):
            continue
        paras = page_paras[page]
        duo_sbert_emb.train()
        for sec in page_sec_paras[page].keys():
            if '/' in sec:
                query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
            else:
                query = sec.replace('enwiki:', '').replace('%20', ' ')
            rel_paras = page_sec_paras[page][sec]
            nonrel_paras = [p for p in paras if p not in rel_paras]
            random.shuffle(rel_paras)
            random.shuffle(nonrel_paras)
            rel_first_texts, nonrel_first_texts = {'q': [], 'p1': [], 'p2': []}, {'q': [], 'p1': [], 'p2': []}
            for p1 in rel_paras:
                for p2 in nonrel_paras:
                    rel_first_texts['q'].append(query)
                    rel_first_texts['p1'].append(paratext[p1])
                    rel_first_texts['p2'].append(paratext[p2])
                    nonrel_first_texts['q'].append(query)
                    nonrel_first_texts['p2'].append(paratext[p1])
                    nonrel_first_texts['p1'].append(paratext[p2])
            for b in range(len(rel_first_texts['q']) // batch_size):
                rel_first_query_fet_batch = duo_sbert_emb.query_sbert.tokenize(
                    rel_first_texts['q'][b * batch_size: (b+1) * batch_size])
                rel_first_p1_fet_batch = duo_sbert_emb.psg_sbert.tokenize(
                    rel_first_texts['p1'][b * batch_size: (b + 1) * batch_size])
                rel_first_p2_fet_batch = duo_sbert_emb.psg_sbert.tokenize(
                    rel_first_texts['p2'][b * batch_size: (b + 1) * batch_size])
                nonrel_first_query_fet_batch = duo_sbert_emb.query_sbert.tokenize(
                    nonrel_first_texts['q'][b * batch_size: (b + 1) * batch_size])
                nonrel_first_p1_fet_batch = duo_sbert_emb.psg_sbert.tokenize(
                    nonrel_first_texts['p1'][b * batch_size: (b + 1) * batch_size])
                nonrel_first_p2_fet_batch = duo_sbert_emb.psg_sbert.tokenize(
                    nonrel_first_texts['p2'][b * batch_size: (b + 1) * batch_size])
                put_features_in_device(rel_first_query_fet_batch, device)
                put_features_in_device(rel_first_p1_fet_batch, device)
                put_features_in_device(rel_first_p2_fet_batch, device)
                put_features_in_device(nonrel_first_query_fet_batch, device)
                put_features_in_device(nonrel_first_p1_fet_batch, device)
                put_features_in_device(nonrel_first_p2_fet_batch, device)
                rel_first_preds = duo_sbert_emb(rel_first_query_fet_batch, rel_first_p1_fet_batch,
                                                rel_first_p2_fet_batch)
                nonrel_first_preds = duo_sbert_emb(nonrel_first_query_fet_batch, nonrel_first_p1_fet_batch,
                                                   nonrel_first_p2_fet_batch)
                loss = - torch.sum(torch.log(rel_first_preds)) - torch.sum(torch.log(1 - nonrel_first_preds))
                loss.backward()
                #print('\rLoss: %.4f' % loss.item(), end='')
                nn.utils.clip_grad_norm_(duo_sbert_emb.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
        if (p+1) % val_step == 0:
            print('\nEvaluating on validation set...')
            val_measures = eval_duo_sbert_emb(duo_sbert_emb, val_page_paras, val_page_sec_paras, val_paratext, val_qrels)
            print('\nValidation eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (val_measures[MAP], val_measures[Rprec],
                                                                      val_measures[nDCG]))
            if val_measures[MAP] > val_eval_score:
                torch.save(duo_sbert_emb, model_out)
                val_eval_score = val_measures[MAP]
    print('\nTraining complete. Evaluating on test set...')
    test_measures = eval_duo_sbert_emb(duo_sbert_emb, test_page_paras, test_page_sec_paras, test_paratext, test_qrels)
    print('\nTest eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (test_measures[MAP], test_measures[Rprec],
                                                                    test_measures[nDCG]))


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    parser = argparse.ArgumentParser(description='Neural ranking')
    parser.add_argument('-td', '--treccar_data',
                        default='D:\\new_cats_data\\QSC_data\\train\\treccar_train_clustering_data_full.npy')
    parser.add_argument('-va', '--val_art_qrels',
                        default='D:\\retrieval_experiments\\val.train.pages.cbor-without-long-article.qrels')
    parser.add_argument('-vq', '--val_qrels',
                        default='D:\\retrieval_experiments\\val.train.pages.cbor-without-long-toplevel.qrels')
    parser.add_argument('-vp', '--val_ptext',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-train-nodup\\by1train_paratext\\by1train_paratext.tsv')
    parser.add_argument('-ta', '--test_art_qrels',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\test.pages.cbor-article.qrels')
    parser.add_argument('-tq', '--test_qrels',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\test.pages.cbor-toplevel.qrels')
    parser.add_argument('-tp', '--test_ptext',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\by1test_paratext\\by1test_paratext.tsv')
    parser.add_argument('-op', '--output_model', default=None)
    parser.add_argument('-ne', '--number_exp', type=int, default=1)
    parser.add_argument('-mn', '--model_name', help='SBERT embedding model name', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('-bt', '--batch', type=int, default=8)
    parser.add_argument('-nt', '--max_num_tokens', type=int, help='Max no. of tokens', default=128)
    parser.add_argument('-gn', '--max_grad_norm', type=float, help='Max gradient norm', default=1.0)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01)
    parser.add_argument('-wu', '--warmup', type=int, default=10000)
    parser.add_argument('-lr', '--lrate', type=float, default=2e-5)
    parser.add_argument('-ep', '--epochs', type=int, default=1)
    parser.add_argument('-ss', '--switch_step', type=int, default=1000)
    parser.add_argument('-ld', '--lambda_val', type=float, default=0.5)
    parser.add_argument('--bin_cluster', action='store_true', default=False)
    parser.add_argument('--ret', action='store_true', default=False)
    parser.add_argument('--clust', action='store_true', default=False)

    args = parser.parse_args()
    if args.number_exp == 1:
        if args.ret and args.clust:
            mode = 1
        elif args.ret:
            mode = 2
        elif args.clust:
            mode = 3
        else:
            sys.exit('No training mode chosen')
        train_duo_sbert_with_clustering(args.treccar_data, args.val_art_qrels, args.val_qrels, args.val_ptext,
                                        args.test_art_qrels, args.test_qrels, args.test_ptext, device, args.output_model,
                                        args.model_name, args.max_num_tokens, args.max_grad_norm, args.weight_decay,
                                        args.warmup, args.lrate, args.epochs, args.batch, args.switch_step,
                                        args.bin_cluster, mode)
    elif args.number_exp == 2:
        train_mono_sbert_with_binary_clustering(args.treccar_data, args.val_art_qrels, args.val_qrels, args.val_ptext,
                                        args.test_art_qrels, args.test_qrels, args.test_ptext, device, args.output_model,
                                        args.model_name, args.max_num_tokens, args.max_grad_norm, args.weight_decay,
                                        args.warmup, args.lrate, args.epochs, args.switch_step, args.lambda_val)

    elif args.number_exp == 3:
        train_mono_sbert_with_clustering_reg(args.treccar_data, args.val_art_qrels, args.val_qrels, args.val_ptext,
                                        args.test_art_qrels, args.test_qrels, args.test_ptext, device, args.output_model,
                                        args.model_name, args.max_num_tokens, args.max_grad_norm, args.weight_decay,
                                        args.warmup, args.lrate, args.epochs, args.switch_step, args.lambda_val)

    elif args.number_exp == 4:
        train_mono_sbert_with_bin_clustering_reg(args.treccar_data, args.val_art_qrels, args.val_qrels, args.val_ptext,
                                        args.test_art_qrels, args.test_qrels, args.test_ptext, device, args.output_model,
                                        args.model_name, args.max_num_tokens, args.max_grad_norm, args.weight_decay,
                                        args.warmup, args.lrate, args.epochs, args.switch_step, args.lambda_val)


if __name__ == '__main__':
    main()