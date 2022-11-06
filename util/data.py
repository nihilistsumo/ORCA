import json
import math
import random

from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset
from rank_bm25 import BM25Okapi
from sklearn.model_selection import StratifiedKFold


def get_article_qrels(art_qrels):
    qparas = {}
    with open(art_qrels, 'r', encoding='utf-8') as f:
        for l in f:
            q = l.split()[0].strip()
            p = l.split()[2].strip()
            if q not in qparas.keys():
                qparas[q] = [p]
            else:
                qparas[q].append(p)
    return qparas


def get_rev_qrels(qrels):
    para_labels = {}
    with open(qrels, 'r', encoding='utf-8') as f:
        for l in f:
            q = l.split()[0].strip()
            p = l.split()[2].strip()
            para_labels[p] = q
    return para_labels


def get_page_sec_para_dict(qrels_like_file):
    page_sec_paras = {}
    with open(qrels_like_file, 'r') as f:
        for l in f:
            q = l.split(' ')[0]
            p = l.split(' ')[2]
            page = q.split('/')[0] if '/' in q else q
            if page not in page_sec_paras.keys():
                page_sec_paras[page] = {q: [p]}
            elif q not in page_sec_paras[page].keys():
                page_sec_paras[page][q] = [p]
            else:
                page_sec_paras[page][q].append(p)
    return page_sec_paras


def remove_by1_from_train(train_art_qrels, train_qrels, by1train_art_qrels, by1test_art_qrels, train_art_qrels_out,
                          train_qrels_out):
    psgs_to_remove = set()
    articles_to_remove = set()
    with open(by1train_art_qrels, 'r') as f:
        for l in f:
            psgs_to_remove.add(l.split(' ')[2])
            articles_to_remove.add(l.split(' ')[0])
    with open(by1test_art_qrels, 'r') as f:
        for l in f:
            psgs_to_remove.add(l.split(' ')[2])
            articles_to_remove.add(l.split(' ')[0])
    with open(train_art_qrels_out, 'w') as f:
        with open(train_art_qrels, 'r') as r:
            for l in r:
                if l.split(' ')[0] in articles_to_remove or l.split(' ')[2] in psgs_to_remove:
                    print('Removing ' + l)
                else:
                    f.write(l.strip() + '\n')
    with open(train_qrels_out, 'w') as f:
        with open(train_qrels, 'r') as r:
            for l in r:
                if l.split(' ')[0] in articles_to_remove or l.split(' ')[0].split('/')[0] in articles_to_remove or l.split(' ')[2] in psgs_to_remove:
                    print('Removing ' + l)
                else:
                    f.write(l.strip() + '\n')


def put_features_in_device(input_features, device):
    for key in input_features.keys():
        if isinstance(input_features[key], Tensor):
            input_features[key] = input_features[key].to(device)


class TRECCAR_sample(object):
    def __init__(self, q, paras, para_labels, para_texts):
        self.q = q
        self.paras = paras
        self.para_labels = para_labels
        self.para_texts = para_texts

    def __str__(self):
        return 'Sample ' + self.q + ' with %d passages and %d unique clusters' % (len(self.paras),
                                                                                  len(set(self.para_labels)))


class Vital_Wiki_sample(object):
    def __init__(self, q, category, paras, para_labels, para_texts):
        self.q = q
        self.category = category
        self.paras = paras
        self.para_labels = para_labels
        self.para_texts = para_texts

    def __str__(self):
        return 'Sample ' + self.q + ' from category ' + self.category + ' with %d passages and %d unique clusters' % (len(self.paras),
                                                                                  len(set(self.para_labels)))


class Arxiv_Training_Sample(object):
    def __init__(self, q, papers, paper_labels, paper_texts):
        self.q = q
        self.papers = papers
        self.paper_labels = paper_labels
        self.paper_texts = paper_texts

    def __str__(self):
        return 'Sample subject: ' + self.q + ' with %d paper abstracts and %d unique clusters' % (len(self.papers),
                                                                                        len(set(self.paper_labels)))


class TRECCAR_Datset(Dataset):
    def __init__(self, article_qrels, qrels, paratext_tsv, max_num_paras, selected_enwiki_titles=None):
        if selected_enwiki_titles is not None:
            q_paras = get_article_qrels(article_qrels)
            self.q_paras = {}
            for q in q_paras.keys():
                if q in selected_enwiki_titles:
                    self.q_paras[q] = q_paras[q]
            del q_paras
        else:
            self.q_paras = get_article_qrels(article_qrels)
        paras = []
        for q in self.q_paras.keys():
            paras += self.q_paras[q]
        self.paras = set(paras)
        self.queries = list(self.q_paras.keys())
        self.rev_para_labels = get_rev_qrels(qrels)
        self.paratext = {}
        with open(paratext_tsv, 'r', encoding='utf-8') as f:
            for l in f:
                p = l.split('\t')[0]
                if p in self.paras:
                    self.paratext[p] = l.split('\t')[1].strip()
        self.max_num_paras = max_num_paras

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q = self.queries[idx]
        paras = self.q_paras[q]
        para_labels = [self.rev_para_labels[p] for p in paras]
        para_texts = [self.paratext[p] for p in paras]
        num_paras = len(paras)
        num_batches = math.ceil(num_paras / self.max_num_paras)
        batched_samples = []
        for b in range(num_batches):
            paras_batch = paras[b * self.max_num_paras: (b + 1) * self.max_num_paras]
            para_labels_batch = para_labels[b * self.max_num_paras: (b + 1) * self.max_num_paras]
            n = len(paras_batch)
            k = len(set(para_labels_batch))
            if k > 1 and k < n:
                batched_samples.append(TRECCAR_sample(q, paras_batch, para_labels_batch,
                                                      para_texts[b * self.max_num_paras: (b + 1) * self.max_num_paras]))
        return batched_samples


class TRECCAR_Dataset_for_cv(Dataset):
    def __init__(self, rev_para_labels, para_texts, training_articles, testing_articles, fold, max_num_paras=35):
        self.samples = []
        self.test_samples = []
        for a in training_articles.keys():
            paras = training_articles[a][fold]
            random.shuffle(paras)
            if len(paras) > max_num_paras:
                paras = random.sample(paras, max_num_paras)
            para_labels = [rev_para_labels[p] for p in paras]
            texts = [para_texts[p] for p in paras]
            self.samples.append(TRECCAR_sample(a, paras, para_labels, texts))
        for a in testing_articles.keys():
            paras = testing_articles[a][fold]
            para_labels = [rev_para_labels[p] for p in paras]
            texts = [para_texts[p] for p in paras]
            self.test_samples.append(TRECCAR_sample(a, paras, para_labels, texts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TRECCAR_Dataset_full_train_and_by1(Dataset):
    def __init__(self, rev_para_labels, para_texts, training_articles, val_articles, testing_articles):
        self.samples = []
        self.val_samples = []
        self.test_samples = []
        for a in training_articles.keys():
            paras = training_articles[a]
            random.shuffle(paras)
            para_labels = [rev_para_labels[p] for p in paras]
            texts = [para_texts[p] for p in paras]
            self.samples.append(TRECCAR_sample(a, paras, para_labels, texts))
        for a in val_articles.keys():
            paras = val_articles[a]
            random.shuffle(paras)
            para_labels = [rev_para_labels[p] for p in paras]
            texts = [para_texts[p] for p in paras]
            self.val_samples.append(TRECCAR_sample(a, paras, para_labels, texts))
        for a in testing_articles.keys():
            paras = testing_articles[a]
            random.shuffle(paras)
            para_labels = [rev_para_labels[p] for p in paras]
            texts = [para_texts[p] for p in paras]
            self.test_samples.append(TRECCAR_sample(a, paras, para_labels, texts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Vital_Wiki_Dataset(Dataset):
    def __init__(self, vital_cats_data, q_paras, rev_para_qrels, paratext_tsv, max_num_paras):
        self.q_paras = q_paras
        self.vital_cats = vital_cats_data
        self.articles = []
        self.paras = []
        self.rev_art_cat = {}
        for cat in self.vital_cats.keys():
            for a in self.vital_cats[cat]:
                enwiki_art = 'enwiki:' + a
                if enwiki_art in self.q_paras.keys():
                    self.articles.append(enwiki_art)
                    self.rev_art_cat[enwiki_art] = cat
                    self.paras += self.q_paras[enwiki_art]
                else:
                    print(enwiki_art + ' missing in qrels')
        self.paras = set(self.paras)
        self.rev_para_labels = rev_para_qrels
        self.paratext = {}
        with open(paratext_tsv, 'r', encoding='utf-8') as f:
            for l in f:
                p = l.split('\t')[0]
                if p in self.paras:
                    self.paratext[p] = l.split('\t')[1].strip()
        self.max_num_paras = max_num_paras

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        a = self.articles[idx]
        paras = self.q_paras[a]
        para_labels = [self.rev_para_labels[p] for p in paras]
        para_texts = [self.paratext[p] for p in paras]
        num_paras = len(paras)
        num_batches = math.ceil(num_paras / self.max_num_paras)
        batched_samples = []
        for b in range(num_batches):
            paras_batch = paras[b * self.max_num_paras: (b + 1) * self.max_num_paras]
            para_labels_batch = para_labels[b * self.max_num_paras: (b + 1) * self.max_num_paras]
            n = len(paras_batch)
            k = len(set(para_labels_batch))
            if k > 1 and k < n:
                batched_samples.append(TRECCAR_sample(self.rev_art_cat[a], paras_batch, para_labels_batch,
                                                      para_texts[b * self.max_num_paras: (b + 1) * self.max_num_paras]))
        return batched_samples


class Vital_Wiki_Dataset_for_cv(Dataset):
    def __init__(self, rev_para_labels, para_texts, training_articles, testing_articles, fold, max_num_paras=35):
        num_articles_per_category = min([len(training_articles[c].keys()) for c in training_articles.keys()])
        self.samples = []
        self.test_samples = []
        for c in training_articles.keys():
            articles = random.sample(list(training_articles[c].keys()), num_articles_per_category)
            for a in articles:
                paras = training_articles[c][a][fold]
                random.shuffle(paras)
                if len(paras) > max_num_paras:
                    paras = random.sample(paras, max_num_paras)
                para_labels = [rev_para_labels[p] for p in paras]
                texts = [para_texts[p] for p in paras]
                self.samples.append(Vital_Wiki_sample(a, c, paras, para_labels, texts))
        for c in testing_articles.keys():
            for a in testing_articles[c].keys():
                paras = testing_articles[c][a][fold]
                para_labels = [rev_para_labels[p] for p in paras]
                texts = [para_texts[p] for p in paras]
                self.test_samples.append(Vital_Wiki_sample(a, c, paras, para_labels, texts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def manual_selection_of_paras_for_highly_unbalanced_article(article, paras, labels, max_num_paras):
    desired_num_paras_per_label = max_num_paras // len(set(labels))
    label_para_dict = {}
    for i in range(len(paras)):
        if labels[i] not in label_para_dict.keys():
            label_para_dict[labels[i]] = [paras[i]]
        else:
            label_para_dict[labels[i]].append(paras[i])
    selected_paras = []
    for l in label_para_dict.keys():
        if len(label_para_dict[l]) < desired_num_paras_per_label:
            selected_paras += label_para_dict[l]
        else:
            selected_paras += random.sample(label_para_dict[l], desired_num_paras_per_label)
    return selected_paras


class Vital_Wiki_Dataset_for_cv_unseen(Dataset):
    def __init__(self, rev_para_labels, para_texts, train_data, test_data, fold, max_num_paras=35):
        self.samples = []
        self.test_samples = []
        current_fold_train_data = train_data[fold]
        current_fold_test_data = test_data[fold]
        for d in current_fold_train_data:
            article = d['article']
            cat = d['category']
            paras = d['paras']
            random.shuffle(paras)
            para_labels_original = [rev_para_labels[p] for p in d['paras']]
            if len(paras) > max_num_paras:
                paras = random.sample(paras, max_num_paras)
                para_labels = [rev_para_labels[p] for p in paras]
                i = 0
                while len(set(para_labels)) == 1:
                    paras = random.sample(paras, max_num_paras)
                    para_labels = [rev_para_labels[p] for p in paras]
                    i += 1
                    if i > 1000:
                        print('Calling expensive sampler for article: ' + article)
                        paras = manual_selection_of_paras_for_highly_unbalanced_article(article, d['paras'],
                                                                                para_labels_original, max_num_paras)
                        break
            para_labels = [rev_para_labels[p] for p in paras]
            texts = [para_texts[p] for p in paras]
            self.samples.append(Vital_Wiki_sample(article, cat, paras, para_labels, texts))
        for d in current_fold_test_data:
            article = d['article']
            cat = d['category']
            paras = d['paras']
            para_labels = [rev_para_labels[p] for p in paras]
            texts = [para_texts[p] for p in paras]
            self.test_samples.append(Vital_Wiki_sample(article, cat, paras, para_labels, texts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Arxiv_Dataset(Dataset):
    def __init__(self, arxiv_, training_ids, testing_ids, fold, max_num_samples_per_query=1000, max_num_papers=35, max_k=5):
        self.arxiv_data = {}
        for s in arxiv_.keys():
            self.arxiv_data[s] = [arxiv_[s][i] for i in training_ids[s][fold]]
        self.test_arxiv_data = {}
        for s in arxiv_.keys():
            self.test_arxiv_data[s] = [arxiv_[s][i] for i in testing_ids[s][fold]]
        self.rev_subfield_abstract = {}
        for s in self.arxiv_data.keys():
            for i in range(len(self.arxiv_data[s])):
                d = self.arxiv_data[s][i]
                if d['label'] not in self.rev_subfield_abstract.keys():
                    self.rev_subfield_abstract[d['label']] = [i]
                else:
                    self.rev_subfield_abstract[d['label']].append(i)
        self.subjects = list(self.arxiv_data.keys())
        self.subfield_dict = {}
        self.samples = []
        for s in self.subjects:
            self.subfield_dict[s] = list(set([d['label'] for d in self.arxiv_data[s]]))
            for i in range(max_num_samples_per_query):
                num_k = min(max_k, len(self.subfield_dict[s]))
                subfields = random.sample(self.subfield_dict[s], num_k)
                papers = []
                for sf in subfields:
                    papers += random.sample(self.rev_subfield_abstract[sf], max_num_papers // num_k)
                random.shuffle(papers)
                self.samples.append((s, papers))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        q = sample[0]
        papers = sample[1]
        paper_labels = [self.arxiv_data[q][pi]['label'] for pi in papers]
        paper_texts = [self.arxiv_data[q][pi]['abstract'] for pi in papers]
        return Arxiv_Training_Sample(q, papers, paper_labels, paper_texts)


def prepare_arxiv_data_for_cv(arxiv_data_file, num_folds=5):
    with open(arxiv_data_file, 'r') as f:
        arxiv_data = json.load(f)
    training_ids, test_ids = {}, {}
    for s in arxiv_data.keys():
        papers = arxiv_data[s]
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
        for train_indices, test_indices in skf.split(papers, [p['label'] for p in papers]):
            train_indices, test_indices = list(train_indices), list(test_indices)
            if s in training_ids.keys():
                training_ids[s].append(train_indices)
            else:
                training_ids[s] = [train_indices]
            if s in test_ids.keys():
                test_ids[s].append(test_indices)
            else:
                test_ids[s] = [test_indices]
    cv_datasets = []
    for i in range(num_folds):
        cv_datasets.append(Arxiv_Dataset(arxiv_data, training_ids, test_ids, i))
    return cv_datasets


def prepare_vital_wiki_data_for_cv(art_qrels, qrels, paratext_tsv, vital_cats_data, num_folds=2):
    page_paras = get_article_qrels(art_qrels)
    rev_para_labels = get_rev_qrels(qrels)
    with open(vital_cats_data, 'r') as f:
        cats = json.load(f)
    all_vital_cats_dict = {}
    for d in cats:
        for c in d.keys():
            if c not in all_vital_cats_dict.keys():
                all_vital_cats_dict[c] = [a for a in d[c] if 'enwiki:' + a in page_paras.keys()]
            else:
                all_vital_cats_dict[c] += [a for a in d[c] if 'enwiki:' + a in page_paras.keys()]
    train_article_paras, test_article_paras = {}, {}
    all_paras = []
    for c in all_vital_cats_dict.keys():
        train_article_paras[c] = {}
        test_article_paras[c] = {}
        for a in all_vital_cats_dict[c]:
            article = 'enwiki:' + a
            paras = page_paras[article]
            para_labels = [rev_para_labels[p] for p in paras]
            para_label_count = {label: para_labels.count(label) for label in para_labels}
            selected_paras, selected_labels = [], []
            for i in range(len(paras)):
                p = paras[i]
                if para_label_count[para_labels[i]] > 3:
                    selected_paras.append(p)
                    selected_labels.append(para_labels[i])
            if article in page_paras.keys() and len(selected_paras) > 0 and len(set(selected_labels)) > 1:
                all_paras += selected_paras
                skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
                for train_indices, test_indices in skf.split(selected_paras, selected_labels):
                    if article in train_article_paras[c].keys():
                        train_article_paras[c][article].append([selected_paras[i] for i in train_indices])
                    else:
                        train_article_paras[c][article] = [[selected_paras[i] for i in train_indices]]
                    if article in test_article_paras[c].keys():
                        test_article_paras[c][article].append([selected_paras[i] for i in test_indices])
                    else:
                        test_article_paras[c][article] = [[selected_paras[i] for i in test_indices]]
    all_paras = set(all_paras)
    paratext = {}
    with open(paratext_tsv, 'r', encoding='utf-8') as f:
        for l in f:
            p = l.split('\t')[0]
            if p in all_paras:
                paratext[p] = l.split('\t')[1].strip()
    cv_datasets = []
    for i in range(num_folds):
        cv_datasets.append(Vital_Wiki_Dataset_for_cv(rev_para_labels, paratext,
                                                     train_article_paras, test_article_paras, i))
    return cv_datasets


def prepare_vital_wiki_data_for_cv_unseen_queries(art_qrels, qrels, paratext_tsv, vital_cats_data, num_folds=5):
    page_paras = get_article_qrels(art_qrels)
    rev_para_labels = get_rev_qrels(qrels)
    with open(vital_cats_data, 'r') as f:
        cats = json.load(f)
    selected_vital_cat_labels = []
    selected_articles = []
    selected_page_paras = {}
    all_selected_paras = []
    for d in cats:
        for c in d.keys():
            for k in tqdm(range(len(d[c]))):
                a = d[c][k]
                article = 'enwiki:' + a
                if article in page_paras.keys():
                    paras = page_paras[article]
                    para_labels = [rev_para_labels[p] for p in paras]
                    para_label_count = {label: para_labels.count(label) for label in para_labels}
                    selected_paras, selected_labels = [], []
                    for i in range(len(paras)):
                        p = paras[i]
                        if para_label_count[para_labels[i]] > 3:
                            selected_paras.append(p)
                            selected_labels.append(para_labels[i])
                    if len(selected_paras) > 0 and len(set(selected_labels)) > 1:
                        selected_page_paras[article] = selected_paras
                        all_selected_paras += selected_paras
                        selected_articles.append(article)
                        selected_vital_cat_labels.append(c)
    paratext = {}
    all_selected_paras = set(all_selected_paras)
    with open(paratext_tsv, 'r', encoding='utf-8') as f:
        for l in f:
            p = l.split('\t')[0]
            if p in all_selected_paras:
                paratext[p] = l.split('\t')[1].strip()
    print('Para texts read')
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    print('k-fold data generating')
    train_data, test_data = [], []
    for train_indices, test_indices in skf.split(selected_articles, selected_vital_cat_labels):
        print('%d train, %d test samples' % (len(train_indices), len(test_indices)))
        current_fold_train_data, current_fold_test_data = [], []
        for i in train_indices:
            a = selected_articles[i]
            c = selected_vital_cat_labels[i]
            current_fold_train_data.append({'article': a, 'category': c, 'paras': selected_page_paras[a]})
        for j in test_indices:
            a = selected_articles[j]
            c = selected_vital_cat_labels[j]
            current_fold_test_data.append({'article': a, 'category': c, 'paras': selected_page_paras[a]})
        train_data.append(current_fold_train_data)
        test_data.append(current_fold_test_data)

    cv_datasets = []
    for i in range(num_folds):
        cv_datasets.append(Vital_Wiki_Dataset_for_cv_unseen(rev_para_labels, paratext, train_data, test_data, i))
    return cv_datasets


def prepare_treccar_data_for_cv(art_qrels, qrels, paratext_tsv, num_folds=2):
    page_paras = get_article_qrels(art_qrels)
    rev_para_labels = get_rev_qrels(qrels)
    train_article_paras, test_article_paras = {}, {}
    all_paras = []
    for article in page_paras.keys():
        paras = page_paras[article]
        para_labels = [rev_para_labels[p] for p in paras]
        para_label_count = {label: para_labels.count(label) for label in para_labels}
        selected_paras, selected_labels = [], []
        for i in range(len(paras)):
            p = paras[i]
            if para_label_count[para_labels[i]] > 3:
                selected_paras.append(p)
                selected_labels.append(para_labels[i])
        if len(selected_paras) > 0 and len(set(selected_labels)) > 1:
            all_paras += selected_paras
            skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
            for train_indices, test_indices in skf.split(selected_paras, selected_labels):
                if article in train_article_paras.keys():
                    train_article_paras[article].append([selected_paras[i] for i in train_indices])
                else:
                    train_article_paras[article] = [[selected_paras[i] for i in train_indices]]
                if article in test_article_paras.keys():
                    test_article_paras[article].append([selected_paras[i] for i in test_indices])
                else:
                    test_article_paras[article] = [[selected_paras[i] for i in test_indices]]
    all_paras = set(all_paras)
    paratext = {}
    with open(paratext_tsv, 'r', encoding='utf-8') as f:
        for l in f:
            p = l.split('\t')[0]
            if p in all_paras:
                paratext[p] = l.split('\t')[1].strip()
    cv_datasets = []
    for i in range(num_folds):
        cv_datasets.append(TRECCAR_Dataset_for_cv(rev_para_labels, paratext, train_article_paras, test_article_paras, i))
    return cv_datasets


def prepare_treccar_data_unseen_full(train_art_qrels, train_qrels, train_paratext, by1train_art_qrels, by1train_qrels,
                                     by1train_paratext, by1test_art_qrels, by1test_qrels, by1test_paratext, max_num_paras=35):
    page_paras = get_article_qrels(train_art_qrels)
    page_paras_by1train = get_article_qrels(by1train_art_qrels)
    page_paras_by1test = get_article_qrels(by1test_art_qrels)
    rev_para_labels = get_rev_qrels(train_qrels)
    rev_para_labels_by1train = get_rev_qrels(by1train_qrels)
    rev_para_labels_by1test = get_rev_qrels(by1test_qrels)
    for k in rev_para_labels_by1train.keys():
        rev_para_labels[k] = rev_para_labels_by1train[k]
    for k in rev_para_labels_by1test.keys():
        rev_para_labels[k] = rev_para_labels_by1test[k]
    training_articles, val_articles, test_articles = {}, {}, {}
    all_paras = []
    print('Building training set')
    articles = list(page_paras.keys())
    for ar in tqdm(range(len(articles))):
        article = articles[ar]
        paras = page_paras[article]
        para_labels = [rev_para_labels[p] for p in paras]
        para_label_count = {label: para_labels.count(label) for label in para_labels}
        selected_paras, selected_labels = [], []
        for i in range(len(paras)):
            p = paras[i]
            if para_label_count[para_labels[i]] > 3:
                selected_paras.append(p)
                selected_labels.append(para_labels[i])
        if len(selected_paras) > 0 and len(set(selected_labels)) > 1:
            if len(selected_paras) > max_num_paras:
                sampled_selected_paras = random.sample(selected_paras, max_num_paras)
                labels = [rev_para_labels[p] for p in sampled_selected_paras]
                i = 0
                while len(set(labels)) == 1:
                    sampled_selected_paras = random.sample(selected_paras, max_num_paras)
                    labels = [rev_para_labels[p] for p in sampled_selected_paras]
                    i += 1
                    if i > 10000:
                        print('Could not generate well-distributed sample from query: ' + article)
                        break
                if i > 10000:
                    continue
                selected_paras = sampled_selected_paras
            training_articles[article] = selected_paras
            all_paras += selected_paras
    for article in page_paras_by1train.keys():
        val_articles[article] = page_paras_by1train[article]
        all_paras += page_paras_by1train[article]
    for article in page_paras_by1test.keys():
        test_articles[article] = page_paras_by1test[article]
        all_paras += page_paras_by1test[article]
    all_paras = set(all_paras)
    paratext = {}
    with open(train_paratext, 'r', encoding='utf-8') as f:
        for l in f:
            p = l.split('\t')[0]
            if p in all_paras:
                paratext[p] = l.split('\t')[1].strip()
    with open(by1train_paratext, 'r', encoding='utf-8') as f:
        for l in f:
            p = l.split('\t')[0]
            if p in all_paras:
                paratext[p] = l.split('\t')[1].strip()
    with open(by1test_paratext, 'r', encoding='utf-8') as f:
        for l in f:
            p = l.split('\t')[0]
            if p in all_paras:
                paratext[p] = l.split('\t')[1].strip()
    dataset = TRECCAR_Dataset_full_train_and_by1(rev_para_labels, paratext, training_articles, val_articles, test_articles)
    return dataset


def separate_multi_batch(sample, max_num_paras):
    num_paras = len(sample.paras)
    num_batches = math.ceil(num_paras / max_num_paras)
    batched_samples = []
    for b in range(num_batches):
        paras = sample.paras[b * max_num_paras: (b+1) * max_num_paras]
        para_labels = sample.para_labels[b * max_num_paras: (b+1) * max_num_paras]
        n = len(paras)
        k = len(set(para_labels))
        if k > 1 and k < n:
            batched_samples.append(TRECCAR_sample(sample.q, paras, para_labels,
                                                  sample.para_texts[b * max_num_paras: (b+1) * max_num_paras]))
    return batched_samples


def get_similar_treccar_articles(query_enwiki_titles, article_qrels, n):
    art_qrels = get_article_qrels(article_qrels)
    enwiki_map = {}
    for q in art_qrels.keys():
        enwiki_map[q.split('enwiki:')[1].replace('%20', ' ')] = q
    corpus = list(enwiki_map.keys())
    tokenized_corpus = [q.split(' ') for q in corpus]
    query_titles = [q.split('enwiki:')[1].replace('%20', ' ') for q in query_enwiki_titles]
    bm25 = BM25Okapi(tokenized_corpus)
    result = []
    print('Finding similar articles')
    for q in query_titles:
        result += [enwiki_map[a] for a in bm25.get_top_n(q.split(' '), corpus, n)
                   if enwiki_map[a] not in query_enwiki_titles]
    return result