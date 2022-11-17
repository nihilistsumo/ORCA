import optuna
import transformers
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from transformers import AdamW
from sentence_transformers import models, SentenceTransformer
from ir_measures import MAP, nDCG
from core.neural_ranking import Mono_SBERT_Clustering_Reg_Model, eval_mono_bert_ranking

trec_data = np.load('D:\\new_cats_data\\QSC_data\\train\\treccar_train_clustering_data_full.npy', allow_pickle=True)[()]['data']
TRAIN_SAMPLES = trec_data.samples[:1000]
VAL_SAMPLES = trec_data.val_samples[:100]


def train_mono_sbert_with_clustering_reg(trans_model_name,
                                            max_len,
                                            max_grad_norm,
                                            weight_decay,
                                            warmup,
                                            lrate,
                                            num_epochs,
                                            lambda_val):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    train_samples = TRAIN_SAMPLES
    val_samples = VAL_SAMPLES
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
    for epoch in range(num_epochs):
        for i in tqdm(range(train_data_len)):
            sample = train_samples[i]
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
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()

    val_rank_eval = eval_mono_bert_ranking(model, val_samples)
    return val_rank_eval[MAP]


def train_and_evaluate(params):
    return train_mono_sbert_with_clustering_reg(params['trans_model_name'],
                                                params['max_len'],
                                                params['max_grad_norm'],
                                                params['weight_decay'],
                                                params['warmup'],
                                                params['lrate'],
                                                params['num_epochs'],
                                                params['lambda_val'])


def objective(trial):
    params = {
        'trans_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'max_len': 128,
        'max_grad_norm': 1.0,
        'weight_decay': 0.01,
        'warmup': 10000,
        'lrate': 2e-5,
        'num_epochs': 1,
        'lambda_val': trial.suggest_float('lambda_val', 0.0, 0.9999)
    }
    map_score = train_and_evaluate(params)
    return map_score


def main():
    EPOCHS = 30
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=EPOCHS)


if __name__ == '__main__':
    main()
