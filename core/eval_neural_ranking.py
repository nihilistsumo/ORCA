import os
import argparse
import torch
import pandas as pd
from ir_measures import AP, nDCG, Rprec
from scipy.stats import sem
from core.neural_ranking import eval_mono_bert_ranking_full, prepare_data, Mono_SBERT_Clustering_Reg_Model
from core.idcm_try import eval_idcm_ranking_full


def do_eval(model_dir, art_qrels, qrels, paratext_tsv, output_dir, output_file_prefix='val', pretrained_model_list=None, only_overview_queries=True):
    page_paras, page_sec_paras, paratext = prepare_data(art_qrels, qrels, paratext_tsv)
    model_name_list = os.listdir(model_dir)
    if pretrained_model_list is not None and len(pretrained_model_list) > 0:
        model_name_list += pretrained_model_list
    model_types = set([m.split('_try')[0] for m in model_name_list])
    modeltype_df = {}
    for mname in model_name_list:
        if mname == 'idcm':
            model_type = 'idcm'
            rank_eval = eval_idcm_ranking_full(page_paras, page_sec_paras, paratext, qrels, True)
        else:
            model = torch.load(model_dir + '\\' + mname)
            model_type = mname.split('_try')[0]
            rank_eval = eval_mono_bert_ranking_full(model, page_paras, page_sec_paras, paratext, qrels, True)
        results_per_query = {}
        for metric in rank_eval:
            q = metric.query_id
            if q in results_per_query.keys():
                results_per_query[q][metric.measure] = metric.value
            else:
                results_per_query[q] = {metric.measure: metric.value}
        queries = list(results_per_query.keys())
        if only_overview_queries:
            queries = [q for q in queries if '/' not in q]
        df = pd.DataFrame.from_dict({'Queries': queries, 'AP': [results_per_query[q][AP] for q in queries],
                                     'nDCG': [results_per_query[q][nDCG] for q in queries],
                                     'Rprec': [results_per_query[q][Rprec] for q in queries]})
        if model_type not in modeltype_df.keys():
            modeltype_df[model_type] = [df]
        else:
            modeltype_df[model_type].append(df)
        print(mname)
    for mtype in model_types:
        df_con = pd.concat(modeltype_df[mtype])
        df_con.to_parquet(output_dir + '\\' + output_file_prefix + '_' + mtype + '.parquet')
        df_mean = df_con[['AP', 'nDCG', 'Rprec']].mean()
        df_sterr = df_con[['AP', 'nDCG', 'Rprec']].apply(lambda x: sem(x))
        print(mtype, ' MAP: %.4f +- %.4f, nDCG: %.4f +- %.4f, Rprec: %.4f +- %.4f' % (
            df_mean['AP'], df_sterr['AP'], df_mean['nDCG'], df_sterr['nDCG'], df_mean['Rprec'], df_sterr['Rprec']))


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    parser = argparse.ArgumentParser(description='Neural ranking evaluation')
    parser.add_argument('-va', '--val_art_qrels',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-train-nodup\\train.pages.cbor-article.qrels')
    parser.add_argument('-vq', '--val_qrels',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-train-nodup\\train.pages.cbor-toplevel.qrels')
    parser.add_argument('-vp', '--val_ptext',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-train-nodup\\by1train_paratext\\by1train_paratext.tsv')
    parser.add_argument('-ta', '--test_art_qrels',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\test.pages.cbor-article.qrels')
    parser.add_argument('-tq', '--test_qrels',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\test.pages.cbor-toplevel.qrels')
    parser.add_argument('-tp', '--test_ptext',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\by1test_paratext\\by1test_paratext.tsv')
    parser.add_argument('-md', '--model_dir', default='D:\\retrieval_experiments\\monobert_clustering_reg\\saved_models')
    parser.add_argument('-ne', '--number_exp', type=int, default=1)
    parser.add_argument('-mn', '--model_name', help='SBERT embedding model name', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('-op', '--output', help='Path to output directory', default='D:\\retrieval_experiments\\monobert_clustering_reg\\new_eval_output')

    args = parser.parse_args()
    print('Validation results')
    print('==================')
    do_eval(args.model_dir, args.val_art_qrels, args.val_qrels, args.val_ptext, args.output, pretrained_model_list=['idcm'])
    print('\nTest results')
    print('============')
    do_eval(args.model_dir, args.test_art_qrels, args.test_qrels, args.test_ptext, args.output, 'test', pretrained_model_list=['idcm'])


if __name__ == '__main__':
    main()