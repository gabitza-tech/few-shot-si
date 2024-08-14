import numpy as np
import sys
from collections import Counter
import random
import torch
import json
import time
from utils.task_generator import Tasks_Generator
from tqdm import tqdm
from methods.simpleshot import Simpleshot
from methods.methods import run_paddle_new,run_2stage_paddle
import os 
from utils.utils import save_pickle
from utils.utils import CL2N_embeddings, embedding_normalize
from methods.paddle import PADDLE
from utils.paddle_utils import get_log_file,Logger

#query_file = 'datasets_splits/embeddings/voxmovies_3s_ecapa_embs_257.pkl'
#support_file = 'datasets_splits/embeddings/voxceleb1_3s_movies_ecapa_embs_257.pkl'
#enroll_dict = np.load(support_file, allow_pickle=True)
#test_dict = np.load(query_file, allow_pickle=True)
#merged_dict = {}
#for key in enroll_dict.keys():
#    merged_dict[key] = np.concatenate((enroll_dict[key],test_dict[key]),axis=0)
dataset_file = 'embeddings_vox2/voxceleb1_test.pkl'
merged_dict = np.load(dataset_file,allow_pickle=True)

seed = 42
n_tasks = 100
batch_size = 100

normalize = True
use_mean = True

args={}
args['iter']=20

n_queries = [5,3,1]
k_shots = [5,3,1]
n_ways_effs = [1]#[5,3,1]

uniq_classes = sorted(list(set(merged_dict['concat_labels'])))
#uniq_classes = sorted(list(set(enroll_dict['concat_labels'])))
print(len(uniq_classes))
tune_dir = "logs/log_alpha_voxceleb1_val_125_ways_3s"
out_dir = "logs/log_alpha_Q_voxceleb1_S_voxmovies_257_ways_3s_tuned+2stage"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

tune_dict = {}
for file in os.listdir(tune_dir):
    filename = file.split(".")[0]
    filepath = os.path.join(tune_dir,file)
    with open(filepath,'r') as f:
        tune_file = json.load(f)
    
    max_acc = 0
    max_key = 0
    for key in tune_file['paddle'].keys():
        if tune_file['paddle'][key] > max_acc:
            max_acc = tune_file['paddle'][key]
            max_key = key

    tune_dict[filename] = max_key

for k_shot in k_shots:
    for n_ways_eff in n_ways_effs:
        for n_query in n_queries:
            
            out_filename = f'k_{k_shot}_neff_{n_ways_eff}_nq_{n_query}.json'
            tuned_alpha = tune_dict[f'k_{k_shot}_neff_{n_ways_eff}_nq_{n_query}']
            out_file = os.path.join(out_dir,out_filename)
            
            #log_file = get_log_file(log_path='log_alpha_experiments', method=f'k_{k_shot}_neff_{n_ways_eff}_nq_{n_query}', backbone='ecapa', dataset='voxceleb1')
            #logger = Logger(__name__, log_file)
            
            task_generator = Tasks_Generator(uniq_classes=uniq_classes,
                                                n_tasks=n_tasks,
                                                n_ways=len(set(merged_dict['concat_labels'])),
                                                n_ways_eff=n_ways_eff,
                                                n_query=n_query,
                                                k_shot=k_shot,
                                                seed=seed)

            test_embs, test_labels, test_audios = task_generator.sampler(test_dict,mode='query')
            enroll_embs, enroll_labels, enroll_audios = task_generator.sampler(enroll_dict,mode='support')
            #test_embs, test_labels, test_audios,enroll_embs, enroll_labels, enroll_audios = task_generator.sampler_unified(merged_dict)
            #enroll_embs, test_embs = CL2N_embeddings(enroll_embs,test_embs,normalize)
            test_embs = embedding_normalize(test_embs,use_mean=use_mean)
            enroll_embs = embedding_normalize(enroll_embs,use_mean=use_mean)

            acc = {}
            acc["simpleshot"] = []
            acc["paddle"] = {}
            acc["paddle"][str(tuned_alpha)] = []
            acc['paddle_2stage'] = {}
            acc['paddle_2stage'][str(tuned_alpha)] = []

            for start in tqdm(range(0,n_tasks,batch_size)):
                end = (start+batch_size) if (start+batch_size) <= n_tasks else n_tasks

                x_q,y_q,x_s,y_s = (test_embs[start:end],
                                test_labels[start:end],
                                enroll_embs[start:end],
                                enroll_labels[start:end])

                if n_ways_eff == 1:
                    eval = Simpleshot(avg="mean",backend="L2",method="transductive_centroid")
                    acc_list,_,_ = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                else:
                    eval = Simpleshot(avg="mean",backend="L2",method="inductive")
                    acc_list,_,_ = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                acc["simpleshot"].extend(acc_list)

                if n_ways_eff == 1:
                    args['maj_vote'] = True
                else:
                    args['maj_vote'] = False

                args['alpha'] = int(tuned_alpha)
                method_info = {'device':'cuda','args':args}#'log_file':log_file,'args':args}
                acc_list,_ = run_paddle_new(x_s, y_s, x_q, y_q,method_info)
                acc["paddle"][str(tuned_alpha)].extend(acc_list)

                if n_ways_eff == 1:
                    acc_list = run_2stage_paddle(x_s, y_s, x_q, y_q, test_audios[start:end], method_info)                
                    acc["paddle_2stage"][str(tuned_alpha)].extend(acc_list)

            final_json = {}
            final_json['simpleshot'] = 100*sum(acc["simpleshot"])/len(acc["simpleshot"])
            final_json['paddle'] = {}
            final_json['paddle_2stage'] = {}
            
            final_json['paddle'][str(tuned_alpha)] = 100*sum(acc["paddle"][str(tuned_alpha)])/len(acc["paddle"][str(tuned_alpha)])
            final_json['paddle_2stage'][str(tuned_alpha)] = 100*sum(acc["paddle_2stage"][str(tuned_alpha)])/len(acc["paddle_2stage"][str(tuned_alpha)])

            with open(out_file,'w') as f:
                json.dump(final_json,f)
                
