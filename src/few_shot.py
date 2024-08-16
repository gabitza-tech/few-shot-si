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
from utils.utils import CL2N_embeddings,embedding_normalize,embs_norm_both
import sys

#query_file = 'embeddings_vox2/voxmovies_257.pkl'
#support_file = 'embeddings_vox2/voxceleb1_257.pkl'
#test_dict = np.load(query_file, allow_pickle=True)
#enroll_dict = np.load(support_file, allow_pickle=True)   
dataset_file = sys.argv[1]#'embeddings_cn2_vox2/cnceleb1_test.pkl'
merged_dict = np.load(dataset_file,allow_pickle=True)

#dev_file = 'embeddings_vox2/voxceleb1_dev.pkl'
#dev_dict = np.load(dev_file, allow_pickle=True)
#dev_mean = np.mean(dev_dict['concat_features'],axis=0)

out_dir = sys.argv[2]
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

seed = 42
n_tasks = 10000
batch_size = 50

args={}
args['iter']=30

use_mean = sys.argv[3]

n_queries = [5,3,1]
k_shots = [int(sys.argv[4])]
n_ways_effs = [1]

#uniq_classes = sorted(list(set(enroll_dict['concat_labels'])))
uniq_classes = sorted(list(set(merged_dict['concat_labels'])))

for k_shot in k_shots:
    for n_ways_eff in n_ways_effs:
        for n_query in n_queries:
            
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            alpha = n_query
            alpha_glasso = 1000
            acc = {}
            acc["simpleshot_ind"] = []
            acc["simpleshot_maj"] = []
            acc["simpleshot_centroid"] = []
            acc["simpleshot_EM"] = []
            acc["simpleshot_EM_frobenius"] = []
            acc["simpleshot_EM_normal"] = []
            acc["paddle"] = {}
            acc['paddle_2stage'] = {}
            acc["paddle"][str(alpha)] = []
            acc['paddle_2stage'][str(alpha_glasso)] = []
            

            out_filename = f'k_{k_shot}_neff_{n_ways_eff}_nq_{n_query}.json'
            out_file = os.path.join(out_dir,out_filename)
            
            task_generator = Tasks_Generator(uniq_classes=uniq_classes,
                                                n_tasks=n_tasks,
                                                n_ways=len(uniq_classes),
                                                n_ways_eff=n_ways_eff,
                                                n_query=n_query,
                                                k_shot=k_shot,
                                                seed=seed)

            #test_embs, test_labels, test_audios = task_generator.sampler(test_dict,mode='query')
            #enroll_embs, enroll_labels, enroll_audios = task_generator.sampler(enroll_dict,mode='support')
            test_embs, test_labels, test_audios, enroll_embs, enroll_labels, enroll_audios = task_generator.sampler_unified(merged_dict)
            
            enroll_embs, test_embs = CL2N_embeddings(enroll_embs,test_embs,use_mean=use_mean)
            
            for start in tqdm(range(0,n_tasks,batch_size)):
                end = (start+batch_size) if (start+batch_size) <= n_tasks else n_tasks

                x_q,y_q,x_s,y_s = (test_embs[start:end],
                                test_labels[start:end],
                                enroll_embs[start:end],
                                enroll_labels[start:end])

                eval = Simpleshot(avg="mean",backend="cosine",method="inductive")
                acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                acc["simpleshot_ind"].extend(acc_list)
                
                if n_ways_eff == 1:
                    eval = Simpleshot(avg="mean",backend="cosine",method="inductive_maj")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc["simpleshot_maj"].extend(acc_list)

                    eval = Simpleshot(avg="mean",backend="L2",method="EM")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc["simpleshot_EM"].extend(acc_list)

                    eval = Simpleshot(avg="mean",backend="L2",method="EM_normal")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc["simpleshot_EM_normal"].extend(acc_list)

                    eval = Simpleshot(avg="mean",backend="L2",method="EM_frobenius")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc["simpleshot_EM_frobenius"].extend(acc_list)

                    eval = Simpleshot(avg="mean",backend="cosine",method="transductive_centroid")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc['simpleshot_centroid'].extend(acc_list)
                else:
                    eval = Simpleshot(avg="mean",backend="cosine",method="inductive")
                    acc_list,_,_ = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc["simpleshot_ind"].extend(acc_list)

                if n_ways_eff == 1:
                    args['maj_vote'] = True
                else:
                    args['maj_vote'] = False

                args['alpha'] = alpha
                method_info = {'device':'cuda','args':args}
                acc_list,_ = run_paddle_new(x_s, y_s, x_q, y_q,method_info,'paddle')                
                acc['paddle'][str(alpha)].extend(acc_list)

                if k_shot == 1:                
                    continue

                args['alpha'] = alpha_glasso
                method_info = {'device':'cuda','args':args}
                if n_ways_eff == 1:
                    try:
                        acc_list = run_2stage_paddle(x_s, y_s, x_q, y_q, test_audios[start:end], method_info)                
                        acc["paddle_2stage"][str(alpha_glasso)].extend(acc_list)
                    except:
                        continue

            final_json = {}
            final_json['simpleshot_ind'] = 100*sum(acc["simpleshot_ind"])/len(acc["simpleshot_ind"])
            final_json['simpleshot_maj'] = 100*sum(acc["simpleshot_maj"])/len(acc["simpleshot_maj"])
            final_json['simpleshot_centroid'] = 100*sum(acc["simpleshot_centroid"])/len(acc["simpleshot_centroid"])
            final_json['simpleshot_EM'] = 100*sum(acc["simpleshot_EM"])/len(acc["simpleshot_EM"])
            final_json['simpleshot_EM_normal'] = 100*sum(acc["simpleshot_EM_normal"])/len(acc["simpleshot_EM_normal"])
            final_json['simpleshot_EM_frobenius'] = 100*sum(acc["simpleshot_EM_frobenius"])/len(acc["simpleshot_EM_frobenius"])

            final_json['paddle'] = {}
            final_json['paddle_2stage'] = {}

            final_json['paddle'][str(alpha)] = 100*sum(acc["paddle"][str(alpha)])/len(acc["paddle"][str(alpha)])
            
            with open(out_file,'w') as f:
                json.dump(final_json,f)

            if k_shot == 1:
                continue    
            try:
                final_json['paddle_2stage'][str(alpha_glasso)] = 100*sum(acc["paddle_2stage"][str(alpha_glasso)])/len(acc["paddle_2stage"][str(alpha_glasso)])
            except:
                continue

            with open(out_file,'w') as f:
                json.dump(final_json,f)