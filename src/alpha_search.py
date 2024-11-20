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
from methods.fsaic import FSAIC
from methods.methods import run_2stage_method,run_method
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
n_tasks = 10#000
batch_size = 10#000
top_k = 5

normalize = True
use_mean = False

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

#alphas = [5]#[i for i in range(0,20)]#[i for i in range(0, 16) if i % 3 == 0 or i % 5 == 0]
alphas_glasso = [1000]#[100,1000,10000]#[0,10,100,1000,10000,100000,1000000]
alphas_tim = [100]#[0,1,10,50,100,200,300,400,500,600,700,800,900,1000]
lmds = [0.9]#[0.9,0.8,0.7,0.6,0.5,0.3,0.2,0.1,0.05,0.01,0.001]
n_queries = [5,3]#,3,1]
k_shots = [5,3,1]#,3,1]
n_ways_effs = [1]

#uniq_classes = sorted(list(set(enroll_dict['concat_labels'])))
uniq_classes = sorted(list(set(merged_dict['concat_labels'])))


for k_shot in k_shots:
    for n_ways_eff in n_ways_effs:
        for n_query in n_queries:

            alphas = [n_query]

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
            #test_embs = embedding_normalize(test_embs,use_mean=False)#True)
            #enroll_embs = embedding_normalize(enroll_embs,use_mean=False)#True)

            acc = {}
            acc["ss"] = []
            acc["smv"] = []
            acc["sscd"] = []
            acc["sscd_5"] = [] 
            acc["fsaic"] = []
            acc["fsaic_centroid"] = []
            acc["fsaic_centroid_5"] = []
            acc['hard_em_dirichlet'] = []
            
            
            acc["paddle"] = {}
            acc['paddle_2stage'] = {}
            acc['tim'] = {}
            acc['tim_2stage'] = {}
            acc['laplacianshot'] = {}
            acc['laplacianshot_2stage'] = {}
            
            for alpha in alphas:
                acc["paddle"][str(alpha)] = []
            for alpha_glasso in alphas_glasso:
                acc['paddle_2stage'][str(alpha_glasso)] = []
            for alpha in alphas_tim:
                acc["tim"][str(alpha)] = []
            for alpha in alphas_tim:
                acc["tim_2stage"][str(alpha)] = []
            for lmd in lmds:
                acc["laplacianshot"][str(lmd)] = []
            for lmd in lmds:
                acc["laplacianshot_2stage"][str(lmd)] = []
                

            for start in tqdm(range(0,n_tasks,batch_size)):
                end = (start+batch_size) if (start+batch_size) <= n_tasks else n_tasks

                x_q,y_q,x_s,y_s = (test_embs[start:end],
                                test_labels[start:end],
                                enroll_embs[start:end],
                                enroll_labels[start:end])

                eval = Simpleshot(avg="mean",backend="L2",method="ss")
                acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                acc["ss"].extend(acc_list)
                
                if n_ways_eff == 1:
                    eval = Simpleshot(avg="mean",backend="L2",method="smv")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc["smv"].extend(acc_list)

                    eval = FSAIC(method="normal")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc["fsaic"].extend(acc_list)
                
                    eval = FSAIC(method="centroid")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end])
                    acc["fsaic_centroid"].extend(acc_list)
                    acc["fsaic_centroid_5"].extend(acc_list_5)
                    
                    eval = Simpleshot(avg="mean",backend="L2",method="sscd")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc['sscd'].extend(acc_list)
                    acc['sscd_5'].extend(acc_list_5)
                
                else:
                    eval = Simpleshot(avg="mean",backend="cosine",method="ss")
                    acc_list,_,_ = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc["ss"].extend(acc_list)
                """
                args ={}
                args['maj_vote']=True
                args['iter'] = 10
                args['num_classes_test'] = 5
                args['k_eff'] = n_ways_eff
                args['n_query'] = n_query
                args['iter_mm'] = 1000
                args['use_softmax_feature'] = True
                method_info = {'device':'cuda','args':args}
                acc_list,_ = run_em_dirichlet(x_s, y_s, x_q, y_q,method_info)              
                acc['hard_em_dirichlet'].extend(acc_list)
                """
                
                for alpha in alphas:
                    print("Doing PADDLE")
                    args = {}
                    
                    if n_ways_eff == 1:
                        args['maj_vote'] = True
                    else:
                        args['maj_vote'] = False

                    args['alpha'] = alpha
                    args['iter'] = 30
                    method_info = {'device':'cuda','args':args}
                    acc_list,_ = run_method(x_s, y_s, x_q, y_q,method_info,'paddle')
                    acc['paddle'][str(alpha)].extend(acc_list)
                
                for lmd in lmds:
                    print("Doing LaplacianShot")
                    args ={}
                    args['maj_vote']=True
                    args['knn'] = 3
                    args['lmd'] = lmd
                    args['norm_type'] = 'L2N'
                    args['iter'] = 100
                    args['batch_size'] = batch_size
                    method_info = {'device':'cuda','args':args}
                    acc_list, preds_q = run_method(x_s, y_s, x_q, y_q,method_info,'laplacianshot')              
                    acc['laplacianshot'][str(lmd)].extend(acc_list)
                    acc_list = run_2stage_method(x_s, y_s, x_q, y_q, test_audios[start:end], method_info,'laplacianshot',top_k)                
                    acc['laplacianshot_2stage'][str(lmd)].extend(acc_list)
                    
                for alpha_tim in alphas_tim:
                    print("Doing Alpha-TIM")
                    args = {}
                    args['maj_vote'] = True
                    args['lr_alpha_tim'] = 1e-4
                    args['temp'] = 15
                    args['loss_weights'] = [1.0, 1.0, 1.0]
                    args['iter'] = 1000
                    args['entropies'] = ['Shannon', 'Alpha', 'Alpha']
                    args['alpha'] = alpha_tim
                    method_info = {'device':'cuda','args':args}
                    acc_list,_ = run_method(x_s, y_s, x_q, y_q,method_info,method='tim_alpha')                
                    acc['tim'][str(alpha_tim)].extend(acc_list)
                    
                    acc_list = run_2stage_method(x_s, y_s, x_q, y_q, test_audios[start:end], method_info,'tim_alpha',top_k)                
                    acc['tim_2stage'][str(alpha_tim)].extend(acc_list)
                
                """ 
                if k_shot == 1:
                    continue
                for alpha_glasso in alphas_glasso:
                    args = {}
                    if n_ways_eff == 1:
                        args['maj_vote'] = True
                    else:
                        args['maj_vote'] = False
                    
                    args['alpha'] = alpha_glasso
                    method_info = {'device':'cuda','args':args}
                    start_time = time.time()
                    if n_ways_eff == 1:
                        acc_list = run_2stage_method(x_s, y_s, x_q, y_q, test_audios[start:end], method_info,'glasso', top_k)                
                        acc['paddle_2stage'][str(alpha_glasso)].extend(acc_list)
                    dur = time.time() - start_time
                """
                
            final_json = {}
            final_json['ss'] = 100*sum(acc["ss"])/len(acc["ss"])
            final_json['smv'] = 100*sum(acc["smv"])/len(acc["smv"])
            final_json['sscd'] = 100*sum(acc["sscd"])/len(acc["sscd"])
            final_json['sscd_5'] = 100*sum(acc["sscd_5"])/len(acc["sscd_5"])
            final_json['fsaic'] = 100*sum(acc["fsaic"])/len(acc["fsaic"])
            final_json['fsaic_centroid'] = 100*sum(acc["fsaic_centroid"])/len(acc["fsaic_centroid"])
            final_json['fsaic_centroid_5'] = 100*sum(acc["fsaic_centroid_5"])/len(acc["fsaic_centroid_5"])
            #final_json['hard_em_dirichlet'] = 100*sum(acc["hard_em_dirichlet"])/len(acc["hard_em_dirichlet"])
            
            final_json['paddle'] = {}
            #final_json['paddle_2stage'] = {}
            final_json['tim'] = {}
            final_json['tim_2stage'] = {}
            final_json['laplacianshot'] = {}
            final_json['laplacianshot_2stage'] = {}    

            for alpha in alphas:
                final_json['paddle'][str(alpha)] = 100*sum(acc["paddle"][str(alpha)])/len(acc["paddle"][str(alpha)])
            for alpha in alphas_tim:
                final_json['tim'][str(alpha)] = 100*sum(acc["tim"][str(alpha)])/len(acc["tim"][str(alpha)])
                final_json['tim_2stage'][str(alpha)] = 100*sum(acc["tim_2stage"][str(alpha)])/len(acc["tim_2stage"][str(alpha)])
            for lmd in lmds:
                final_json['laplacianshot'][str(lmd)] = 100*sum(acc["laplacianshot"][str(lmd)])/len(acc["laplacianshot"][str(lmd)])
                final_json['laplacianshot_2stage'][str(lmd)] = 100*sum(acc["laplacianshot_2stage"][str(lmd)])/len(acc["laplacianshot_2stage"][str(lmd)])
                
            #for alpha in alphas_glasso:
            #    if k_shot == 1:
            #        continue
            #    final_json['paddle_2stage'][str(alpha)] = 100*sum(acc["paddle_2stage"][str(alpha)])/len(acc["paddle_2stage"][str(alpha)])

            with open(out_file,'w') as f:
                json.dump(final_json,f)