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
from methods.fsaic_tunable import FSAIC_tunable
from methods.methods import run_2stage_method,run_method
import os 
from utils.utils import CL2N_embeddings,embedding_normalize,embs_norm_both
import sys

#query_file = 'embeddings_vox2/voxmovies_257.pkl'
#support_file = 'embeddings_vox2/voxceleb1_257.pkl'
#test_dict = np.load(query_file, allow_pickle=True)
#enroll_dict = np.load(support_file, allow_pickle=True)   
dataset_file = sys.argv[1]
merged_dict = np.load(dataset_file,allow_pickle=True)

out_dir = sys.argv[2]
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

seed = int(sys.argv[5])

n_tasks = 200
batch_size = 200

args={}
args['iter']=30

use_mean = sys.argv[3]

n_queries =[5,3,1]
k_shots = [int(sys.argv[4])]
n_ways_effs = [1]

#uniq_classes = sorted(list(set(enroll_dict['concat_labels'])))
uniq_classes = sorted(list(set(merged_dict['concat_labels'])))

for k_shot in k_shots:
    for n_ways_eff in n_ways_effs:
        for n_query in n_queries:
            print(f"Seed:{seed},Kshot:{k_shot},n_query:{n_query}")            
            
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            # we set alpha value to the number of samples in query
            alpha = n_query
            alpha_glasso = 1000

            acc = {}
            acc["ss"] = []
            acc["smv"] = []
            acc["sscd"] = []
            acc["sscd_5"] = [] 

            acc["fsaic"] = []
            acc["fsaic_centroid"] = []
            acc["fsaic_centroid_5"] = []
            #acc["normal_mahalanobis_latesum"] = []
            #acc["normal_mahalanobis_latesum_5"] = []
            #acc["mahalanobis_latesum"] = []
            #acc["mahalanobis_latesum_5"] = []
            acc["mahalanobis_cd_latesum"] = []
            acc["mahalanobis_cd_latesum_5"] = []
            acc["mahalanobis_cd_latesum_wsq"] = []
            acc["mahalanobis_cd_latesum_wsq_5"] = []

            # Paddle methods evaluated
            acc["paddle"] = {}
            acc['paddle_maj'] = {} 
            acc['paddle_2stage'] = {}
            acc["paddle"][str(alpha)] = []
            acc['paddle_maj'][str(alpha)] = []
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

            # Sample from support and query the tasks
            #test_embs, test_labels, test_audios = task_generator.sampler(test_dict,mode='query')
            #enroll_embs, enroll_labels, enroll_audios = task_generator.sampler(enroll_dict,mode='support') 
            test_embs, test_labels, test_audios, enroll_embs, enroll_labels, enroll_audios = task_generator.sampler_unified(merged_dict) 
            # Normalize the extracted embeddings
            enroll_embs, test_embs = CL2N_embeddings(enroll_embs,test_embs,use_mean=use_mean)
            
            for start in tqdm(range(0,n_tasks,batch_size)):
                end = (start+batch_size) if (start+batch_size) <= n_tasks else n_tasks

                x_q,y_q,x_s,y_s = (test_embs[start:end],
                                test_labels[start:end],
                                enroll_embs[start:end],
                                enroll_labels[start:end])
                
                start_time = time.time()
                eval = Simpleshot(avg="mean",backend="L2",method="ss")
                acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                acc["ss"].extend(acc_list)
                
                if n_ways_eff == 1:
                    eval = Simpleshot(avg="mean",backend="L2",method="smv")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc["smv"].extend(acc_list)
                    
                    eval = Simpleshot(avg="mean",backend="L2",method="sscd")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc['sscd'].extend(acc_list)
                    acc['sscd_5'].extend(acc_list_5)

                    #eval = FSAIC(method="normal")
                    #acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    #acc["fsaic"].extend(acc_list)
                
                    #eval = FSAIC(method="centroid")
                    #acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end])
                    #acc["fsaic_centroid"].extend(acc_list)
                    #acc["fsaic_centroid_5"].extend(acc_list_5)
                    
                    eval = FSAIC(method="mahalanobis_cd_latesum")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end])
                    acc["mahalanobis_cd_latesum"].extend(acc_list)
                    acc["mahalanobis_cd_latesum_5"].extend(acc_list_5)
                    

                    eval = FSAIC(method="mahalanobis_cd_latesum_wsq")
                    acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end])
                    acc["mahalanobis_cd_latesum_wsq"].extend(acc_list)
                    acc["mahalanobis_cd_latesum_wsq_5"].extend(acc_list_5)
                    
                else:
                    eval = Simpleshot(avg="mean",backend="cosine",method="ss")
                    acc_list,_,_ = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                    acc["ss"].extend(acc_list)

                #args['maj_vote'] = False
                #args['alpha'] = alpha
                #method_info = {'device':'cpu','args':args}
                #acc_list,_ = run_method(x_s, y_s, x_q, y_q,method_info,'paddle')  
                #acc['paddle'][str(alpha)].extend(acc_list)

                if n_ways_eff == 1:
                    args['maj_vote'] = True
                else:
                    args['maj_vote'] = False

                
                args['alpha'] = alpha
                method_info = {'device':'cuda','args':args}
                acc_list,_ = run_method(x_s, y_s, x_q, y_q,method_info,'paddle')
                acc['paddle_maj'][str(alpha)].extend(acc_list)
                duration = time.time() - start_time
                print(f'Durations is: {duration} s')
                continue
                args['alpha'] = alpha_glasso
                method_info = {'device':'cuda','args':args}
                if n_ways_eff == 1:
                    try:
                        acc_list = run_2stage_method(x_s, y_s, x_q, y_q, test_audios[start:end], method_info,'glasso', top_k)
                        acc['paddle_2stage'][str(alpha_glasso)].extend(acc_list)
                    except:
                        continue
            

            final_json = {}
            final_json['ss'] = 100*sum(acc["ss"])/len(acc["ss"])
            final_json['smv'] = 100*sum(acc["smv"])/len(acc["smv"])
            final_json['sscd'] = 100*sum(acc["sscd"])/len(acc["sscd"])
            final_json['sscd_5'] = 100*sum(acc["sscd_5"])/len(acc["sscd_5"])
            #final_json['fsaic'] = 100*sum(acc["fsaic"])/len(acc["fsaic"])
            #final_json['fsaic_centroid'] = 100*sum(acc["fsaic_centroid"])/len(acc["fsaic_centroid"])
            #final_json['fsaic_centroid_5'] = 100*sum(acc["fsaic_centroid_5"])/len(acc["fsaic_centroid_5"])
            final_json['mahalanobis_cd_latesum'] = 100*sum(acc["mahalanobis_cd_latesum"])/len(acc["mahalanobis_cd_latesum"])
            final_json['mahalanobis_cd_latesum_5'] = 100*sum(acc["mahalanobis_cd_latesum_5"])/len(acc["mahalanobis_cd_latesum_5"])
            final_json['mahalanobis_cd_latesum_wsq'] = 100*sum(acc["mahalanobis_cd_latesum_wsq"])/len(acc["mahalanobis_cd_latesum_wsq"])
            final_json['mahalanobis_cd_latesum_wsq_5'] = 100*sum(acc["mahalanobis_cd_latesum_wsq_5"])/len(acc["mahalanobis_cd_latesum_5"])
            
            #final_json['paddle'] = {}
            #final_json['paddle'][str(alpha)] = 100*sum(acc["paddle"][str(alpha)])/len(acc["paddle"][str(alpha)])

            final_json['paddle_maj'] = {}
            final_json['paddle_maj'][str(alpha)] = 100*sum(acc["paddle_maj"][str(alpha)])/len(acc["paddle_maj"][str(alpha)])
            with open(out_file, 'w') as f:
                json.dump(final_json,f)

            try:
                final_json['paddle_2stage'] = {}
                final_json['paddle_2stage'][str(alpha_glasso)] = 100*sum(acc['paddle_2stage'][str(alpha_glasso)])/len(acc['paddle_2stage'][str(alpha_glasso)])
            except:
                continue

            with open(out_file,'w') as f:
                json.dump(final_json,f)
