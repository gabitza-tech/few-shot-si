# SCRIPT TO TUNE PADDLE ALPHA

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
from methods.methods import run_method
import os 
from utils.utils import CL2N_embeddings
import sys

dataset_file = sys.argv[1]
merged_dict = np.load(dataset_file,allow_pickle=True)

out_dir = sys.argv[2]
print(out_dir)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

seed = 42
n_tasks = 200
batch_size = 200
top_k = 5

normalize = True
use_mean = False

alphas = [i for i in range(0,20)]

n_queries = [5,3,1]
k_shots = [5,3,1]
n_ways_effs = [1]

uniq_classes = sorted(list(set(merged_dict['concat_labels'])))

for k_shot in k_shots:
    for n_ways_eff in n_ways_effs:
        for n_query in n_queries:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

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

            test_embs, test_labels, test_audios, enroll_embs, enroll_labels, enroll_audios = task_generator.sampler_unified(merged_dict)
            
            if normalize:
                enroll_embs, test_embs = CL2N_embeddings(enroll_embs,test_embs,use_mean=use_mean)

            acc = {}
            acc["paddle"] = {}
            
            for alpha in alphas:
                acc["paddle"][str(alpha)] = []
            
            for start in tqdm(range(0,n_tasks,batch_size)):
                end = (start+batch_size) if (start+batch_size) <= n_tasks else n_tasks

                x_q,y_q,x_s,y_s = (test_embs[start:end],
                                test_labels[start:end],
                                enroll_embs[start:end],
                                enroll_labels[start:end])

                
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
                
            final_json = {}
            final_json['paddle'] = {}
            
            for alpha in alphas:
                final_json['paddle'][str(alpha)] = 100*sum(acc["paddle"][str(alpha)])/len(acc["paddle"][str(alpha)])
            
            with open(out_file,'w') as f:
                json.dump(final_json,f)
