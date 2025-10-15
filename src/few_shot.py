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
out_dir = sys.argv[2]
use_mean = sys.argv[3]
k_shots = [int(sys.argv[4])]
seed = int(sys.argv[5])

# Few-Shot Tasks Parameters
n_tasks = 10000
batch_size = 50
args={}
n_queries =[5,3,1]
n_ways_effs = [1]


# Load already extracted embeddings
merged_dict = np.load(dataset_file,allow_pickle=True)
# Create output dir if it doesn't exist already
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

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

            acc = {}
            acc["ss"] = []
            acc["smv"] = []
            acc["fsaic"] = []
            
            # Paddle methods evaluated
            acc['paddle_maj'] = {} 
            acc['paddle_maj'][str(alpha)] = []
            
            # Output file in which results will be saved for this configuration
            out_filename = f'k_{k_shot}_neff_{n_ways_eff}_nq_{n_query}.json'
            out_file = os.path.join(out_dir,out_filename)
            
            # Class that generates the tasks
            task_generator = Tasks_Generator(uniq_classes=uniq_classes,
                                                n_tasks=n_tasks,
                                                n_ways=len(uniq_classes),
                                                n_ways_eff=n_ways_eff,
                                                n_query=n_query,
                                                k_shot=k_shot,
                                                seed=seed)

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
                # Inductive SimpleShot
                eval = Simpleshot(avg="mean",backend="L2",method="ss")
                acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                acc["ss"].extend(acc_list)
                
                # SimpleShot Majority Vote
                eval = Simpleshot(avg="mean",backend="L2",method="smv")
                acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                acc["smv"].extend(acc_list)
            
                # FSAIC method
                eval = FSAIC()
                acc_list, acc_list_5, pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end])
                acc["fsaic"].extend(acc_list)
            
                # Paddle with Majority Vote
                args['iter']=30
                args['maj_vote'] = True
                args['alpha'] = alpha
                method_info = {'device':'cuda','args':args}
                acc_list,_ = run_method(x_s, y_s, x_q, y_q,method_info,'paddle')
                acc['paddle_maj'][str(alpha)].extend(acc_list)
                
                # Get duration of task
                duration = time.time() - start_time
                print(f'Durations is: {duration} s')
            

            final_json = {}
            final_json['ss'] = 100*sum(acc["ss"])/len(acc["ss"])
            final_json['smv'] = 100*sum(acc["smv"])/len(acc["smv"])
            final_json['fsaic'] = 100*sum(acc["fsaic"])/len(acc["fsaic"])
            
            final_json['paddle_maj'] = {}
            final_json['paddle_maj'][str(alpha)] = 100*sum(acc["paddle_maj"][str(alpha)])/len(acc["paddle_maj"][str(alpha)])
            
            with open(out_file,'w') as f:
                json.dump(final_json,f)