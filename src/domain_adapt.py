import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from collections import Counter
import json
import time
from utils.task_generator import Tasks_Generator
from tqdm import tqdm
from methods.simpleshot import Simpleshot,compute_acc
from methods.methods import run_paddle_new, run_2stage_paddle
import os 
from utils.utils import save_pickle
from utils.utils import CL2N_embeddings, embedding_normalize
from utils.paddle_utils import get_log_file,Logger
from numpy.linalg import norm
from utils.utils import coral,calculate_centroids,ana_A,iterative_A,plot_embeddings,class_compute_transform_A,class_compute_diagonal_A,class_compute_sums_A
import sys
import random
import torch
import sys
import matplotlib.pyplot as plt

method = sys.argv[1]

#query_file = 'saved_embs/voxceleb1_3s/voxceleb1_3s_query_ecapa_embs.pkl'#voxmovies_3s_ecapa_embs.pkl'#'datasets_splits/embeddings/voxmovies_3s_ecapa_embs_257.pkl'
query_file = 'saved_embs/voxmovies_3s/voxmovies_3s_ecapa_embs.pkl'
support_file = 'saved_embs/voxceleb1_3s/voxceleb1_3s_support_ecapa_embs.pkl'

classes_file = 'datasets_splits/voxmovies_domain_adapt.txt'#'datasets_splits/voxmovies_257_labels.txt'#'datasets_splits/voxmovies_257_labels.txt'#'datasets_splits/voxmovies_domain_adapt.txt'#'datasets_splits/voxceleb1_test_labels.txt'#
test_classes_file = 'datasets_splits/voxmovies_257_labels.txt'##'datasets_splits/voxmovies_257_labels.txt'#

test_dict = np.load(query_file, allow_pickle=True)
enroll_dict = np.load(support_file, allow_pickle=True)   

with open(classes_file,'r') as f:
    lines = f.readlines()
    ids = []
    for line in lines: 
        ids.append(line.strip())


with open(test_classes_file,'r') as f:
    lines = f.readlines()
    test_ids = []
    for line in lines:
        test_ids.append(line.strip())

labels_count = []
indices_celeb = []
indices_movies = []

indices_celeb_test = []
indices_movies_test = []

for i,label in enumerate(test_dict['concat_labels']):
    if label in ids:
        indices_movies.append(i)
    if label in test_ids:
        indices_movies_test.append(i)

for i,label in enumerate(enroll_dict['concat_labels']):
    if label in ids:
        indices_celeb.append(i)
    if label in test_ids:
        indices_celeb_test.append(i)

print(len(indices_movies))
print(len(indices_celeb))

celeb_dict = {}
celeb_dict['concat_features'] = enroll_dict['concat_features'][indices_celeb]
celeb_feat = celeb_dict['concat_features']
celeb_dict['concat_labels'] = np.array(enroll_dict['concat_labels'])[indices_celeb]
celeb_labels = celeb_dict['concat_labels']
celeb_dict['concat_slices'] = np.array(enroll_dict['concat_slices'])[indices_celeb]
celeb_dict['concat_patchs'] = np.array(enroll_dict['concat_patchs'])[indices_celeb]

celeb_dict_test = {}
celeb_dict_test['concat_features'] = enroll_dict['concat_features'][indices_celeb_test]
celeb_dict_test['concat_labels'] = np.array(enroll_dict['concat_labels'])[indices_celeb_test]
celeb_dict_test['concat_slices'] = np.array(enroll_dict['concat_slices'])[indices_celeb_test]
celeb_dict_test['concat_patchs'] = np.array(enroll_dict['concat_patchs'])[indices_celeb_test]

movies_dict = {}
movies_dict['concat_features'] = test_dict['concat_features'][indices_movies]
movies_feat = movies_dict['concat_features']
movies_dict['concat_labels'] = np.array(test_dict['concat_labels'])[indices_movies]
movies_labels = movies_dict['concat_labels']
movies_dict['concat_slices'] = np.array(test_dict['concat_slices'])[indices_movies]
movies_dict['concat_patchs'] = np.array(test_dict['concat_patchs'])[indices_movies]

movies_dict_test = {}
movies_dict_test['concat_features'] = test_dict['concat_features'][indices_movies_test]
movies_dict_test['concat_labels'] = np.array(test_dict['concat_labels'])[indices_movies_test]
movies_dict_test['concat_slices'] = np.array(test_dict['concat_slices'])[indices_movies_test]
movies_dict_test['concat_patchs'] = np.array(test_dict['concat_patchs'])[indices_movies_test]
    
out_dir = f'voxmovies_normalized_diagonal_validation_{method}'#"voxmovies_test_no_normalization"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

start = time.time()

seed = 42
n_tasks = 200
batch_size = 10

args={}
args['iter']=20

normalize = True
use_mean = True

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

alpha_paddle = 1
alpha_glasso = 1000
n_q = 3
k_shot = 3
n_ways_eff = 1

out_file = os.path.join(out_dir,f'k_{k_shot}_neff_{n_ways_eff}_nq_{n_q}.json')

thetas = [0.001,0.01,0.1,1,10,100,1000]
#thetas = [0.001,0.005,0.01,0.015,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,1]
#thetas = [1000]
#thetas = []
thetas.append("no_adapt")

final_json = {}
final_json['simpleshot'] = {}
final_json['simpleshot_5'] = {}
final_json['paddle'] = {}
final_json['2stage_paddle'] = {}
    
acc = {}
acc["simpleshot"] = []
acc["paddle"] = []
acc["2stage_paddle"] = []
start = time.time()

print(movies_feat.shape)
print(celeb_feat.shape)

if normalize == True:
    #movies_feat,celeb_feat = CL2N_embeddings(np.expand_dims(movies_feat,0),np.expand_dims(celeb_feat,0),normalize,use_mean=True)   
    movies_feat = embedding_normalize(movies_feat,use_mean=use_mean)
    celeb_feat = embedding_normalize(celeb_feat,use_mean=use_mean)
    movies_feat = np.squeeze(movies_feat)
    celeb_feat = np.squeeze(celeb_feat)
       
dur = time.time()-start
print(f"Time taken to compute A:{dur} seconds")

uniq_classes = sorted(list(set(movies_dict_test['concat_labels'])))

print(len(set(movies_dict_test['concat_labels'])))
task_generator = Tasks_Generator(uniq_classes=uniq_classes,
                                n_tasks=n_tasks,
                                n_ways=len(set(movies_dict_test['concat_labels'])),
                                n_ways_eff=n_ways_eff,
                                n_query=n_q,
                                k_shot=k_shot,
                                seed=seed)

test_embs, test_labels, test_audios = task_generator.sampler(movies_dict_test,mode='query')
enroll_embs, enroll_labels, enroll_audios = task_generator.sampler(celeb_dict_test,mode='support')

if normalize == True:
    #enroll_embs, initial_test_embs = CL2N_embeddings(enroll_embs,test_embs,normalize,use_mean=True)    
    initial_test_embs = embedding_normalize(test_embs,use_mean=use_mean)
    initial_enroll_embs = embedding_normalize(enroll_embs,use_mean=use_mean)
else:
    initial_test_embs = np.copy(test_embs)
    initial_test_embs = np.copy(enroll_embs)

for theta in thetas:
    acc = {}
    acc["simpleshot"] = []
    acc["simpleshot_5"] = []
    acc["paddle"] = []
    acc["2stage_paddle"] = []
    all_pred_labels_5 = []

    test_embs= np.copy(initial_test_embs)
    enroll_embs = np.copy(initial_enroll_embs)
    #A_matrix = np.zeros((192,192))
    if theta != "no_adapt":
        # Iterative A
        # Voxceleb closer to VoxMovies
        #sampled_classes=sorted(list(set(celeb_labels)))
        #sampled_classes_dict = {label:i for i,label in enumerate(sampled_classes)}
        print('--')
        mu_N = []
        for label in celeb_labels:
            indices = np.where(movies_labels == label)
            #embedding = (movies_feat[indices[0]].sum(axis=0).squeeze()) / len(indices[0])
            embedding = np.median(movies_feat[indices[0]], axis=0)
            mu_N.append(embedding)
        mu_N = np.array(mu_N)
        print('!!')
        
        #mu = calculate_centroids(movies_feat,movies_labels)#(celeb_feat,celeb_labels)#
        #processed_labels_q = []
        #for label in celeb_labels:#movies_labels:#
        #    processed_labels_q.append(sampled_classes_dict[label])
        #processed_labels_q = np.array(processed_labels_q)    
        #mu_N = []
        #for label in processed_labels_q:
        #    mu_N.append(mu[label])
        #mu_N = np.array(mu_N)
        
        A, As, crit = iterative_A(celeb_feat,mu_N,alpha=theta)
        #A, As, crit = iterative_A(movies_feat,mu_N,alpha=theta)
        #A, As, crit = ana_A(celeb_feat.T,mu_N.T,lambda_=theta)
        enroll_embs = enroll_embs @ A.astype(np.float32).T
        #test_embs = test_embs @ A.astype(np.float32).T
        # Coral adaptation
        #A = coral(movies_feat,celeb_feat,theta)
        #test_embs = test_embs @ A.astype(np.float32).T
        #A = coral(celeb_feat,movies_feat,theta)
        #enroll_embs = enroll_embs @ A.astype(np.float32).T
              
    for start in tqdm(range(0,n_tasks,batch_size)):
        end = (start+batch_size) if (start+batch_size) <= n_tasks else n_tasks
        
        x_q,y_q,x_s,y_s = (test_embs[start:end],
                        test_labels[start:end],
                        enroll_embs[start:end],
                        enroll_labels[start:end])
        
        if n_ways_eff == 1:
            eval = Simpleshot(avg="mean",backend="L2",method="transductive_centroid")
            acc_list,acc_list_5,pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
        else:
            eval = Simpleshot(avg="mean",backend="L2",method="inductive")
            acc_list = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 

        acc["simpleshot"].extend(acc_list)
        acc["simpleshot_5"].extend(acc_list_5)
        all_pred_labels_5.extend(pred_labels_5)

        if n_ways_eff == 1:
            args['maj_vote'] = True
        else:
            args['maj_vote'] = False

        args['alpha'] = alpha_paddle
        method_info = {'device':'cuda:0','args':args}#'log_file':log_file,'args':args}
        acc_list_paddle,preds_q = run_paddle_new(x_s, y_s, x_q, y_q,method_info,'paddle')
        acc["paddle"].extend(acc_list_paddle)
        continue
        args['alpha'] = alpha_glasso
        method_info = {'device':'cuda','args':args}
        if n_ways_eff == 1:
            acc_list = run_2stage_paddle(x_s, y_s, x_q, y_q, test_audios[start:end], method_info)                
            acc["2stage_paddle"].extend(acc_list)        
    
    
    final_json['simpleshot'][str(theta)] = 100*sum(acc["simpleshot"])/len(acc["simpleshot"])
    final_json['simpleshot_5'][str(theta)] = 100*sum(acc["simpleshot_5"])/len(acc["simpleshot_5"])
    final_json['paddle'][str(theta)] = 100*sum(acc["paddle"])/len(acc["paddle"])
    #final_json['2stage_paddle'][str(theta)] = 100*sum(acc["2stage_paddle"])/len(acc["2stage_paddle"])

    #print(len(acc['simpleshot']))
    #print(len(acc['simpleshot_5']))
    #print(len(acc['paddle']))
    #print(len(stage2_acc_list))

with open(out_file,'w') as f:
    json.dump(final_json,f)
