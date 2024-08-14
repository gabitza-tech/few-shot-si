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
from methods.methods import run_paddle_new
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
from sklearn.manifold import TSNE
from utils.utils import tsne_query_support
from numpy.linalg import norm


out_name = sys.argv[1]
out_dir = f'voxmovies_normalized_diagonal_validation'
out_file = os.path.join(out_dir,f'{out_name}.json')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

#query_file = 'saved_embs/voxceleb1_3s/voxceleb1_3s_query_ecapa_embs.pkl'#voxmovies_3s_ecapa_embs.pkl'#'datasets_splits/embeddings/voxmovies_3s_ecapa_embs_257.pkl'
query_file = 'saved_embs/voxmovies_3s/voxmovies_3s_ecapa_embs.pkl'
support_file = 'saved_embs/voxceleb1_3s/voxceleb1_3s_support_ecapa_embs.pkl'

#classes_file = 'datasets_splits/voxmovies_257_labels.txt'#'datasets_splits/voxmovies_257_labels.txt'#'datasets_splits/voxceleb1_test_labels.txt'#
classes_file = 'datasets_splits/voxmovies_domain_adapt.txt'

test_dict = np.load(query_file, allow_pickle=True)
enroll_dict = np.load(support_file, allow_pickle=True)   

with open(classes_file,'r') as f:
    lines = f.readlines()
    orig_ids = []
    for line in lines: 
        orig_ids.append(line.strip())

no_classes = len(orig_ids)
split_size = int(len(orig_ids)/10)

thetas = [0.001,0.01,0.1,1,10,100,1000]
thetas.append("no_adapt")

final_json = {}
final_json['simpleshot'] = {}
final_json['simpleshot_5'] = {} 

temp_json = {}
temp_json['simpleshot'] = {}
temp_json['simpleshot_5'] = {} 
for theta in thetas:
    temp_json['simpleshot'][str(theta)] = []
    temp_json['simpleshot_5'][str(theta)] = []

n_tasks = 10000
batch_size = 1000


for i in range(0,len(orig_ids),split_size):

    start = i
    if start+ split_size >= len(orig_ids):
        break
    else:
        end = start+split_size
    
    # Split the FOLD for cross-validation
    test_ids = orig_ids.copy()[start:end]
    ids = orig_ids.copy()
    for label in test_ids:
        ids.remove(label)

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
        
    print(movies_feat.shape)
    print(celeb_feat.shape)
    print(movies_dict_test['concat_features'].shape)



    start = time.time()

    seed = 42
    normalize = True
    use_mean = True
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_queries = [1]
    k_shots = [1,3,5]
    n_ways_effs = [1]
    
    if normalize == True:
        #movies_feat,celeb_feat = CL2N_embeddings(np.expand_dims(movies_feat,0),np.expand_dims(celeb_feat,0),normalize,use_mean=True)   
        movies_feat = embedding_normalize(movies_feat,use_mean=use_mean)
        celeb_feat = embedding_normalize(celeb_feat,use_mean=use_mean)
        movies_feat = np.squeeze(movies_feat)
        celeb_feat = np.squeeze(celeb_feat)

    uniq_classes = sorted(list(set(movies_dict_test['concat_labels'])))

    for k_shot in k_shots:
        for n_ways_eff in n_ways_effs:
            for n_q in n_queries:
                            
                
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
                    
                    test_embs= np.copy(initial_test_embs)
                    enroll_embs = np.copy(initial_enroll_embs)
                    #A_matrix = np.zeros((192,192))
                    if theta != "no_adapt":
                        # Iterative A
                        # Voxceleb closer to VoxMovies     
                        """sampled_classes=sorted(list(set(celeb_labels)))
                        sampled_classes_dict = {label:i for i,label in enumerate(sampled_classes)}
                        print('--')
                        mu = calculate_centroids(movies_feat,movies_labels)#(celeb_feat,celeb_labels)#
                        processed_labels_q = []
                        for label in celeb_labels:#movies_labels:#
                            processed_labels_q.append(sampled_classes_dict[label])
                        processed_labels_q = np.array(processed_labels_q)    
                        mu_N = []
                        for label in processed_labels_q:
                            mu_N.append(mu[label])
                        mu_N = np.array(mu_N)
                        print('!!')
                        """
                        # Coral adaptation
                        A = coral(movies_feat,celeb_feat,theta)
                        test_embs = test_embs @ A.astype(np.float32).T
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

                    acc_simple = 100*sum(acc["simpleshot"])/len(acc["simpleshot"])
                    
                    acc_simple5 = 100*sum(acc["simpleshot_5"])/len(acc["simpleshot_5"])
                    temp_json['simpleshot'][str(theta)].append(acc_simple)
                    temp_json['simpleshot_5'][str(theta)].append(acc_simple5)

for theta in thetas:
    final_json['simpleshot'][str(theta)] = sum(temp_json['simpleshot'][str(theta)])/len(temp_json['simpleshot'][str(theta)])
    final_json['simpleshot_5'][str(theta)] = sum(temp_json['simpleshot_5'][str(theta)])/len(temp_json['simpleshot_5'][str(theta)])

with open(out_file,'w') as f:
    json.dump(final_json,f)
