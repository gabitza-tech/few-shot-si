import os
import numpy as np
from tqdm import tqdm
import torch
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval, get_one_hot
from methods.paddle import PADDLE
from methods.paddle_glasso import PADDLE_GLASSO
import torch.nn.functional as F
from methods.simpleshot import Simpleshot, compute_acc

def simpleshot_inductive(enroll_embs,enroll_labels,test_embs,avg="mean",backend="l2"):

    """
    enroll_embs: [n_tasks,k_shot*n_ways,192]
    enroll_labels: [n_tasks,k_shot*n_ways]
    test_embs: [n_tasks,n_query,192]
    test_labels: [n_tasks,n_query]
    """
    print("Using SimpleShot method")
    # Calculate the mean embeddings for each class in the support

    # sampled_classes: [n_tasks,n_ways]
    sampled_classes=[]
    for task in enroll_labels:
        sampled_classes.append(sorted(list(set(task))))

    avg_enroll_embs = []
    for i,task_classes in enumerate(sampled_classes):
        task_enroll_embs = []
        for label in task_classes:
            indices = np.where(enroll_labels[i] == label)
            if avg == "mean":
                embedding = (enroll_embs[i][indices].sum(axis=0).squeeze()) / len(indices[0])
            if avg == "median":
                embedding = np.median(enroll_embs[i][indices], axis=0)
            task_enroll_embs.append(embedding)
        avg_enroll_embs.append(task_enroll_embs)

    avg_enroll_embs = np.asarray(avg_enroll_embs)

    
    if backend == "cosine":
        print("Using cosine similarity")
        scores = np.matmul(test_embs, avg_enroll_embs.T)
        pred_labels = scores.argmax(axis=-1)
    else:
        print("Using L2 norm")
        test_embs = np.expand_dims(test_embs,2) # [n_tasks,n_query,1,emb_shape]
        avg_enroll_embs = np.expand_dims(avg_enroll_embs,1) # [n_tasks,1,1251,emb_shape]
        test_embs = torch.from_numpy(test_embs).float().to(torch.device('cpu'))
        avg_enroll_embs = torch.from_numpy(avg_enroll_embs).float().to(torch.device('cpu'))

        # Class distance
        dist = (test_embs-avg_enroll_embs)**2
        C_l = torch.sum(dist,dim=-1) # [n_tasks,n_query,1251]

        pred_labels = torch.argmin(C_l, dim=-1)#.tolist()
        _,pred_labels_top5 = torch.topk(C_l, k=5, dim=-1, largest=False)

    print(pred_labels)
    return pred_labels, pred_labels_top5

def run_paddle(enroll_embs,enroll_labels,test_embs,test_labels,k_shot,method_info):
    """
    label_dict = {}
    for i,label in enumerate(sampled_classes):
        label_dict[label]=i
    
    new_test_labels = []
    for label in test_labels:
        new_test_labels.append(label_dict[label])
    new_test_labels = np.asarray(new_test_labels)

    new_enroll_labels = []
    for label in enroll_labels:
        new_enroll_labels.append(label_dict[label])
    new_enroll_labels = np.asarray(new_enroll_labels)
    """
    acc_mean_list = []
    acc_conf_list = []

    if k_shot == 1:
        query_batch = 512
    elif k_shot == 3:
        query_batch = 256
    elif k_shot == 5:
        query_batch = 128
    else: # k_shot == 10
        query_batch = 50
    
    for j in tqdm(range(0,test_labels.shape[0],query_batch)):
    #for j in tqdm(range(test_labels.shape[0])):
        
        end = j+query_batch
        if end>test_labels.shape[0]-1:
            end = test_labels.shape[0]-1

        len_batch = end - j

        #x_q = torch.tensor([test_embs[j]]).unsqueeze(0)
        x_q = torch.tensor(test_embs[j:end]).unsqueeze(1)
        #y_q = torch.tensor([new_test_labels[j]]).long().unsqueeze(0).unsqueeze(2)
        y_q = torch.tensor([test_labels[j:end]]).long().view(-1,1).unsqueeze(2)
        #x_s = torch.tensor(enroll_embs).unsqueeze(0)
        x_s = torch.tensor(enroll_embs).unsqueeze(0).repeat(len_batch,1,1)
        #y_s = torch.tensor(new_enroll_labels).long().unsqueeze(0).unsqueeze(2)
        y_s = torch.tensor(enroll_labels).long().unsqueeze(0).unsqueeze(2).repeat(len_batch,1,1)

        task_dic = {}
        task_dic['y_s'] = y_s
        task_dic['y_q'] = y_q
        task_dic['x_s'] = x_s
        task_dic['x_q'] = x_q

        method = PADDLE(**method_info)
        logs = method.run_task(task_dic)
        acc_sample, _ = compute_confidence_interval(logs['acc'][:, -1])

        # Mean accuracy per batch
        acc_mean_list.append(acc_sample*len_batch)
        
    avg_acc_task = sum(acc_mean_list)/test_labels.shape[0]

    return avg_acc_task

def run_paddle_transductive(enroll_embs,enroll_labels,test_embs,test_labels,k_shot,method_info, batch_size):
    """
    This function predicts using the PADDLE algorithm over a SINGLE TASK. 
    We can also iterate over the query with a batch_size! 

    INPUT:
    test_embs: (Q_samples, n_patches_sample, feature_dim) shape, Q_samples represents the total number of query samples
    test_labels: (Q_samples,) shape, enrollment labels
    enroll_embs: (S_samples, feature_dim) shape, S_samples represents the total number of support samples
    enroll_labels: (S_samples,) shape, enrollment labels
    k_shot: depending on k_shot, we choose the batch-size
    method_info: arguments for paddle!

    INTERMEDIATE:
    x_q: [len_batch, n_patches_sample, feature_dim]
    y_q: [len_batch, n_patches_sample, 1] (OBSERVATION: it could also be [len_batch,1,1]])
    x_s: [len_batch, S_samples, feature_dim]
    y_s: [len_batch, S_samples, 1]

    RETURN:
    avg_acc_task: average accuracy over the task
    """

    """
    if k_shot == 1:
        query_batch = 512
    elif k_shot == 3:
        query_batch = 256
    elif k_shot == 5:
        query_batch = 128
    else: 
        query_batch = 50
    """  

    query_batch = int(batch_size)
    acc_mean_list = []
    acc_mean_list_top5 = []

    for j in tqdm(range(0,test_labels.shape[0],query_batch)):
        
        end = j+query_batch
        if end>test_labels.shape[0]:
            end = test_labels.shape[0]
        if end == j:
            end = j+1

        len_batch = end - j

        x_q = torch.tensor(test_embs[j:end])
        y_q = torch.tensor(test_labels[j:end]).long().unsqueeze(1).unsqueeze(2).repeat(1,x_q.shape[1],1) # It can also work with a shape of [len_batch,1,1] (repeat operation is not needed, but it is nicer and clearer in this way)
        x_s = torch.tensor(enroll_embs).unsqueeze(0)#.repeat(len_batch,1,1)
        y_s = torch.tensor(enroll_labels).long().unsqueeze(0).unsqueeze(2)#.repeat(len_batch,1,1)
        
        task_dic = {}
        task_dic['y_s'] = y_s
        task_dic['y_q'] = y_q
        task_dic['x_s'] = x_s
        task_dic['x_q'] = x_q

        
        method = PADDLE(**method_info)
        logs = method.run_task(task_dic)
        acc_sample, _ = compute_confidence_interval(logs['acc'][:, -1])
        #acc_sample_top5, _ = compute_confidence_interval(logs['acc_top5'][:, -1])
        
        #if acc_sample < 1:
        #    print(acc_sample)
        # Mean accuracy per batch
        #print(f"Accuracy of the sample/samples is:{acc_sample}\n")
        acc_mean_list.append(acc_sample*len_batch)
        #acc_mean_list_top5.append(acc_sample_top5*len_batch)
    
    avg_acc_task = 100*sum(acc_mean_list)/test_labels.shape[0]
    #avg_acc_task_top5 = 100*sum(acc_mean_list_top5)/test_labels.shape[0]
    
    return avg_acc_task#, avg_acc_task_top5

def run_algo_transductive(enroll_embs,enroll_labels,test_embs,test_labels, batch_size):
    """
    This function predicts using the PADDLE algorithm over a SINGLE TASK. 
    We can also iterate over the query with a batch_size! 

    INPUT:
    test_embs: (Q_samples, n_patches_sample, feature_dim) shape, Q_samples represents the total number of query samples
    test_labels: (Q_samples,) shape, enrollment labels
    enroll_embs: (S_samples, feature_dim) shape, S_samples represents the total number of support samples
    enroll_labels: (S_samples,) shape, enrollment labels
    k_shot: depending on k_shot, we choose the batch-size
    method_info: arguments for paddle!

    INTERMEDIATE:
    x_q: [len_batch, n_patches_sample, feature_dim]
    y_q: [len_batch, n_patches_sample, 1] (OBSERVATION: it could also be [len_batch,1,1]])
    x_s: [len_batch, S_samples, feature_dim]
    y_s: [len_batch, S_samples, 1]

    RETURN:
    avg_acc_task: average accuracy over the task
    """

    """
    if k_shot == 1:
        query_batch = 512
    elif k_shot == 3:
        query_batch = 256
    elif k_shot == 5:
        query_batch = 128
    else: 
        query_batch = 50
    """  
    acc_list = []
    acc_list_top5 = []

    query_batch = batch_size
    for j in tqdm(range(0,test_labels.shape[0],query_batch)):
    #for j in tqdm(range(test_labels.shape[0])):
        
        end = j+query_batch
        if end>test_labels.shape[0]-1:
            end = test_labels.shape[0]-1
        if end == j:
            end = j+1

        len_batch = end - j

        w_s = []
        w_sq = []
        z_s = []
        z_q = []
        labels = sorted(list(np.unique(enroll_labels)))
        
        for label in labels:
            
            indices = np.where(enroll_labels == label)

            #print(test_embs[j:end].sum(axis=1).shape)
            #print(enroll_embs[indices].sum(axis=0).shape)
            
            w_s_l = np.expand_dims(enroll_embs[indices].sum(axis=0),0) / len(indices[0])
            
            w_sq_l = (np.expand_dims(enroll_embs[indices].sum(axis=0),0)+test_embs[j:end].sum(axis=1)) / (len(indices[0])+test_embs[j:end].shape[1])
            
            #print(w_s_l.shape)
            #print(w_sq_l.shape)
            z_s.append(enroll_embs[indices])
            w_s.append(w_s_l)
            w_sq.append(w_sq_l)

            
        #print(test_embs.shape[1])
        z_s = np.expand_dims(np.array(z_s),0)
        w_s = np.expand_dims(np.asarray(w_s),0)
        z_q = np.expand_dims(test_embs[j:end],1)
        w_q = np.expand_dims(z_q.sum(axis=2)/z_q.shape[2],2)
        w_sq = np.expand_dims(np.transpose(np.asarray(w_sq),(1,0,2)),2)

        
        # Assuming w_sq, z_s, w_s, z_q are NumPy arrays or existing PyTorch tensors
        # Convert NumPy arrays to PyTorch tensors if needed
        w_sq = torch.from_numpy(w_sq).float()
        z_s = torch.from_numpy(z_s).float()
        w_s = torch.from_numpy(w_s).float()
        z_q = torch.from_numpy(z_q).float()
        w_q = torch.from_numpy(w_q).float()

        # Move tensors to GPU if available
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda:0')
        w_sq = w_sq.to(device)
        z_s = z_s.to(device)
        w_s = w_s.to(device)
        z_q = z_q.to(device)
        w_q = w_q.to(device)

        """"""
        # Calculate sum_s and sum_q in PyTorch
        sum_s = torch.sum((w_sq - z_s)**2, dim=2) - torch.sum((w_s - z_s)**2, dim=2)
        sum_q = torch.sum((w_sq - z_q)**2, dim=2)
        # Calculate C_l
        C_l = torch.sum(sum_s + sum_q, dim=2)
        pred_labels = torch.argmin(C_l, dim=-1)
        _,pred_labels_top5 = torch.topk(C_l, k=5, dim=1, largest=False)
        
        """
        #COS w_s w_q
        C_l = torch.matmul(w_q.squeeze().squeeze(),w_s.squeeze().squeeze().T)
        pred_labels = C_l.argmax(axis=-1)
        _,pred_labels_top5 = torch.topk(C_l, k=5, dim=1, largest=True)
        """
        """
        #dist w_s z_q or w_s w_q

        dist = ((w_s-w_q)**2).squeeze()
        

        C_l = torch.sum(dist,dim=2)
        pred_labels = torch.argmin(C_l, dim=-1)
        _,pred_labels_top5 = torch.topk(C_l, k=5, dim=1, largest=False)
        """
        """
        #dist z_s z_q
        dist = torch.sum(((z_s-z_q)**2),dim=2)
        C_l = torch.sum(dist,dim=2)
        pred_labels = torch.argmin(C_l, dim=-1)
        _,pred_labels_top5 = torch.topk(C_l, k=5, dim=1, largest=False)
        """
        """
        #Algo new
        sum_sq= torch.sum((w_sq - z_s)**2, dim=2) + torch.sum((w_sq - z_q)**2, dim=2)
        sum_s_q = torch.sum((w_s - z_s)**2, dim=2) + torch.sum((w_q - z_q)**2, dim=2)
        C_l = torch.sum(sum_sq - sum_s_q, dim=2)
        pred_labels = torch.argmin(C_l, dim=-1)
        """
        #print(sum_s.shape)
        #print(sum_q.shape)
        #print(pred_labels)

        acc_list.extend(pred_labels)
        acc_list_top5.extend(pred_labels_top5)

    cor = 0
    cor_top5 = 0
    for i in range(len(acc_list)):
        if acc_list[i] == test_labels[i][0]:
            cor += 1

        if test_labels[i][0] in acc_list_top5[i]:
            cor_top5 += 1
        #else:
            #print(test_labels[i])
            #print(acc_list_top5[i])

    avg_acc_task = 100*cor/len(acc_list)
    avg_acc_task_top5 = 100*cor_top5/len(acc_list_top5)
    print(f"Accuracy is {avg_acc_task}.")
    print(f"Top 5 accuracy is {avg_acc_task_top5}.")
    return avg_acc_task, avg_acc_task_top5

def run_paddle_new(enroll_embs,enroll_labels,test_embs,test_labels,method_info,method):
    x_q = torch.tensor(test_embs)
    y_q = torch.tensor(test_labels).long().unsqueeze(2)
    x_s = torch.tensor(enroll_embs)
    y_s = torch.tensor(enroll_labels).long().unsqueeze(2)
    
    task_dic = {}
    task_dic['y_s'] = y_s
    task_dic['y_q'] = y_q
    task_dic['x_s'] = x_s
    task_dic['x_q'] = x_q

    if method == 'glasso':
        print("GLASSO")
        method = PADDLE_GLASSO(**method_info)
        logs = method.run_task(task_dic=task_dic,
                           all_features_trainset=x_s,
                           all_labels_trainset=y_s,
                           support_features_params="support_features")
    else:
        method = PADDLE(**method_info)
        logs = method.run_task(task_dic=task_dic)
        
    #acc_sample, _ = compute_confidence_interval(logs['acc'][:, -1])
    return logs['acc'][:,-1].tolist(),logs['preds_q']

def run_2stage_paddle(enroll_embs,enroll_labels,test_embs,test_labels, test_audios,method_info):

    eval = Simpleshot(avg="mean",backend="L2",method="EM")
    acc_list, acc_list_5, pred_labels_5 = eval.eval(enroll_embs, enroll_labels, test_embs, test_labels, test_audios) 

    new_x_s = []
    new_y_s = []
    new_x_q = []
    new_y_q = []
    stage2_acc_list = []
    for i in range(len(acc_list_5)):
        task_x_s = []
        task_y_s = []
        
        top_classes = pred_labels_5[i][0].tolist()
        classes_dict = {str(label):i for i,label in enumerate(top_classes)}

        for label in sorted(top_classes):
            top_indices = np.where(enroll_labels[i] == label)            
            task_x_s.extend(enroll_embs[i][top_indices[0]])
            task_y_s.extend(np.array([classes_dict[str(label)] for label in enroll_labels[i][top_indices[0]]]))

        task_x_s = np.array(task_x_s)
        task_y_s = np.array(task_y_s)
        new_x_s.append(task_x_s)
        new_y_s.append(task_y_s)
        new_x_q.append(test_embs[i])
        new_y_q.append(np.array(test_labels[i]))

    
    new_x_s = np.array(new_x_s)
    new_y_s = np.array(new_y_s)
    new_x_q = np.array(new_x_q)
    new_y_q = np.array(new_y_q)

    acc_list, preds_q = run_paddle_new(new_x_s, new_y_s, new_x_q, new_y_q,method_info,'glasso')
    print(preds_q)
    preds_q = preds_q[0].tolist()
    original_preds_q = []
    for task in range(len(preds_q)):
        task_preds = []
        for pred in preds_q[task]:
            original_top_5 = pred_labels_5[task][0].tolist()
            task_preds.append(original_top_5[pred])

        original_preds_q.append(task_preds)
    
    acc_list_stage2 = compute_acc(torch.tensor(original_preds_q),torch.tensor(new_y_q))
 
    return acc_list_stage2