import os
import numpy as np
from tqdm import tqdm
import torch
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval, get_one_hot
from methods.paddle import PADDLE
from methods.paddle_glasso import PADDLE_GLASSO
import torch.nn.functional as F
from methods.simpleshot import Simpleshot, compute_acc
from methods.fsaic import FSAIC
from methods.tim import ALPHA_TIM, TIM_GD
from methods.laplacianshot import LAPLACIAN_SHOT
from methods.hard_em_dirichlet import HARD_EM_DIRICHLET

def run_em_dirichlet(enroll_embs,enroll_labels,test_embs,test_labels,method_info,method='hard_em_dirichlet'):
    x_q = torch.tensor(test_embs)
    y_q = torch.tensor(test_labels).long().unsqueeze(2)
    x_s = torch.tensor(enroll_embs)
    y_s = torch.tensor(enroll_labels).long().unsqueeze(2)
    
    task_dic = {}
    task_dic['y_s'] = y_s
    task_dic['y_q'] = y_q
    task_dic['x_s'] = x_s
    task_dic['x_q'] = x_q

    if method == 'hard_em_dirichlet':
        method = HARD_EM_DIRICHLET(**method_info)
        logs = method.run_task(task_dic=task_dic)
    
    return logs['acc'][:,-1].tolist(),logs['preds_q']

def run_method(enroll_embs,enroll_labels,test_embs,test_labels,method_info,method='paddle'):
    x_q = torch.tensor(test_embs)
    y_q = torch.tensor(test_labels).long().unsqueeze(2)
    x_s = torch.tensor(enroll_embs)
    y_s = torch.tensor(enroll_labels).long().unsqueeze(2)
    
    task_dic = {}
    task_dic['y_s'] = y_s
    task_dic['y_q'] = y_q
    task_dic['x_s'] = x_s
    task_dic['x_q'] = x_q

    if method == 'paddle':
        method = PADDLE(**method_info)
        logs = method.run_task(task_dic=task_dic)
    elif method == 'glasso':
        method = PADDLE_GLASSO(**method_info)
        logs = method.run_task(task_dic=task_dic,
                           all_features_trainset=x_s,
                           all_labels_trainset=y_s,
                           support_features_params="support_features")
    elif method == 'laplacianshot':
        method = LAPLACIAN_SHOT(**method_info)
        logs = method.run_task(task_dic=task_dic)
    elif method == 'tim_alpha':
        method = ALPHA_TIM(**method_info)
        logs = method.run_task(task_dic=task_dic)
    elif method == 'tim_grad':
        method = TIM_GD(**method_info)
        logs = method.run_task(task_dic=task_dic)
        
    return logs['acc'][:,-1].tolist(),logs['preds_q']

def run_2stage_method(enroll_embs, enroll_labels, test_embs, test_labels, test_audios, method_info, method='paddle', top_k=5):

    eval = FSAIC(method="centroid", top_k=top_k)
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

    acc_list,preds_q = run_method(new_x_s, new_y_s, new_x_q, new_y_q, method_info, method)
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
