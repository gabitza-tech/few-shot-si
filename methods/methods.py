import os
import numpy as np
from tqdm import tqdm
import torch
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval, get_one_hot
from methods.paddle import PADDLE
import torch.nn.functional as F
from methods.simpleshot import Simpleshot, compute_acc
from methods.fsaic import FSAIC

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
        
    return logs['acc'][:,-1].tolist(),logs['preds_q']