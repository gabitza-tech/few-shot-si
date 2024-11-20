import os
import numpy as np
from tqdm import tqdm
import torch
from utils.utils import majority_or_original,embedding_normalize
import torch.nn.functional as F
import random

class FSAIC():

    def __init__(self, device='cpu', method="centroid",top_k=5):
        self.device = torch.device(device)
        self.method = method
        self.top_k = top_k

    def eval(self,enroll_embs,enroll_labels,test_embs,test_labels, test_audios):

        if self.method == "normal":
            pred_labels, pred_labels_5 = self.fsaic(enroll_embs,enroll_labels,test_embs,test_labels,method='normal')

        # The FSAiC used in paper!
        elif self.method == "centroid":
            pred_labels, pred_labels_5 = self.fsaic(enroll_embs,enroll_labels,test_embs,test_labels,method='centroid')

        test_labels = torch.from_numpy(test_labels).long()

        acc_tasks = compute_acc(pred_labels, test_labels)
        acc_tasks_5 = compute_acc_5(pred_labels_5, test_labels)

        return acc_tasks, acc_tasks_5, pred_labels_5


    def calculate_fsaic_centroids(self,enroll_embs,test_embs,enroll_labels, method='centroid'):
        # Returns [n_tasks,n_ways,192] tensor with the centroids
        # sampled_classes: [n_tasks,n_ways]
        sampled_classes=[]
        for task in enroll_labels:
            sampled_classes.append(sorted(list(set(task))))

        distances = []
        z_s_all = []
        w_s_all = []
        w_sq_all = []
        w_q_all = []
        for i,task_classes in enumerate(sampled_classes):
            task_distances = []
            # Query samples for task i
            z_q = test_embs[i]
            z_s_task = []
            w_s_task = []
            w_sq_task = []
            w_q_task = []
            for label in task_classes:

                indices = np.where(enroll_labels[i] == label)
                # Samples from class 'label' from S
                z_s = enroll_embs[i][indices]

                sum_z_s = z_s.sum(axis=0).squeeze()
                sum_z_q = z_q.sum(axis=0).squeeze()

                w_sq = (sum_z_s + sum_z_q)

                z_s_task.append(z_s)
                w_q_task.append(sum_z_q)
                w_s_task.append(sum_z_s)
                w_sq_task.append(w_sq)

            #distances.append(task_distances)
            z_s_all.append(z_s_task)
            w_q_all.append(w_q_task)
            w_sq_all.append(w_sq_task)
            w_s_all.append(w_s_task)

        test_embs = np.expand_dims(test_embs,1)
        z_s = np.asarray(z_s_all)

        w_q_all = np.expand_dims(np.asarray(w_q_all),2)
        w_q = w_q_all/ np.expand_dims(np.linalg.norm(w_q_all, ord=2, axis=-1),axis=-1)

        w_sq_all = np.expand_dims(np.asarray(w_sq_all),2)
        w_sq = w_sq_all / np.expand_dims(np.linalg.norm(w_sq_all, ord=2, axis=-1),axis=-1)

        w_s_all = np.expand_dims(np.asarray(w_s_all),2)
        w_s = w_s_all / np.expand_dims(np.linalg.norm(w_s_all, ord=2, axis=-1),axis=-1)

        if method == 'normal':
            dist_w_sq_support = np.sum((w_sq-z_s)**2,axis=2)
            dist_w_s_support = np.sum((w_s-z_s)**2,axis=2)
            dist_w_sq_query = np.sum((w_sq-test_embs)**2,axis=2)

        elif method == 'centroid': # BETTER
            dist_w_sq_support = np.sum((w_sq-z_s)**2,axis=2)
            dist_w_s_support = np.sum((w_s-z_s)**2,axis=2)
            dist_w_sq_query = np.sum((w_sq-w_q)**2,axis=2)

        final_distance = dist_w_sq_query + (dist_w_sq_support - dist_w_s_support)

        return final_distance

    def fsaic(self,enroll_embs,enroll_labels,test_embs,test_labels, method='centroid'):
        print("Using FSAiC method")
        n_query = test_embs.shape[1]

        dist = torch.from_numpy(self.calculate_fsaic_centroids(enroll_embs, test_embs, enroll_labels,method))
        C_l = torch.sum(dist,dim=-1)

        pred_labels = torch.argmin(C_l,-1).unsqueeze(1).repeat(1,n_query).to(torch.device('cpu'))
        _,pred_labels_top5 = torch.topk(C_l, k=self.top_k, dim=-1, largest=False)
        pred_labels_top5 = pred_labels_top5.unsqueeze(1).repeat(1,n_query,1).to(torch.device('cpu'))

        return pred_labels,pred_labels_top5

def compute_acc(pred_labels, test_labels):
    # Check if the input tensors have the same shape
    assert pred_labels.shape == test_labels.shape, "Shape mismatch between predicted and groundtruth labels"
    # Calculate accuracy for each task
    acc_list = (pred_labels == test_labels).float().mean(dim=1).tolist()

    return acc_list

def compute_acc_5(pred_labels, test_labels):
    # Check if the input tensors have the same shape
    acc_list = []
    for i in range(test_labels.shape[0]):
        if test_labels[i][0] in pred_labels[i][0]:
            acc_list.append([1])
        else:
            acc_list.append([0])

    acc_list = torch.tensor(np.array(acc_list)).float().mean(dim=1).tolist()

    return acc_list

