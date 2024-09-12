import os
import numpy as np
from tqdm import tqdm
import torch
from utils.utils import majority_or_original,embedding_normalize
import torch.nn.functional as F
import random

class Simpleshot():
    
    def __init__(self,avg="mean",backend="cosine", device='cpu', method="inductive"):
        self.avg = avg
        self.backend = backend
        self.device = torch.device(device)
        self.method = method

    def eval(self,enroll_embs,enroll_labels,test_embs,test_labels, test_audios):

        if self.method == "ss":
            pred_labels, pred_labels_5 = self.inductive(enroll_embs,enroll_labels,test_embs,test_labels)
        
        elif self.method == "smv":
            pred_labels, pred_labels_5 = self.inductive(enroll_embs,enroll_labels,test_embs,test_labels)
            pred_labels = majority_or_original(pred_labels)
        
        elif self.method == "sscd":
            pred_labels, pred_labels_5 = self.sscd(enroll_embs,enroll_labels,test_embs,test_labels)
        
        test_labels = torch.from_numpy(test_labels).long()
        
        acc_tasks = compute_acc(pred_labels, test_labels)
        acc_tasks_5 = compute_acc_5(pred_labels_5, test_labels)

        return acc_tasks, acc_tasks_5, pred_labels_5

    def calculate_centroids(self,enroll_embs,enroll_labels):
        # Returns [n_tasks,n_ways,192] tensor with the centroids
        # sampled_classes: [n_tasks,n_ways]
        
        sampled_classes=[]
        for task in enroll_labels:
            sampled_classes.append(sorted(list(set(task))))

        avg_enroll_embs = []
        for i,task_classes in enumerate(sampled_classes):
            task_enroll_embs = []
            
            for label in task_classes:
                indices = np.where(enroll_labels[i] == label)
                if self.avg == "mean":
                    embedding = (enroll_embs[i][indices[0]].sum(axis=0).squeeze()) / len(indices[0])
                if self.avg == "median":
                    embedding = np.median(enroll_embs[i][indices[0]], axis=0)
                task_enroll_embs.append(embedding)
            avg_enroll_embs.append(task_enroll_embs)
        
        avg_enroll_embs = np.asarray(avg_enroll_embs)

        return avg_enroll_embs

    def inductive(self,enroll_embs,enroll_labels,test_embs,test_labels):
        """
        enroll_embs: [n_tasks,k_shot*n_ways,192]
        enroll_labels: [n_tasks,k_shot*n_ways]
        test_embs: [n_tasks,n_query,192]
        test_labels: [n_tasks,n_query]
        """
        # Calculate the mean embeddings for each class in the support
        avg_enroll_embs = self.calculate_centroids(enroll_embs, enroll_labels)

        test_embs = torch.from_numpy(test_embs).float().to(self.device)
        avg_enroll_embs = torch.from_numpy(avg_enroll_embs).float().to(self.device)
      
        if self.backend == "cosine":
            print("Using SimpleShot inductive method with cosine similarity backend")
            
            avg_enroll_embs = avg_enroll_embs / np.expand_dims(np.linalg.norm(avg_enroll_embs, ord=2, axis=-1),axis=-1)

            scores = 1 - torch.einsum('ijk,ilk->ijl', test_embs, avg_enroll_embs)
            
        else:
            print("Using SimpleShot inductive method with L2 norm backend")

            test_embs = torch.unsqueeze(test_embs,2) # [n_tasks,n_query,1,emb_shape]

            avg_enroll_embs = avg_enroll_embs / np.expand_dims(np.linalg.norm(avg_enroll_embs, ord=2, axis=-1),axis=-1)
            avg_enroll_embs = torch.unsqueeze(avg_enroll_embs,1)

            # Class distance
            dist = (test_embs-avg_enroll_embs)**2
            scores = torch.sum(dist,dim=-1) # [n_tasks,n_query,1251]

        pred_labels = torch.argmin(scores, dim=-1).long()#.tolist()
        _,pred_labels_top5 = torch.topk(scores, k=5, dim=-1, largest=False)

        return pred_labels, pred_labels_top5


    def sscd(self,enroll_embs,enroll_labels,test_embs,test_labels):
        """
        enroll_embs: [n_tasks,k_shot*n_ways,192]
        enroll_labels: [n_tasks,k_shot*n_ways]
        test_embs: [n_tasks,n_query,192]
        test_labels: [n_tasks,n_query]
        """
        # Calculate the mean embeddings for each class in the support

        n_query = test_embs.shape[1]
        avg_enroll_embs = torch.from_numpy(self.calculate_centroids(enroll_embs, enroll_labels)).float().to(self.device)
        avg_test_embs = torch.from_numpy(self.calculate_centroids(test_embs, test_labels)).float().to(self.device)
        
        if self.backend == "cosine":
            print("Using SSCD method with cosine similarity backend.")

            avg_test_embs = avg_test_embs / np.expand_dims(np.linalg.norm(avg_test_embs, ord=2, axis=-1),axis=-1)
            avg_enroll_embs = avg_enroll_embs / np.expand_dims(np.linalg.norm(avg_enroll_embs, ord=2, axis=-1),axis=-1)

            scores = 1 - torch.einsum('ijk,ilk->ijl', avg_test_embs, avg_enroll_embs).repeat(1,n_query,1)
        
        else:
            print("Using SSCD method with L2 norm backend.")
            avg_test_embs = avg_test_embs / np.expand_dims(np.linalg.norm(avg_test_embs, ord=2, axis=-1),axis=-1)
            avg_enroll_embs = avg_enroll_embs / np.expand_dims(np.linalg.norm(avg_enroll_embs, ord=2, axis=-1),axis=-1)
            
            # Class distance
            dist = (avg_test_embs-avg_enroll_embs)**2
            dist = torch.unsqueeze(dist,1)
            scores = torch.sum(dist,dim=-1).repeat(1,n_query,1) # [n_tasks,n_query,1251]

        pred_labels = torch.argmin(scores, dim=-1).long()
        _,pred_labels_top5 = torch.topk(scores, k=5, dim=-1, largest=False)
        
        return pred_labels, pred_labels_top5
    
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

