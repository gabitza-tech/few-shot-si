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
        
        elif self.method == "fsaic":
            pred_labels, pred_labels_5 = self.fsaic(enroll_embs,enroll_labels,test_embs,test_labels,method='normal')
        
        elif self.method == "fsaic_centroid":
            pred_labels, pred_labels_5 = self.fsaic(enroll_embs,enroll_labels,test_embs,test_labels,method='centroid')
        
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

    def calculate_sq_z_dist(self,enroll_embs,test_embs,enroll_labels, method='centroid'):
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

                w_sq = (sum_z_s + sum_z_q) #/ np.expand_dims(np.linalg.norm((sum_z_s+sum_z_q), ord=2, axis=-1),axis=-1)
                w_s = sum_z_s #/ np.expand_dims(np.linalg.norm(sum_z_s, ord=2, axis=-1),axis=-1)
                w_q = sum_z_q #/ np.expand_dims(np.linalg.norm(sum_z_q, ord=2, axis=-1),axis=-1)
                
                z_s_task.append(z_s)
                w_q_task.append(w_q)
                w_s_task.append(w_s)
                w_sq_task.append(w_sq)

            #distances.append(task_distances)
            z_s_all.append(z_s_task)
            w_q_all.append(w_q_task)
            w_sq_all.append(w_sq_task)
            w_s_all.append(w_s_task)

        test_embs = np.expand_dims(test_embs,1)
        #print(f'test embs:{test_embs.shape}')
        z_s = np.asarray(z_s_all)
        #print(f'shape z_s:{z_s.shape}')
        w_q_all = np.expand_dims(np.asarray(w_q_all),2)
        w_q = w_q_all/ np.expand_dims(np.linalg.norm(w_q_all, ord=2, axis=-1),axis=-1)
        #print(f'shape w_q:{w_q.shape}')
        w_sq_all = np.expand_dims(np.asarray(w_sq_all),2)
        w_sq = w_sq_all / np.expand_dims(np.linalg.norm(w_sq_all, ord=2, axis=-1),axis=-1)
        #print(f'shape w_sq:{w_sq.shape}')
        w_s_all = np.expand_dims(np.asarray(w_s_all),2)
        w_s = w_s_all / np.expand_dims(np.linalg.norm(w_s_all, ord=2, axis=-1),axis=-1)
        #print(f'shape w_s:{w_s.shape}')

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
            
            avg_enroll_embs = avg_enroll_embs / avg_enroll_embs.norm(dim=-1, keepdim=True)
            
            scores = 1 - torch.einsum('ijk,ilk->ijl', test_embs, avg_enroll_embs)
            
        else:
            print("Using SimpleShot inductive method with L2 norm backend")

            test_embs = torch.unsqueeze(test_embs,2) # [n_tasks,n_query,1,emb_shape]

            #avg_enroll_embs = torch.unsqueeze(avg_enroll_embs,1) # [n_tasks,1,1251,emb_shape]
            avg_enroll_embs = torch.unsqueeze(avg_enroll_embs / avg_enroll_embs.norm(dim=-1, keepdim=True),1)

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
        avg_enroll_embs = self.calculate_centroids(enroll_embs, enroll_labels)
        avg_test_embs = self.calculate_centroids(test_embs, test_labels)

        avg_test_embs = torch.from_numpy(avg_test_embs).float().to(self.device)
        avg_enroll_embs = torch.from_numpy(avg_enroll_embs).float().to(self.device)
        
        if self.backend == "cosine":
            print("Using SSCD method with cosine similarity backend.")

            avg_test_embs = avg_test_embs / avg_test_embs.norm(dim=-1, keepdim=True)
            avg_enroll_embs = avg_enroll_embs / avg_enroll_embs.norm(dim=-1, keepdim=True)

            scores = 1 - torch.einsum('ijk,ilk->ijl', avg_test_embs, avg_enroll_embs).repeat(1,n_query,1)

        else:
            print("Using SSCD method with L2 norm backend.")

            #avg_test_embs = torch.unsqueeze(avg_test_embs,2)
            #avg_enroll_embs = torch.unsqueeze(avg_enroll_embs,1) # [n_tasks,1,1251,emb_shape]

            avg_test_embs = torch.unsqueeze(avg_test_embs / avg_test_embs.norm(dim=-1, keepdim=True),2)
            avg_enroll_embs = torch.unsqueeze(avg_enroll_embs / avg_enroll_embs.norm(dim=-1, keepdim=True),1)
            
            # Class distance
            dist = (avg_test_embs-avg_enroll_embs)**2
            scores = torch.sum(dist,dim=-1).repeat(1,n_query,1) # [n_tasks,n_query,1251]

        pred_labels = torch.argmin(scores, dim=-1).long()
        _,pred_labels_top5 = torch.topk(scores, k=5, dim=-1, largest=False)

        return pred_labels, pred_labels_top5
    
    def fsaic(self,enroll_embs,enroll_labels,test_embs,test_labels, method='centroid'):
        print("Using FSAiC method")
        n_query = test_embs.shape[1]
        
        dist = torch.from_numpy(self.calculate_sq_z_dist(enroll_embs, test_embs, enroll_labels,method))
        C_l = torch.sum(dist,dim=-1)
        pred_labels = torch.argmin(C_l,-1).unsqueeze(1).repeat(1,n_query).to(torch.device('cpu'))
        _,pred_labels_top5 = torch.topk(C_l, k=5, dim=1, largest=False)
        
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

