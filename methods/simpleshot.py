import os
import numpy as np
from tqdm import tqdm
import torch
from utils.utils import majority_or_original,embedding_normalize
import torch.nn.functional as F
import random

class Simpleshot():
    def __init__(self,avg="mean",backend="cosine", majority="True",device='cpu', method="inductive"):
        self.avg = avg
        self.backend = backend
        self.majority = majority
        self.device = torch.device(device)
        self.method = method

    def eval(self,enroll_embs,enroll_labels,test_embs,test_labels, test_audios):

        if self.method == "inductive":
            pred_labels, pred_labels_5 = self.inductive(enroll_embs,enroll_labels,test_embs,test_labels)
        elif self.method == "transductive_centroid":
            pred_labels, pred_labels_5 = self.transductive_centroid(enroll_embs,enroll_labels,test_embs,test_labels)
        elif self.method == "EM":
            pred_labels, pred_labels_5 = self.estimation_maximization(enroll_embs,enroll_labels,test_embs,test_labels)
        elif self.method == "EM_frobenius":
            pred_labels, pred_labels_5 = self.estimation_maximization_frobenius(enroll_embs,enroll_labels,test_embs,test_labels)
        elif self.method == "EM_normal":
            pred_labels, pred_labels_5 = self.estimation_maximization_normal(enroll_embs,enroll_labels,test_embs,test_labels)
        elif self.method == "inductive_maj":
            pred_labels, pred_labels_5 = self.inductive(enroll_embs,enroll_labels,test_embs,test_labels)
            pred_labels = majority_or_original(pred_labels)
            
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

    def calculate_sq_z_dist(self,enroll_embs,test_embs,enroll_labels):
        # Returns [n_tasks,n_ways,192] tensor with the centroids
        # sampled_classes: [n_tasks,n_ways]
        
        sampled_classes=[]
        for task in enroll_labels:
            sampled_classes.append(sorted(list(set(task))))

        distances = []
        for i,task_classes in enumerate(sampled_classes):
            task_distances = []
            z_q = test_embs[i] # this one is the same for all classes, differs only per task
            for label in task_classes:
                indices = np.where(enroll_labels[i] == label)
                # ALL THESE ARE FOR CLASS l (label)
                z_s = enroll_embs[i][indices]
                
                N_samples_SQ = len(indices[0])+z_q.shape[0]
                N_samples_S = len(indices[0])
                
                w_sq = (z_s.sum(axis=0).squeeze() + z_q.sum(axis=0).squeeze()) / N_samples_SQ
                #w_sq = np.expand_dims((torch.from_numpy(w_sq) / torch.from_numpy(w_sq).norm(dim=-1, keepdim=True)).numpy(),0)
                w_sq_norm = np.expand_dims(np.linalg.norm(w_sq, ord=2, axis=-1), axis=-1)
                w_sq = w_sq / w_sq_norm

                w_s = (z_s.sum(axis=0).squeeze() / N_samples_S)
                #w_s = np.expand_dims((torch.from_numpy(w_s) / torch.from_numpy(w_s).norm(dim=-1,keepdim=True)).numpy(),0)
                w_s_norm = np.expand_dims(np.linalg.norm(w_s, ord=2, axis=-1), axis=-1)
                w_s = w_s / w_s_norm
                
                dist_w_sq_support = np.sum((w_sq-z_s)**2,axis=0)
                dist_w_s_support = np.sum((w_s-z_s)**2,axis=0)
                dist_w_sq_query = np.sum((w_sq-z_q)**2,axis=0)
                
                final_distance = dist_w_sq_query + dist_w_sq_support - dist_w_s_support

                task_distances.append(final_distance)
            distances.append(task_distances)

        distances = np.asarray(distances)

        return distances
    def calculate_sq_z_dist_frobenius(self,enroll_embs,test_embs,enroll_labels):
        # Returns [n_tasks,n_ways,192] tensor with the centroids
        # sampled_classes: [n_tasks,n_ways]
        
        sampled_classes=[]
        for task in enroll_labels:
            sampled_classes.append(sorted(list(set(task))))

        distances = []
        for i,task_classes in enumerate(sampled_classes):
            task_distances = []
            z_q = test_embs[i] # this one is the same for all classes, differs only per task
            for label in task_classes:
                indices = np.where(enroll_labels[i] == label)
                # ALL THESE ARE FOR CLASS l (label)
                z_s = enroll_embs[i][indices]
                
                N_samples_SQ = len(indices[0])+z_q.shape[0]
                N_samples_S = len(indices[0])
                
                w_sq = (z_s.sum(axis=0).squeeze() + z_q.sum(axis=0).squeeze()) / N_samples_SQ
                w_sq = np.expand_dims((torch.from_numpy(w_sq) / torch.from_numpy(w_sq).norm(dim=-1, keepdim=True)).numpy(),0)
                #w_sq_norm = np.expand_dims(np.linalg.norm(w_sq, ord=2, axis=-1), axis=-1)
                #w_sq = w_sq / w_sq_norm

                w_s = (z_s.sum(axis=0).squeeze() / N_samples_S)
                w_s = np.expand_dims((torch.from_numpy(w_s) / torch.from_numpy(w_s).norm(dim=-1,keepdim=True)).numpy(),0)
                #w_s_norm = np.expand_dims(np.linalg.norm(w_s, ord=2, axis=-1), axis=-1)
                #w_s = w_s / w_s_norm
                
                dist_w_sq_support = np.sum((w_sq-z_s)**2,axis=0)
                dist_w_s_support = np.sum((w_s-z_s)**2,axis=0)
                dist_w_sq_query = np.sum((w_sq-z_q)**2,axis=0)
                
                final_distance = dist_w_sq_query + dist_w_sq_support - dist_w_s_support

                task_distances.append(final_distance)
            distances.append(task_distances)

        distances = np.asarray(distances)

        return distances

    def calculate_sq_z_dist_normal(self,enroll_embs,test_embs,enroll_labels):
        # Returns [n_tasks,n_ways,192] tensor with the centroids
        # sampled_classes: [n_tasks,n_ways]
        
        sampled_classes=[]
        for task in enroll_labels:
            sampled_classes.append(sorted(list(set(task))))

        distances = []
        for i,task_classes in enumerate(sampled_classes):
            task_distances = []
            z_q = test_embs[i] # this one is the same for all classes, differs only per task
            for label in task_classes:
                indices = np.where(enroll_labels[i] == label)
                # ALL THESE ARE FOR CLASS l (label)
                z_s = enroll_embs[i][indices]
                
                N_samples_SQ = len(indices[0])+z_q.shape[0]
                N_samples_S = len(indices[0])
                
                w_sq = (z_s.sum(axis=0).squeeze() + z_q.sum(axis=0).squeeze()) / N_samples_SQ
                w_sq = np.expand_dims(w_sq,0)
                
                w_s = z_s.sum(axis=0).squeeze() / N_samples_S
                w_s = np.expand_dims(w_s,0)
                
                dist_w_sq_support = np.sum((w_sq-z_s)**2,axis=0)
                dist_w_s_support = np.sum((w_s-z_s)**2,axis=0)
                dist_w_sq_query = np.sum((w_sq-z_q)**2,axis=0)
                
                final_distance = dist_w_sq_query + dist_w_sq_support - dist_w_s_support

                task_distances.append(final_distance)
            distances.append(task_distances)

        distances = np.asarray(distances)

        return distances


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


    def transductive_centroid(self,enroll_embs,enroll_labels,test_embs,test_labels):
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
            print("Using SimpleShot transductive centroid method with cosine similarity backend.")

            avg_test_embs = avg_test_embs / avg_test_embs.norm(dim=-1, keepdim=True)
            avg_enroll_embs = avg_enroll_embs / avg_enroll_embs.norm(dim=-1, keepdim=True)

            scores = 1 - torch.einsum('ijk,ilk->ijl', avg_test_embs, avg_enroll_embs).repeat(1,n_query,1)

        else:
            print("Using SimpleShot transductive centroid method with L2 norm backend.")

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
    
    def estimation_maximization(self,enroll_embs,enroll_labels,test_embs,test_labels):
        print("Using Estimation maximization method")
        n_query = test_embs.shape[1]
        
        device_here = torch.device('cuda:0')
        
        dist = torch.from_numpy(self.calculate_sq_z_dist(enroll_embs, test_embs,enroll_labels))
        C_l = torch.sum(dist,dim=-1)
        pred_labels = torch.argmin(C_l,-1).unsqueeze(1).repeat(1,n_query).to(torch.device('cpu'))
        _,pred_labels_top5 = torch.topk(C_l, k=5, dim=1, largest=False)
        
        return pred_labels,pred_labels_top5
    
    def estimation_maximization_normal(self,enroll_embs,enroll_labels,test_embs,test_labels):
        print("Using Estimation maximization method")
        n_query = test_embs.shape[1]
        
        device_here = torch.device('cuda:0')
        
        dist = torch.from_numpy(self.calculate_sq_z_dist_normal(enroll_embs, test_embs,enroll_labels))
        C_l = torch.sum(dist,dim=-1)
        pred_labels = torch.argmin(C_l,-1).unsqueeze(1).repeat(1,n_query).to(torch.device('cpu'))
        _,pred_labels_top5 = torch.topk(C_l, k=5, dim=1, largest=False)
        
        return pred_labels,pred_labels_top5
    
    def estimation_maximization_frobenius(self,enroll_embs,enroll_labels,test_embs,test_labels):
        print("Using Estimation maximization method")
        n_query = test_embs.shape[1]
        
        device_here = torch.device('cuda:0')
        
        dist = torch.from_numpy(self.calculate_sq_z_dist_frobenius(enroll_embs, test_embs,enroll_labels))
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

