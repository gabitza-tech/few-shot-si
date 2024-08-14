import os
import numpy as np
from tqdm import tqdm
import torch
from utils.utils import majority_or_original,embedding_normalize
import torch.nn.functional as F
import random

class Simpleshot():
    def __init__(self,avg="mean",backend="l2", majority="True",device='cpu', method="inductive"):
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
        elif self.method == "transductive_L2_sum":
            pred_labels, pred_labels_5 = self.transductive_L2_sum(enroll_embs,enroll_labels,test_embs,test_labels)
        elif self.method == "EM":
            pred_labels, pred_labels_5 = self.estimation_maximization(enroll_embs,enroll_labels,test_embs,test_labels)
        elif self.method == "inductive_maj":
            pred_labels, pred_labels_5 = self.inductive(enroll_embs,enroll_labels,test_embs,test_labels)
            pred_labels = majority_or_original(pred_labels)
            
        test_labels = torch.from_numpy(test_labels).long()

        acc_tasks = compute_acc(pred_labels, test_labels)
        acc_tasks_5 = compute_acc_5(pred_labels_5, test_labels)

        #exit(0)
        for i,acc in enumerate(acc_tasks):
            if self.method == "EM" and acc == 0:
                audio = test_audios[i]
                label = test_labels[i]
                with open("log_EM",'a') as f:
                    f.write(f"For test audio {audio} with label {label} accuracy is: {acc}")
                    f.write("\n")

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
                    #embs_class = enroll_embs[i][indices]
                    #count = 0 
                    #for j in range(192):
                    #    if embs_class[0][j]*embs_class[1][j] < 0:
                    #        count += 1
                    #print(count)
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
                
                w_sq = np.expand_dims(((z_s.sum(axis=0).squeeze()+z_q.sum(axis=0).squeeze()) / N_samples_SQ),0)
                w_s = np.expand_dims((z_s.sum(axis=0).squeeze() / N_samples_S),0)
                
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
            
            avg_enroll_embs = torch.from_numpy(embedding_normalize(avg_enroll_embs.numpy(),use_mean=True))
            scores = torch.einsum('ijk,ilk->ijl', test_embs, avg_enroll_embs)

            pred_labels = torch.argmax(scores, dim=-1).long()#.tolist()
            _,pred_labels_top5 = torch.topk(scores, k=5, dim=-1, largest=True)
            
        else:
            print("Using SimpleShot inductive method with L2 norm backend")

            test_embs = torch.unsqueeze(test_embs,2) # [n_tasks,n_query,1,emb_shape]
            avg_enroll_embs = torch.unsqueeze(avg_enroll_embs,1) # [n_tasks,1,1251,emb_shape]
            
            # Class distance
            dist = (test_embs-avg_enroll_embs)**2
            C_l = torch.sum(dist,dim=-1) # [n_tasks,n_query,1251]

            pred_labels = torch.argmin(C_l, dim=-1).long()#.tolist()
            _,pred_labels_top5 = torch.topk(C_l, k=5, dim=-1, largest=False)

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

            avg_test_embs = torch.from_numpy(embedding_normalize(avg_test_embs.numpy(),use_mean=True))
            avg_enroll_embs = torch.from_numpy(embedding_normalize(avg_enroll_embs.numpy(),use_mean=True))

            scores = torch.einsum('ijk,ilk->ijl', avg_test_embs, avg_enroll_embs).repeat(1,n_query,1)
            pred_labels = torch.argmax(scores, dim=-1).long()
            _,pred_labels_top5 = torch.topk(scores, k=10, dim=-1, largest=True)
            
        else:
            print("Using SimpleShot transductive centroid method with L2 norm backend.")

            avg_test_embs = torch.unsqueeze(avg_test_embs,2) # [n_tasks,n_query,1,emb_shape]
            avg_enroll_embs = torch.unsqueeze(avg_enroll_embs,1) # [n_tasks,1,1251,emb_shape]

            # Class distance
            dist = (avg_test_embs-avg_enroll_embs)**2
            C_l = torch.sum(dist,dim=-1).repeat(1,n_query,1) # [n_tasks,n_query,1251]

            pred_labels = torch.argmin(C_l, dim=-1).long()
            _,pred_labels_top5 = torch.topk(C_l, k=5, dim=-1, largest=False)

        return pred_labels, pred_labels_top5
    
    def transductive_L2_sum(self,enroll_embs,enroll_labels,test_embs,test_labels):
        """
        enroll_embs: [n_tasks,k_shot*n_ways,192]
        enroll_labels: [n_tasks,k_shot*n_ways]
        test_embs: [n_tasks,n_query,192]
        test_labels: [n_tasks,n_query]
        """
        n_query = test_embs.shape[1]
    
        avg_enroll_embs = self.calculate_centroids(enroll_embs, enroll_labels)
        test_embs = torch.from_numpy(test_embs).float().to(self.device)
        avg_enroll_embs = torch.from_numpy(avg_enroll_embs).float().to(self.device)
      
        print("Using SimpleShot transductive L2_sum method with L2 norm backend")
        test_embs = torch.unsqueeze(test_embs,2) # [n_tasks,n_query,1,emb_shape]
        avg_enroll_embs = torch.unsqueeze(avg_enroll_embs,1) # [n_tasks,1,1251,emb_shape]

        # Class distance
        dist = torch.sum((test_embs-avg_enroll_embs)**2,dim=-1)
        # We sum the distance of all the samples in the query, then repeat it in order to have the same n_query as the test labels
        C_l = torch.unsqueeze(torch.sum(dist,dim=1),dim=1).repeat(1,n_query,1) # [n_tasks, 1251]

        pred_labels = torch.argmin(C_l, dim=-1).long()
        _,pred_labels_top5 = torch.topk(C_l, k=10, dim=-1, largest=False)
        
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
        #for j in range(test_labels.shape[1]):
        if test_labels[i][0] in pred_labels[i][0]:
            acc_list.append([1])
        else:
            acc_list.append([0])
    
    acc_list = torch.tensor(np.array(acc_list)).float().mean(dim=1).tolist()
    

    return acc_list

