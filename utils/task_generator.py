import torch
import numpy as np
import random
from utils.utils import find_matching_positions
from tqdm import tqdm
import time

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

class Tasks_Generator:
    def __init__(self, uniq_classes, n_tasks=1, n_ways=1251, n_ways_eff=1, n_query=1, k_shot=1, seed="42"):
        """
        uniq_classes: all labels of classes in the support set [List]
        n_tasks: number of generated tasks
        n_ways: number of sampled classes in the support (number of support classes)
        n_ways_eff: number of sampled classes in the query, are part of the support classes (closed set) and much fewer (number of query classes)
        n_query: number of samples per query class
        k_shot: number of samples per support class
        """
        self.uniq_classes=sorted(uniq_classes)
        # Convert all class ids to integer values.

        self.n_tasks = n_tasks 
        self.n_ways= n_ways
        self.n_ways_eff = n_ways_eff
        self.n_query = n_query
        self.k_shot = k_shot
        self.seed = seed
         
        set_seed(self.seed)

        self.support_classes = []
        self.query_classes = []

        for i in range(n_tasks):
            sampled_classes=sorted(random.sample(self.uniq_classes, n_ways))
            self.support_classes.append(sampled_classes)
            # Query classes must be part of the sampled support classes
            self.query_classes.append(sorted(random.sample(sampled_classes, n_ways_eff)))
        
    def sampler(self, data_dict, mode):
        """
        Every time I sample, I set the seed.. sometimes it doesn't pick the same samples 
        when I use the sampler second time with a different batch size. 
        Setting the seed everytime removes this problem
        
        There are 2 modes: query and support. Depending on the mode, we either load the sampled support/query classes for n_tasks
        """
        out_embs = []
        out_labels = []
        out_slices = []

        if mode == "support":
            tasks_classes = self.support_classes
            no_samples = self.k_shot
        else:
            tasks_classes = self.query_classes
            no_samples = self.n_query
            

        for task, sampled_classes in tqdm(enumerate(tasks_classes)):
            
            # Get indices of samples that are part of the sampled classes in the support for this task.
            # The query must use the same indices as the support!
            self.label_dict = {label:i for i,label in enumerate(self.support_classes[task])}

            # Get the indices where elements in concat_labels are in sampled_classes
            data_label_indices = np.where(np.isin(np.array(data_dict['concat_labels']), sampled_classes))[0].tolist()
            
            all_labels = np.asarray(data_dict['concat_labels'])[data_label_indices]
            all_slices = np.asarray(data_dict['concat_audios'])[data_label_indices]
            all_embs = np.asarray(data_dict['concat_features'])[data_label_indices]

            combined_array = np.column_stack((all_labels, all_slices))
            unique_pairs, inverse_indices = np.unique(combined_array, axis=0, return_inverse=True)

            random_pairs = [(label, np.random.choice(unique_pairs[unique_pairs[:, 0] == str(label), 1], size=no_samples, replace=False)) for label in sorted(sampled_classes)]
            random_pairs_array = np.concatenate([[[label, id_] for id_ in ids] for label, ids in random_pairs])

            data_indices = np.array(find_matching_positions(combined_array, random_pairs_array))

            data_embs = all_embs[data_indices]
            data_labels = all_labels[data_indices]
            data_labels = np.asarray([self.label_dict[label] for label in data_labels])
            
            data_slices = all_slices[data_indices]

            out_embs.append(data_embs)
            out_labels.append(data_labels)
            out_slices.append(data_slices)

            #print("Sampled classes: "+str([self.label_dict[label] for label in sampled_classes]))
            #print("Labels: " + str(data_labels))

        out_embs = np.array(out_embs)
        out_labels = np.array(out_labels)
        out_slices = np.array(out_slices)

        return out_embs, out_labels, out_slices

    def sampler_unified(self, data_dict):
        """
        Every time I sample, I set the seed.. sometimes it doesn't pick the same samples 
        when I use the sampler second time with a different batch size. 
        Setting the seed everytime removes this problem
        """
        #set_seed(self.seed)
        """
        There are 2 modes: query and support. Depending on the mode, we either load the sampled support/query classes for n_tasks
        """
        support_embs = []
        support_labels = []
        support_slices = []
        support_embs2 = []
        support_labels2 = []
        support_slices2 = []
        query_embs = []
        query_labels = []
        query_slices = []

        only_support_classes = [[x for x in self.support_classes[task] if x not in self.query_classes[task]] for task,sup_classes in enumerate(self.support_classes)]
          
        for task, sampled_classes in tqdm(enumerate(only_support_classes)):
            # Get indices of samples that are part of the sampled classes in the support for this task.
            # The query must use the same indices as the support!
            self.label_dict = {label:i for i,label in enumerate(self.support_classes[task])}
            # Get the indices where elements in concat_labels are in sampled_classes
            data_label_indices = np.where(np.isin(np.array(data_dict['concat_labels']), sampled_classes))[0].tolist()
            
            all_labels = np.asarray(data_dict['concat_labels'])[data_label_indices]
            all_slices = np.asarray(data_dict['concat_audios'])[data_label_indices]
            all_embs = np.asarray(data_dict['concat_features'])[data_label_indices]

            combined_array = np.column_stack((all_labels, all_slices))
            unique_pairs, inverse_indices = np.unique(combined_array, axis=0, return_inverse=True)

            random_pairs = [(label, np.random.choice(unique_pairs[unique_pairs[:, 0] == str(label), 1], size=self.k_shot, replace=False)) for label in sorted(sampled_classes)]
            random_pairs_array = np.concatenate([[[label, id_] for id_ in ids] for label, ids in random_pairs])

            data_indices = np.array(find_matching_positions(combined_array, random_pairs_array))

            data_embs = all_embs[data_indices]
            data_labels = all_labels[data_indices]
            data_labels = np.asarray([self.label_dict[label] for label in data_labels])
            
            data_slices = all_slices[data_indices]

            support_embs.append(data_embs)
            support_labels.append(data_labels)
            support_slices.append(data_slices)

        for task, sampled_classes in tqdm(enumerate(self.query_classes)):
            

            # Get indices of samples that are part of the sampled classes in the support for this task.
            # The query must use the same indices as the support!
            self.label_dict = {label:i for i,label in enumerate(self.support_classes[task])}
            # Get the indices where elements in concat_labels are in sampled_classes
            data_label_indices = np.where(np.isin(np.array(data_dict['concat_labels']), sampled_classes))[0].tolist()
            
            all_labels = np.asarray(data_dict['concat_labels'])[data_label_indices]
            all_slices = np.asarray(data_dict['concat_audios'])[data_label_indices]
            all_embs = np.asarray(data_dict['concat_features'])[data_label_indices]

            combined_array = np.column_stack((all_labels, all_slices))
            unique_pairs, inverse_indices = np.unique(combined_array, axis=0, return_inverse=True)

            random_pairs = [(label, np.random.choice(unique_pairs[unique_pairs[:, 0] == str(label), 1], size=(self.k_shot+self.n_query), replace=False)) for label in sorted(sampled_classes)]
            random_pairs_array = np.concatenate([[[label, id_] for id_ in ids] for label, ids in random_pairs])

            data_indices = np.array(find_matching_positions(combined_array, random_pairs_array))

            data_embs = all_embs[data_indices]
            data_labels = all_labels[data_indices]
            data_labels = np.asarray([self.label_dict[label] for label in data_labels])
            
            data_slices = all_slices[data_indices]

            class_s_embs2 =[]
            class_s_labels2 =[]
            class_s_slices2 =[]
            class_q_embs2 =[]
            class_q_labels2 =[]
            class_q_slices2 =[]
            for label in self.query_classes[task]:
                label = self.label_dict[label]
                indices = np.where(data_labels == label)

                class_s_embs2.extend(data_embs[indices][:self.k_shot])
                class_s_labels2.extend(data_labels[indices][:self.k_shot])
                class_s_slices2.extend(data_slices[indices][:self.k_shot])

                class_q_embs2.extend(data_embs[indices][self.k_shot:])
                class_q_labels2.extend(data_labels[indices][self.k_shot:])
                class_q_slices2.extend(data_slices[indices][self.k_shot:])

            support_embs2.append(class_s_embs2)
            support_labels2.append(class_s_labels2)
            support_slices2.append(class_s_slices2)
            query_embs.append(class_q_embs2)
            query_labels.append(class_q_labels2)
            query_slices.append(class_q_slices2)

        support_embs = np.array(support_embs)
        support_labels = np.array(support_labels)
        support_slices = np.array(support_slices)
        support_embs2 = np.array(support_embs2)
        support_labels2 = np.array(support_labels2)
        support_slices2 = np.array(support_slices2)

        support_embs = np.concatenate((support_embs,support_embs2),axis=1)
        support_labels = np.concatenate((support_labels,support_labels2),axis=1)
        support_slices = np.concatenate((support_slices,support_slices2),axis=1)

        query_embs = np.array(query_embs)
        query_labels = np.array(query_labels)
        query_slices = np.array(query_slices)

        return query_embs, query_labels, query_slices,support_embs,support_labels,support_slices
