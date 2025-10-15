import os
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigs
import torch.nn.functional as F

def setup_logger(log_file):
    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    # Create a file handler that writes to the specified log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

# Usage example
log_file = 'my_log_file.log'
logger = setup_logger(log_file)

def majority_or_original(tensor):
    majority_labels = []
    for task in tensor:
        values, counts = task.unique(return_counts=True)
        max_count = counts.max().item()
        modes = values[counts == max_count]
        
        # If there's a tie (multiple modes), keep the original values for this task
        if len(modes) > 1:
            majority_labels.append(task)
        else:
            majority_labels.append(modes.repeat(len(task)))
    
    return torch.stack(majority_labels)

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def sampler_query(test_dict,sampled_classes):
    # We find which indices in the lists are part of the sampled classes
    test_indices = [index for index, element in enumerate(test_dict['concat_labels']) if element in sampled_classes]
            
    """
    We first construct the test/query, as it is easier and we always use all of it, one file at a time
    We will create a tensor of size [n_query,192], where n_query are all queries that are part of the sampled classes.
    """
    print("Creating test embeddings vector and the reference labels")
    test_embs = test_dict['concat_features'][test_indices] 

    label_dict = {label:i for i,label in enumerate(sampled_classes)}
    test_labels = np.asarray(test_dict['concat_labels'])[test_indices]
    test_labels = np.asarray([label_dict[label] for label in test_labels])

    return test_embs, test_labels

def sampler_windows_query(test_dict,sampled_classes):
    # We find which indices in the lists are part of the sampled classes
    test_label_indices = [index for index, element in enumerate(test_dict['concat_labels']) if element in sampled_classes]

    all_labels = np.asarray(test_dict['concat_labels'])[test_label_indices]
    all_slices = np.asarray(test_dict['concat_slices'])[test_label_indices]
    all_embs = np.asarray(test_dict['concat_features'])[test_label_indices]

    combined_array = np.column_stack((all_labels, all_slices))
    unique_pairs, inverse_indices = np.unique(combined_array, axis=0, return_inverse=True)
    grouped_indices = [np.where(inverse_indices == i)[0] for i in range(len(unique_pairs))]

    test_embs = np.asarray([all_embs[indices] for indices in grouped_indices])

    label_dict = {label:i for i,label in enumerate(sampled_classes)}
    test_labels = np.asarray([all_labels[indices[0]] for indices in grouped_indices])
    test_labels = np.asarray([label_dict[label] for label in test_labels])

    

    return test_embs, test_labels

def sampler_support(enroll_dict, sampled_classes,k_shot):
    enroll_indices = [index for index, element in enumerate(enroll_dict['concat_labels']) if element in sampled_classes]
    """
    We sample the embeddings from k_shot audios in each sampled class
    If audios are normal, it wil sample exactly k_shot audios per class
    If audios are split, it will sample a variable number of window_audios, but still coming from k_shot audios
    We don't oversample embs in a class to equalize the classes lengths at the moment, as we calculate the mean anyway
    """
    
    all_enroll_labels = np.array(enroll_dict['concat_labels'])
    all_slices = np.asarray(enroll_dict['concat_slices'])

    value_masks = [all_enroll_labels == class_name for class_name in sorted(sampled_classes)]
    # Extract indices for each value
    value_indices = [np.where(mask)[0] for mask in value_masks]

    # We do this in order to not take more samples from a class than the existing maximum, or simply not duplicate samples in classes when extracting
    if k_shot > 10:
        k_shot_list = []
        for indices in value_indices:
            if k_shot > len(indices):
                k_shot_list.append(len(indices))
            else:
                k_shot_list.append(k_shot)

        # Shuffle the indices for each value
        enroll_indices = np.concatenate([np.random.choice(indices, size=k_shot_list[index], replace=False) for index,indices in enumerate(value_indices)])
    else:
        enroll_indices = np.concatenate([np.random.choice(indices, size=k_shot, replace=False) for indices in value_indices])
    
    enroll_embs = enroll_dict['concat_features'][enroll_indices]
    
    label_dict = {label:i for i,label in enumerate(sampled_classes)}
    enroll_labels = all_enroll_labels[enroll_indices]
    enroll_labels = np.asarray([label_dict[label] for label in enroll_labels])

    return enroll_embs, enroll_labels


def sampler_windows_support(enroll_dict, sampled_classes,k_shot):
    # We find which indices in the lists are part of the sampled classes

    #enroll_dict['concat_labels'] = np.repeat(enroll_dict['concat_labels'],10,0)
    #enroll_dict['concat_features'] = np.repeat(enroll_dict['concat_features'],10,0)
    #enroll_dict['concat_slices'] = np.repeat(enroll_dict['concat_slices'],10,0)
    #print(enroll_dict['concat_labels'].shape)
    #print(enroll_dict['concat_features'].shape)
    #print(enroll_dict['concat_slices'].shape)

    enroll_label_indices = [index for index, element in enumerate(enroll_dict['concat_labels']) if element in sampled_classes]

    all_labels = np.asarray(enroll_dict['concat_labels'])[enroll_label_indices]
    all_slices = np.asarray(enroll_dict['concat_slices'])[enroll_label_indices]
    all_patchs = np.asarray(enroll_dict['concat_patchs'])[enroll_label_indices]
    all_embs = np.asarray(enroll_dict['concat_features'])[enroll_label_indices]

    combined_array = np.column_stack((all_labels, all_slices))
    unique_pairs, inverse_indices = np.unique(combined_array, axis=0, return_inverse=True)
    
    random_pairs = [(label, np.random.choice(unique_pairs[unique_pairs[:, 0] == label, 1], size=k_shot, replace=False)) for label in sorted(sampled_classes)]
    random_pairs_array = np.concatenate([[[label, id_] for id_ in ids] for label, ids in random_pairs])

    enroll_indices = np.array(find_matching_positions(combined_array, random_pairs_array))

    enroll_embs = all_embs[enroll_indices]
    
    label_dict = {label:i for i,label in enumerate(sampled_classes)}
    enroll_labels = all_labels[enroll_indices]
    enroll_labels = np.asarray([label_dict[label] for label in enroll_labels])

    return enroll_embs, enroll_labels#, enroll_slices,enroll_patchs

def data_SQ_from_pkl(filepath):
    data_dict = load_pickle(filepath)
        
    test_embs = data_dict['test_embs']
    test_labels = data_dict['test_labels']
    test_audios = data_dict['test_audios']
    enroll_embs = data_dict['enroll_embs']
    enroll_labels = data_dict['enroll_labels']
    enroll_audios = data_dict['enroll_audios']

    return test_embs,test_labels,test_audios,enroll_embs,enroll_labels,enroll_audios

def find_matching_positions(list1, list2):
    set_list2 = set(map(tuple, list2))
    matching_positions = [i for i, vector in enumerate(list1) if tuple(vector) in set_list2]
    return matching_positions

def analyze_data(data):
    unique_labels, counts = np.unique(data, return_counts=True)

    # Calculate additional information
    num_unique_labels = len(unique_labels)
    min_appearances = np.min(counts)
    max_appearances = np.max(counts)
    average_appearances = np.mean(counts)

    # Print the results (optional)
    print(f"Number of unique labels: {num_unique_labels}")
    print(f"Minimum appearances of a label: {min_appearances}")
    print(f"Maximum appearances of a label: {max_appearances}")
    print(f"Average appearances of a label: {average_appearances}")


def CL2N_embeddings(enroll_embs, test_embs, use_mean=True, use_std=False, eps=1e-10):

    all_embs = np.concatenate((enroll_embs,test_embs),axis=1)

    if use_mean:
        all_embs = all_embs - np.expand_dims(all_embs.mean(axis=1),1)
    
    if use_std:
        all_embs = all_embs / (all_embs.std(axis=1) + eps)

    embs_l2_norm = np.expand_dims(np.linalg.norm(all_embs, ord=2, axis=-1), axis=-1)
    all_embs = all_embs / embs_l2_norm
    
    enroll_embs = all_embs[:,:enroll_embs.shape[1]]
    test_embs = all_embs[:,enroll_embs.shape[1]:]

    return enroll_embs,test_embs


def plot_embeddings(celeb_avg,movie_avg,movies_avg_adapted,theta):
    if theta is None:
        theta='None'
    fig = plt.figure(figsize=(21, 7))
    # Histogram plot
    plt.subplot(1, 2, 1)
    plt.hist(movie_avg, bins=50, alpha=0.5, label='Movie')
    plt.hist(movies_avg_adapted, bins=50, alpha=0.5, label='Movie new')
    plt.hist(celeb_avg, bins=50, alpha=0.5, label='Celeb')
    plt.legend(loc='upper right')
    plt.title(f'Histogram of Movie and Celeb for alpha:{theta}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # KDE plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(movie_avg, fill=True, label='Movie')
    sns.kdeplot(movies_avg_adapted, fill=True, label='Movie new')
    sns.kdeplot(celeb_avg, fill=True, label='Celeb')
    plt.legend(loc='upper right')
    plt.title(f'KDE of Movie and Celeb for alpha:{theta}')
    plt.xlabel('Value')
    plt.ylabel('Density')

    plt.tight_layout()
    #plt.show()
    fig.savefig(f'plot_alpha_{theta}.png')

def tsne_domains_per_class(features_1,features_2,labels_1,labels_2):
    uniq_classes = sorted(list(set(labels_1)))

    for label in uniq_classes:
        celeb_cls_indices = np.where(labels_1 == label)[0]
        movies_cls_indices = np.where(labels_2 == label)[0]

        celeb_samples = features_1[celeb_cls_indices]
        movies_samples = features_2[movies_cls_indices]

        # Print shapes for verification
        print(f"Class {label}:")
        print(f"Celeb samples shape: {celeb_samples.shape}")
        print(f"Movies samples shape: {movies_samples.shape}")

        # Combine samples and create domain labels
        combined_samples = np.vstack((celeb_samples, movies_samples))
        domain_labels = np.array([0]*celeb_samples.shape[0] + [1]*movies_samples.shape[0])

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(combined_samples)

        # Calculate centroids
        celeb_centroid = tsne_results[domain_labels == 0].mean(axis=0)
        movies_centroid = tsne_results[domain_labels == 1].mean(axis=0)
        reunion_centroid = tsne_results[:].mean(axis=0)

        # Distance centroids 

        centroid1 = celeb_samples.mean(axis=0)
        centroid2 = movies_samples.mean(axis=0)
        distance = np.linalg.norm(centroid1 - centroid2)
        print(distance)

        # Plot t-SNE
        plt.figure(figsize=(10, 6))
        plt.scatter(tsne_results[domain_labels == 0, 0], tsne_results[domain_labels == 0, 1], label='Celeb Domain', alpha=0.6)
        plt.scatter(tsne_results[domain_labels == 1, 0], tsne_results[domain_labels == 1, 1], label='Movies Domain', alpha=0.6)


        # Plot centroids
        plt.scatter(celeb_centroid[0], celeb_centroid[1], label='Celeb Centroid', color='blue', marker='X', s=100)
        plt.scatter(movies_centroid[0], movies_centroid[1], label='Movies Centroid', color='orange', marker='X', s=100)
        #plt.scatter(reunion_centroid[0], reunion_centroid[1], label='Global Centroid', color='black', marker='X', s=100)

        plt.title(f't-SNE plot for class {label}')
        plt.legend()
        plt.show()

def tsne_domains_multi_class(features_1,features_2,labels_1,labels_2,classes):
    # Combine samples from the selected classes
    celeb_samples = []
    movies_samples = []
    celeb_labels = []
    movies_labels = []

    for label in classes:
        # Celeb domain samples
        celeb_cls_indices = np.where(labels_1 == label)
        celeb_samples.append(features_1[celeb_cls_indices[0]])
        celeb_labels.extend([label] * features_1[celeb_cls_indices[0]].shape[0])
        
        # Movies domain samples
        movies_cls_indices = np.where(labels_2 == label)
        movies_samples.append(features_2[movies_cls_indices[0]])
        movies_labels.extend([label] * features_2[movies_cls_indices[0]].shape[0])

    # Convert to numpy arrays
    celeb_samples = np.vstack(celeb_samples)
    movies_samples = np.vstack(movies_samples)
    combined_samples = np.vstack((celeb_samples, movies_samples))

    celeb_labels = np.array(celeb_labels)
    movies_labels = np.array(movies_labels)
    combined_labels = np.concatenate((celeb_labels, movies_labels))

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(combined_samples)

    # Plot t-SNE
    plt.figure(figsize=(12, 8))

    # Create a color map
    colors = plt.cm.get_cmap('tab10', len(classes))

    for idx, label in enumerate(classes):
        # Celeb domain points
        celeb_indices = np.where(celeb_labels == label)[0]
        plt.scatter(tsne_results[celeb_indices, 0], tsne_results[celeb_indices, 1], 
                    label=f'Class {label} (Celeb)', color=colors(idx), marker='o', alpha=0.2)
        
        # Movies domain points
        movies_indices = np.where(movies_labels == label)[0]
        plt.scatter(tsne_results[movies_indices + len(celeb_labels), 0], tsne_results[movies_indices + len(celeb_labels), 1], 
                    label=f'Class {label} (Movies)', color=colors(idx), marker='^', alpha=1)

    plt.title('t-SNE plot for 5 random classes from both domains')
    plt.legend()
    plt.show()

def tsne_query_support(x_q,x_s,y_q,y_s,classes):
    # Combine samples from the selected classes
    support_samples = []
    query_samples = []
    support_labels = []
    query_labels = []
    
    for label in classes:
        # Celeb domain samples

        support_cls_indices = np.where(y_s == label)
        support_samples.append(x_s[support_cls_indices[0]])
        support_labels.extend([label] * x_s[support_cls_indices[0]].shape[0])
        
    # Convert to numpy arrays
    query_samples = x_q
    support_samples = np.vstack(support_samples)
    combined_samples = np.vstack((query_samples, support_samples))

    query_labels = y_q
    support_labels = np.array(support_labels)
    combined_labels = np.concatenate((query_labels, support_labels))

    perplexity = min(30, combined_samples.shape[0] - 1)

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity,random_state=90)#42)
    tsne_results = tsne.fit_transform(combined_samples)

    # Define marker styles for each class
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    marker_dict = {label: markers[idx % len(markers)] for idx, label in enumerate(np.unique(support_labels))}

    # Plot t-SNE
    plt.figure(figsize=(12, 8))

    # Plot query samples in black
    plt.scatter(tsne_results[:len(query_samples), 0], tsne_results[:len(query_samples), 1], c='black', label='Query Samples', marker='x')

    # Plot support samples with different shapes for each class
    unique_labels = np.unique(support_labels)
    for label in unique_labels:
        indices = np.where(support_labels == label)
        plt.scatter(tsne_results[len(query_samples) + indices[0], 0], tsne_results[len(query_samples) + indices[0], 1], label=f'Support Class {label}', marker=marker_dict[label])

    plt.legend()
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Query and Support Samples')
    plt.show()

def calculate_centroids(enroll_embs,enroll_labels):
    # Returns [n_tasks,n_ways,192] tensor with the centroids
    # sampled_classes: [n_tasks,n_ways]
    
    sampled_classes=sorted(list(set(enroll_labels)))
    avg_enroll_embs = []
    for i,label in enumerate(sampled_classes):
        indices = np.where(enroll_labels == label)
        embedding = (enroll_embs[indices[0]].sum(axis=0).squeeze()) / len(indices[0])
    
        avg_enroll_embs.append(embedding)

    avg_enroll_embs = np.asarray(avg_enroll_embs)

    return avg_enroll_embs