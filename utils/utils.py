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
    #test_dict['concat_labels'] = np.repeat(test_dict['concat_labels'],10,0)
    #test_dict['concat_features'] = np.repeat(test_dict['concat_features'],10,0)
    #test_dict['concat_slices'] = np.repeat(test_dict['concat_slices'],10,0)
    #print(test_dict['concat_labels'].shape)
    #print(test_dict['concat_features'].shape)
    #print(test_dict['concat_slices'].shape)

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



def embedding_normalize(embs, use_mean=True,use_std=False, eps=1e-10):
    """
    Mean and l2 length normalize the input speaker embeddings

    Args:
        embs: embeddings of shape (Batch,emb_size)
    Returns:
        embs: normalized embeddings of shape (Batch,emb_size)
    """
    initial_shape = embs.copy().shape
    if len(initial_shape) == 3:
        embs =embs.reshape((-1, embs.shape[-1]))

    if use_mean:
        embs = embs - embs.mean(axis=0)
    
    if use_std:
        embs = embs / (embs.std(axis=0) + eps)
    embs_l2_norm = np.expand_dims(np.linalg.norm(embs, ord=2, axis=-1), axis=1)
    embs = embs / embs_l2_norm

    if len(initial_shape) == 3:
        embs = embs.reshape(initial_shape)

    return embs

def embs_norm_both(test_embs, enroll_embs, mean=None,use_std=False, eps=1e-10):
    """
    Mean and l2 length normalize the input speaker embeddings

    Args:
        embs: embeddings of shape (Batch,emb_size)
    Returns:
        embs: normalized embeddings of shape (Batch,emb_size)
    """
    if mean is not None:
        test_embs = test_embs - mean
        enroll_embs = enroll_embs - mean
    
    if use_std:
        test_embs = test_embs / (test_embs.std(axis=0) + eps)
        enroll_embs = enroll_embs / (enroll_embs.std(axis=0) + eps)
    
    test_embs_l2_norm = np.expand_dims(np.linalg.norm(test_embs, ord=2, axis=-1), axis=-1)
    test_embs = test_embs / test_embs_l2_norm

    enroll_embs_l2_norm = np.expand_dims(np.linalg.norm(enroll_embs, ord=2, axis=-1), axis=-1)
    enroll_embs = enroll_embs / enroll_embs_l2_norm

    return test_embs, enroll_embs

def CL2N_embeddings(enroll_embs, test_embs, use_mean=True, use_std=False, eps=1e-10):

    all_embs = np.concatenate((enroll_embs,test_embs),axis=1)

    initial_shape = all_embs.copy().shape
    #if len(initial_shape) == 3:
    #    all_embs = all_embs.reshape((-1, all_embs.shape[-1]))

    if use_mean:
        all_embs = all_embs - np.expand_dims(all_embs.mean(axis=1),1)
    
    if use_std:
        all_embs = all_embs / (all_embs.std(axis=1) + eps)

    embs_l2_norm = np.expand_dims(np.linalg.norm(all_embs, ord=2, axis=-1), axis=-1)
    all_embs = all_embs / embs_l2_norm

    #if len(initial_shape) == 3:
    #    all_embs = all_embs.reshape(initial_shape)
    
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

def class_compute_transform_A(x_q,y_q,x_s,y_s, theta=0):
    uniq_classes = sorted(list(set(y_s)))
    
    sum_s1 = 0
    sum_s2 = 0
    count = 0
    for label in uniq_classes:
        old_indices_q = np.where(y_q == label)
        old_indices_s = np.where(y_s == label)

        indices_q = []
        indices_s = []
        n = len(old_indices_s[0])
        
        if len(old_indices_q[0]) > n:
            indices_q.append(old_indices_q[0][:n])
            indices_q.append(None)
        else:
            indices_q.append(old_indices_q[0])
            indices_q.append(None)

        if len(old_indices_s[0]) > n:
            indices_s.append(old_indices_s[0][:n])
            indices_s.append(None)
        else:
            indices_s.append(old_indices_s[0])
            indices_s.append(None)

        if len(indices_q[0]) > len(indices_s[0]):
            samples_q = x_q[indices_q[0]]
            initial_samples_s = x_s[indices_s[0]]

            target_size = len(indices_q[0])
            original_array_size = len(indices_s[0])

            repeat_factor = target_size // original_array_size
            remainder = target_size % original_array_size

            # Repeat the rows
            oversampled_array = np.repeat(initial_samples_s, repeat_factor, axis=0)

            # Add additional random rows if there is a remainder
            if remainder > 0:
                additional_rows = initial_samples_s[np.random.choice(original_array_size, remainder, replace=True)]
                oversampled_array = np.vstack([oversampled_array, additional_rows])

            samples_s = oversampled_array
            
        else:
            initial_samples_q = x_q[indices_q[0]]
            samples_s = x_s[indices_s[0]]

            target_size = len(indices_s[0])
            original_array_size = len(indices_q[0])

            repeat_factor = target_size // original_array_size
            remainder = target_size % original_array_size

            # Repeat the rows
            oversampled_array = np.repeat(initial_samples_q, repeat_factor, axis=0)

            # Add additional random rows if there is a remainder
            if remainder > 0:
                additional_rows = initial_samples_q[np.random.choice(original_array_size, remainder, replace=True)]
                oversampled_array = np.vstack([oversampled_array, additional_rows])

            samples_q = oversampled_array

        count += samples_q.shape[0]

        for i in range(samples_q.shape[0]):
            sum_s1 += np.matmul(np.expand_dims(samples_s[i],0).T,np.expand_dims(samples_q[i],0))
            sum_s2 += np.matmul(np.expand_dims(samples_q[i],0).T,np.expand_dims(samples_q[i],0))

    print(count)
    return sum_s1,sum_s2
    sum_s2 += np.eye(192)*theta
    matrix_inverse = np.linalg.inv(sum_s2)
    A = np.matmul(sum_s1,matrix_inverse)

    return A     


def class_compute_diagonal_A(x_q,y_q,x_s,y_s, theta=0):
    uniq_classes = sorted(list(set(y_s)))
    
    sum_up = np.zeros((192,192))
    sum_down = np.zeros((192,192))
    count = 0
    count_nocor = 0
    A = np.zeros((192,192))

    for label in uniq_classes:
        old_indices_q = np.where(y_q == label)
        old_indices_s = np.where(y_s == label)

        indices_q = []
        indices_s = []
        n = len(old_indices_s[0])
        
        if len(old_indices_q[0]) > n:
            indices_q.append(old_indices_q[0][:n])
            indices_q.append(None)
        else:
            indices_q.append(old_indices_q[0])
            indices_q.append(None)

        if len(old_indices_s[0]) > n:
            indices_s.append(old_indices_s[0][:n])
            indices_s.append(None)
        else:
            indices_s.append(old_indices_s[0])
            indices_s.append(None)

        if len(indices_q[0]) > len(indices_s[0]):
            samples_q = x_q[indices_q[0]]
            initial_samples_s = x_s[indices_s[0]]

            target_size = len(indices_q[0])
            original_array_size = len(indices_s[0])

            repeat_factor = target_size // original_array_size
            remainder = target_size % original_array_size

            # Repeat the rows
            oversampled_array = np.repeat(initial_samples_s, repeat_factor, axis=0)

            # Add additional random rows if there is a remainder
            if remainder > 0:
                additional_rows = initial_samples_s[np.random.choice(original_array_size, remainder, replace=True)]
                oversampled_array = np.vstack([oversampled_array, additional_rows])

            samples_s = oversampled_array
            
        else:
            initial_samples_q = x_q[indices_q[0]]
            samples_s = x_s[indices_s[0]]

            target_size = len(indices_s[0])
            original_array_size = len(indices_q[0])

            repeat_factor = target_size // original_array_size
            remainder = target_size % original_array_size

            # Repeat the rows
            oversampled_array = np.repeat(initial_samples_q, repeat_factor, axis=0)

            # Add additional random rows if there is a remainder
            if remainder > 0:
                additional_rows = initial_samples_q[np.random.choice(original_array_size, remainder, replace=True)]
                oversampled_array = np.vstack([oversampled_array, additional_rows])

            samples_q = oversampled_array

        count += samples_q.shape[0]

        samples_number = samples_q.shape[0] 
        embeddings_length = A.shape[0] # 192
        
        for i in range(samples_number):
            for j in range(embeddings_length): 
                sum_up[j][j] += samples_s[i][j]*samples_q[i][j]
                sum_down[j][j] += samples_q[i][j]*samples_q[i][j]
                if samples_s[i][j]*samples_q[i][j] < 0:
                    count_nocor += 1
    #print(sum_up)
    #print(sum_down)
    #print(count)
    #print(count_nocor)
    #print(count_nocor/count)
    for j in range(embeddings_length):
        A[j,j] = sum_up[j][j]/(sum_down[j][j]+theta)

    return A     

def class_compute_sums_A(x_q,y_q,x_s,y_s):
    uniq_classes = sorted(list(set(y_s)))
    
    sum_up = np.zeros((192,192))
    norm_sum_up_s = np.zeros((192,192))
    norm_sum_up_q = np.zeros((192,192))
    norm_sum_down = np.zeros((192,192))
    sum_down = np.zeros((192,192))
    count = 0
    count_nocor = 0
    A = np.zeros((192,192))
    uncorr = []
    for label in uniq_classes:
        old_indices_q = np.where(y_q == label)
        old_indices_s = np.where(y_s == label)

        indices_q = []
        indices_s = []
        n = len(old_indices_s[0])
        
        if len(old_indices_q[0]) > n:
            indices_q.append(old_indices_q[0][:n])
            indices_q.append(None)
        else:
            indices_q.append(old_indices_q[0])
            indices_q.append(None)

        if len(old_indices_s[0]) > n:
            indices_s.append(old_indices_s[0][:n])
            indices_s.append(None)
        else:
            indices_s.append(old_indices_s[0])
            indices_s.append(None)

        if len(indices_q[0]) > len(indices_s[0]):
            samples_q = x_q[indices_q[0]]
            initial_samples_s = x_s[indices_s[0]]

            target_size = len(indices_q[0])
            original_array_size = len(indices_s[0])

            repeat_factor = target_size // original_array_size
            remainder = target_size % original_array_size

            # Repeat the rows
            oversampled_array = np.repeat(initial_samples_s, repeat_factor, axis=0)

            # Add additional random rows if there is a remainder
            if remainder > 0:
                additional_rows = initial_samples_s[np.random.choice(original_array_size, remainder, replace=True)]
                oversampled_array = np.vstack([oversampled_array, additional_rows])

            samples_s = oversampled_array
            
        else:
            initial_samples_q = x_q[indices_q[0]]
            samples_s = x_s[indices_s[0]]

            target_size = len(indices_s[0])
            original_array_size = len(indices_q[0])

            repeat_factor = target_size // original_array_size
            remainder = target_size % original_array_size

            # Repeat the rows
            oversampled_array = np.repeat(initial_samples_q, repeat_factor, axis=0)

            # Add additional random rows if there is a remainder
            if remainder > 0:
                additional_rows = initial_samples_q[np.random.choice(original_array_size, remainder, replace=True)]
                oversampled_array = np.vstack([oversampled_array, additional_rows])

            samples_q = oversampled_array

        count += samples_q.shape[0]

        samples_number = samples_q.shape[0] 
        embeddings_length = A.shape[0] # 192
        samples_up = []
        samples_down = []
        class_up = np.zeros((192,192))
        class_down = np.zeros((192,192))
        norm_class_s = np.zeros((192,192))
        norm_class_q = np.zeros((192,192))
        
        class_cross_corr_list = []
        class_negatives = []
        for i in range(samples_number):
            sample_uncor = 0

            for j in range(embeddings_length): 
                class_up[j][j] += samples_s[i][j]*samples_q[i][j]
                class_down[j][j] += samples_q[i][j]*samples_q[i][j]
                norm_class_s[j][j] += (samples_s[i][j]**2)
                norm_class_q[j][j] += (samples_q[i][j]**2)

                sum_up[j][j] += samples_s[i][j]*samples_q[i][j]
                norm_sum_up_s[j][j] += (samples_s[i][j]**2)
                norm_sum_up_q[j][j] += (samples_q[i][j]**2)
                
                sum_down[j][j] += samples_q[i][j]*samples_q[i][j]
                norm_sum_down[j][j] += samples_q[i][j]*samples_q[i][j]
        """
        class_negatives.append(np.sum(class_up<0))
        class_cross_corr = np.nan_to_num(class_up/(np.sqrt(norm_class_q)*np.sqrt(norm_class_s)))
        class_cross_corr_list.append(class_cross_corr)
        #print(class_cross_corr)

    class_cross_corr_list = np.array(class_cross_corr_list)
    norm_class_cross_corr_list = np.sqrt(np.sum(class_cross_corr_list**2,axis=0))
    class_cross_corr_list = np.sum(class_cross_corr_list,axis=0)
    
    print(np.max(class_cross_corr_list))

    normed_class_cross_corr_list = np.nan_to_num(class_cross_corr_list/norm_class_cross_corr_list)
    print(normed_class_cross_corr_list)
    print(class_cross_corr_list)
    avg_class_cross_corr = np.sum(class_cross_corr_list)/class_cross_corr_list.shape[0]
    print(avg_class_cross_corr)
    avg_normed_class_cross_corr = np.sum(normed_class_cross_corr_list)/normed_class_cross_corr_list.shape[0]
    print(avg_normed_class_cross_corr)
    min_list = []
    for j in range(192):
        min_list.append(class_cross_corr_list[j][j])

    print(min(min_list))
    sum_down = np.identity(192)
    print(sum_down)

    return class_cross_corr_list, sum_down
    """
    #print(sum(class_negatives)/len(class_negatives))   
    #print(np.sum(sum_up<0)) 
    normed_sum_up = np.nan_to_num(sum_up/(np.sqrt(norm_sum_up_q)*np.sqrt(norm_sum_up_s)))
    #print(normed_sum_up)
    avg_cross_corr = np.sum(normed_sum_up)/normed_sum_up.shape[0]
    #print(avg_cross_corr)
    
    min_list = []
    for j in range(192):
        min_list.append(normed_sum_up[j][j])

    #print(min(min_list))

    #print(np.max(normed_sum_up))
    #normed_sum_down = np.nan_to_num(sum_down/norm_sum_down)
    #print(normed_sum_down)
    #avg_auto_corr = np.sum(normed_sum_down)/normed_sum_up.shape[0]
    #print(avg_auto_corr)
    
    #print(sum_up/(np.sqrt(norm_sum_up_q)*np.sqrt(norm_sum_up_s)))
    #print(sum_down/norm_sum_up_q)
    #print(sum(samples_up)/len(samples_up))
    #print(sum(samples_down)/len(samples_down))
    #print(np.trace(sum_up/(np.sqrt(norm_sum_up_q)*np.sqrt(norm_sum_up_s))))

    #print(sum_up)
    #print(sum_down)
    #print(x_q.shape)
    #print(x_s.shape)
    #print(count)
    #print(count_nocor)
    #print(len(uncorr))
    #print(sum(uncorr)/len(uncorr))
    #print(min(uncorr))
    #print(max(uncorr))

    return sum_up,sum_down

def huber_loss(input, target, delta=0.5):
    error = input - target
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return loss.mean()

def median_absolute_deviation(data):
    """
    Calculate the Median Absolute Deviation (MAD) of the data.
    
    Parameters:
    data (np.ndarray): Input data.
    
    Returns:
    float: The MAD of the data.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return mad

def choose_delta_robust(y_true, y_pred, k=1.0):
    """
    Choose delta parameter for Huber loss based on the Median Absolute Deviation (MAD) of the errors.
    
    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.
    k (float): A constant to scale the MAD.
    
    Returns:
    float: The chosen delta value.
    """
    error = y_true - y_pred
    mad = median_absolute_deviation(error)
    delta = k * mad
    return delta

def iterative_A(x_q,mu_N, alpha=0, seuil=1e-2, nitm=5000, prec=[1e-4, 1e-4]):
 
    N = x_q.shape[-1]
 
    logger.info(f'Alpha regularization param: {alpha}')
    logger.info(f'Dimension of features {N}') # 192
    logger.info(f'Shape of domain to be transfered: {x_q.shape}') # x_q: [no_samples, 192]
    logger.info(f'Shape of corresponding centroids for each sample: {mu_N.shape}') # mu_N: [no_samples, 192]

    Rc = x_q.T @ x_q
    Rd = mu_N.T @ x_q

    d = np.linalg.eigvals(Rc).max()
    gam = 1.9/d

    A = np.eye(N)
    indA = np.where(A == 0)

    delta_robust = choose_delta_robust(x_q, mu_N, k=1.0)
    print(delta_robust)

    crit = []
    for nit in range(nitm):
        #crit_val = np.linalg.norm(x_q @ A.T - mu_N, 'fro') ** 2 / 2 + alpha * np.sum(np.abs(A))
        data = x_q @ A.T
        crit_val = F.huber_loss(torch.tensor(data,dtype=torch.float32), torch.tensor(mu_N,dtype=torch.float32),reduction="mean") ** 2 / 2 + alpha * np.sum(np.abs(A))
        #crit_val = huber_loss(torch.tensor(data,dtype=torch.float32), torch.tensor(mu_N,dtype=torch.float32),delta=delta_robust) ** 2 / 2 + alpha * np.sum(np.abs(A))
        crit_val = crit_val.item()
        
        crit.append(crit_val)
        logger.info(f'{nit + 1} : crit = {crit_val}')

        grad = A @ Rc.T - Rd # CHANGEEEE
        A = A - gam * grad
        A[indA] = np.maximum(np.abs(A[indA]) - gam * alpha * np.ones(len(indA[0])), 0)

        if nit > 0 and abs(crit_old - crit_val) <= prec[0] * crit_old:
            break
        crit_old = crit_val

    As = A.copy()
    
    logger.info('*** debiasing estimator ***')
    indseuil = np.where(np.abs(A) < seuil)
    A[indseuil] = 0
    for nit in range(nitm):
        #crit2 = np.linalg.norm(x_q @ A.T - mu_N, 'fro') ** 2 / 2
        data = x_q @ A.T
        crit2 = F.huber_loss(torch.tensor(data,dtype=torch.float32), torch.tensor(mu_N,dtype=torch.float32),reduction="mean") ** 2 / 2 
        #crit2 = huber_loss(torch.tensor(data,dtype=torch.float32), torch.tensor(mu_N,dtype=torch.float32),delta=delta_robust) ** 2 / 2
        crit2 = crit2.item()

        logger.info(f'{nit + 1} : crit2 = {crit2}')

        grad = A @ Rc.T - Rd# CHANGEE
        A = A - gam * grad
        indseuil = np.where(np.abs(A) < seuil)
        A[indseuil] = 0

        if nit > 0 and abs(crit_old - crit2) <= prec[1] * crit_old:
            break
        crit_old = crit2

    return A, As, crit
    

def ana_A(y, mu_N, lambda_, seuil=1e-2, nitm=5000, prec=[1e-4, 1e-4]):
    N = mu_N.shape[0]

    Rc = y @ y.T
  
    Rd = mu_N @ y.T

    d = eigs(Rc, k=1, which='LM', return_eigenvectors=False)[0]
    d = np.real(d)
    print(d)
    gam = 1.9 / d

    print('*** computing sparse solution ****')
    A = np.eye(N)
    # Af= A.flatten()
    indA = np.where(A == 0)
   

    crit = []
    for nit in range(nitm):
        
        current_crit = np.linalg.norm(A @ y - mu_N, 'fro')**2 / 2 + lambda_ * np.sum(np.abs(A))
        print(current_crit)
        crit.append(current_crit)
        print(f'{nit + 1} : crit = {current_crit}')

        grad = A @ Rc - Rd
        A = A - gam * grad
        A[indA] = np.maximum(np.abs(A[indA]) - gam * lambda_, 0)

        if nit > 0 and abs(critold - current_crit) <= prec[0] * critold:
            break
        critold = current_crit

    As = A.copy()

    print('*** debiasing estimator ***')
    indseuil = np.where(np.abs(A) < seuil)
    A[indseuil] = 0
    for nit in range(nitm):
        crit2 = np.linalg.norm(A @ y - mu_N, 'fro')**2 / 2
        print(f'{nit + 1} : crit = {crit2}')

        grad = A @ Rc - Rd
        A = A - gam * grad
        indseuil = np.where(np.abs(A) < seuil)
        A[indseuil] = 0

        if nit > 0 and abs(critold - crit2) <= prec[1] * critold:
            break
        critold = crit2

    return A, As, crit

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

def coral(Ds, Dt, alpha):

    initial_shape = Ds.shape

    Ds = np.reshape(Ds,(-1,192))
    Dt = np.reshape(Dt, (-1,192))
    
    Cs = Ds.T @ Ds + alpha*np.eye(Ds.shape[-1])
    Ct = Dt.T @ Dt + alpha*np.eye(Ds.shape[-1])

    print(Ds.shape)
    print(Cs.shape)
    A = fractional_matrix_power(Cs,-0.5) @ fractional_matrix_power(Ct,0.5)
    
    return A