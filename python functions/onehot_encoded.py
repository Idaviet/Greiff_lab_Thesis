#!/usr/bin/env python
# coding: utf-8

# # Stuff from robert

# In[1]:


### load data from torch file. Unncessary if .npy files containing the IG is already done
#import torch
# Mason
#pt = torch.load('mason_unique_results.pt', map_location='cpu').numpy()
#labels = torch.load('mason_unique_label.pt', map_location='cpu').numpy()
#sequ = torch.load('mason_unique_sequence.pt', map_location='cpu').numpy()

# Brij
#pt = torch.load('brij_short_results.pt', map_location='cpu').numpy()
#labels = torch.load('brij_short_label.pt', map_location='cpu').numpy()
#sequ = torch.load('brij_short_sequence.pt', map_location='cpu').numpy()


# pt = torch.load('mason_unique_results.pt', map_location='cpu').numpy()
# pt should be the same as your masonIG (probably a npy file and not a csv?)
# 
# labels = torch.load('mason_unique_label.pt', map_location='cpu').numpy()
# labels should be a number with either 1 or 0, and should be in the mason_sequences_labeled.csv' file
# 
# 
# sequ = torch.load('mason_unique_sequence.pt', map_location='cpu').numpy()
# are the sequences them self and are also in the mason_sequences_labeled.csv' file. However in my code they are a one-hot encoded numpy tensor, while you already have the complete sequences. You dont need the one-hot encoding, so you need to remove the code that transforms the one-hot encoding into a string

# # Imports

# In[2]:


# pip install plotly
#pip install umap-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns
import os
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE


#import logomaker #only to make a logo graph


# # Load Integrated Gradients (IG) labels and sequences into separate numpy arrays

# In[3]:


### Load data from numpy files
###Takes IG data from xyzIG.npy file and seq + label data from xyz_sequences_label.csv file


#Mason
ig_data =r"/Users/isaacdaviet/Desktop/mason_igfold_models/masonIG.npy"
labeled_seq_data = r"/Users/isaacdaviet/Desktop/mason_igfold_models/mason_sequences_label.csv"

labeled_seq_data = pd.read_csv(labeled_seq_data)

pt = np.load(ig_data) #load Ig data

labels = labeled_seq_data.iloc[:,1].to_numpy()
labels = np.array(labels)

sequ = labeled_seq_data.iloc[:,0].to_numpy()
sequ = np.array(sequ)

print


# In[11]:


def flatten_labeled_data(unlabeled_data, labels):
    """
    Flatten the unlabeled IG data and combine it with labels.

    Parameters:
    - unlabeled_data (numpy.ndarray): Input data to be flattened.
    - labels (numpy.ndarray): Labels corresponding to the input data.

    Returns:
    pandas.DataFrame: DataFrame containing flattened data with feature names and labels.
    """
    
    flattened_data = unlabeled_data.reshape(unlabeled_data.shape[0], -1)
    columns = [f'Feature-{i}' for i in range(1, flattened_data.shape[1] + 1)]
    df = pd.DataFrame(flattened_data, columns=columns)
    df['Labels'] = pd.DataFrame(labels, columns=['Labels'])['Labels'].apply(lambda x: 'Binder' if x > 0.5 else 'Non Binder')
    return df


# # Functions to convert IG 1-hot encoded sequences from selection of row positions

# In[22]:


def one_hot_to_one_letter(integrated_gradient_sequence):
    '''
    Function to convert one-hot encoded sequence to one-letter amino acid sequence string.
    '''
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    one_letter_code = ""
    
    for block in integrated_gradient_sequence:
        index = next((i for i, value in enumerate(block) if np.any(value != 0)), -1)
        if index != -1:
            one_letter_code += amino_acids[index]
        else:
            one_letter_code += "X"  # Placeholder for unknown or padding

    return one_letter_code



def onehot_to_oneletter_selection(selection, download_selection = 'n', save_path = None, selection_name = None):
    '''
    Returns one-letter amino acid sequence of entries, using a list of selected rows as input
    
    Can be used as an extension of graph_pca, compute_and_graph_pca and pca_selection
    '''
    selected_cdrh3_seqs = []

    for row in selection:
        ig_entry = pt[row]
        aa_seq = one_hot_to_one_letter(ig_entry)
        selected_cdrh3_seqs.append(aa_seq)

    # if download_selection == 'y':
    #     save_filename = os.path.join(save_path, f'{selection_name}_selected_rows.txt')
    #     with open(save_filename, 'w') as file:
    #         file.write(">Binder Rows:\n")
    #         file.write(','.join(map(str, binder_rows)) + '\n')
    #         file.write(">Non-Binder Rows:\n")
    #         file.write(','.join(map(str, nonbinder_rows)) + '\n')
    
    return selected_cdrh3_seqs
    
# print(onehot_to_oneletter_selection(binders))

