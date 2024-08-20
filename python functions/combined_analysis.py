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

# In[1]:


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
import umap

import sys
sys.path.append('/Users/isaacdaviet/Desktop/thesis/python versions')

import pdb_extraction as extract
import onehot_encoded as onehot
import umap_calc as umap
from umap_calc import flatten_labeled_data
from umap_calc import flatten_from_files
import pca_calc as pca
import tSNE_calc as tsne

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


# # Full Dimensionality Reduction Exploartion and Download

# In[30]:


def load_data(ig_data, labeled_seq_data):
    ### Load data from IG and labeled sequence files
    labeled_seq_data = pd.read_csv(labeled_seq_data)

    pt = np.load(ig_data) #load Ig data

    labels = labeled_seq_data.iloc[:,1].to_numpy()
    labels = np.array(labels)

    sequ = labeled_seq_data.iloc[:,0].to_numpy()
    sequ = np.array(sequ)

    return pt, labels, sequ

def pca_pair_plot_exploration(ig_data, labeled_seq_data, project_name, save_path, iterate_pt_types = 'y'):
    
    pt, labels, sequ = load_data(ig_data, labeled_seq_data)

    analysis_folder = os.path.join(save_path, f'{project_name}_DimRed_Analysis')
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)

    pca.pca_pair_plot(pt, labels, 10, show_graph='n', save_graph='y', save_path=analysis_folder, project_name=project_name, iterate_all_pt_types = iterate_pt_types)



def full_pca_exploration(ig_data, labeled_seq_data, project_name, save_path):

    pt, labels, sequ = load_data(ig_data, labeled_seq_data)

    ### Create new folder to contain all resulting graphs
    analysis_folder = os.path.join(save_path, f'{project_name}_DimRed_Analysis')
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)

    ### Save bar plot of explained variance for first 20 PCA components
    pca.pca_explained_variance_bar_plot(pt, n_components = 20, show_graph = 'n', save_graph = 'y', save_path = analysis_folder, project_name = project_name)


    ### Download all possible combinations for the first 10 PCA components
    pca_options = ['all', 'binders', 'non binders']

    for option in pca_options:
        pca.show_all_pc_combos(pt, labels, sequ, 10, show_pts=option, show_graph='n', save_graph='y', save_path=analysis_folder, project_name=project_name)


def full_umap_exploration(ig_data, labeled_seq_data, project_name, save_path):
    
    ### load and flatten data into data frame
    pt, labels, sequ = load_data(ig_data, labeled_seq_data)
    umap_df = flatten_labeled_data(pt, labels)

    ### Create new folder to contain all resulting graphs
    analysis_folder = os.path.join(save_path, f'{project_name}_DimRed_Analysis')
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)

    ### Calculate first 5 UMAP components for all possible UMAP metrics
    umap_metrics = ['euclidean', 'manhattan', 'l1', 'cosine', 'correlation', 'hamming', 'jaccard']


    for option in umap_metrics:
        umap_result = umap.umap_reduction(umap_df, n_components=5, n_neighbors=15, min_dist=0.1, metric=option, random_state=42)

        umap.show_all_umap_combos(umap_result, metric=option, show_pts=None, show_graph='n', save_graph='y', save_path=analysis_folder, project_name=project_name)

def full_tsne_exploration(ig_data, labeled_seq_data, project_name, save_path):
    
    ### load and flatten data into data frame
    pt, labels, sequ = load_data(ig_data, labeled_seq_data)
    tsne_df = flatten_labeled_data(pt, labels)

    ### Create new folder to contain all resulting graphs
    analysis_folder = os.path.join(save_path, f'{project_name}_DimRed_Analysis')
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)

    ### Calculate first 5 UMAP components for all possible UMAP metrics
    tsne_metrics = ['euclidean', 'manhattan', 'l1', 'cosine', 'correlation', 'hamming', 'jaccard']


    for option in tsne_metrics:
        tsne_result = tsne.tsne_calculate(tsne_df, n_components=5, n_neighbors=15, min_dist=0.1, metric=option, random_state=42)

        tsne.show_all_tsne_combos(tsne_result, metric=option, show_pts=None, show_graph='n', save_graph='y', save_path=analysis_folder, project_name=project_name)


def umap_pair_plot_exploration(ig_data, labeled_seq_data, project_name, save_path, iterate_all_pp_types = 'y'):

    ### load and flatten data into data frame
    pt, labels, sequ = load_data(ig_data, labeled_seq_data)
    umap_df = flatten_labeled_data(pt, labels)

    ### Create new folder to contain all resulting graphs
    analysis_folder = os.path.join(save_path, f'{project_name}_DimRed_Analysis')
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)

    ### Calculate first 5 UMAP components for all possible UMAP metrics
    umap_metrics = ['euclidean', 'manhattan', 'l1', 'cosine', 'correlation', 'hamming', 'jaccard']

    show_pts_options = ['all', 'binders', 'non binders']


    for option in umap_metrics:
        umap_result = umap.umap_reduction(umap_df, n_components=10, n_neighbors=15, min_dist=0.1, metric=option, random_state=42)

        for points in show_pts_options:
            umap.umap_pair_plot(umap_result, option, 10, show_pts='all', show_graph='n', save_graph='y', save_path=analysis_folder, project_name=project_name)

def the_whole_enchilada(ig_data, labeled_seq_data, project_name, save_path, pair_plots='y', full_plots = 'y'):

    if pair_plots == 'y':
        pca_pair_plot_exploration(ig_data, labeled_seq_data, project_name, save_path)
        umap_pair_plot_exploration(ig_data, labeled_seq_data, project_name, save_path)

    if full_plots == 'y':
        full_pca_exploration(ig_data, labeled_seq_data, project_name, save_path)
        full_umap_exploration(ig_data, labeled_seq_data, project_name, save_path)
        full_tsne_exploration(ig_data, labeled_seq_data, project_name, save_path)


ig_file = r'/Users/isaacdaviet/Desktop/mason_igfold_models/masonIG.npy'
labels_file = r'/Users/isaacdaviet/Desktop/mason_igfold_models/mason_sequences_label.csv'
output_folder = r'/Users/isaacdaviet/Desktop/thesis/'




# full_pca_exploration(ig_file, labels_file, 'mason', output_folder)
# full_umap_exploration(ig_file, labels_file, 'mason', output_folder)

the_whole_enchilada(ig_file, labels_file, 'mason_full_test', output_folder)


