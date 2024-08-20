#!/usr/bin/env python
# coding: utf-8

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
import umap


#import logomaker #only to make a logo graph


# # Load Integrated Gradients (IG) labels and sequences into separate numpy arrays

# In[3]:


### Load data from numpy files
###Takes IG data from xyzIG.npy file and seq + label data from xyz_sequences_label.csv file


#Mason
# ig_data =r"/Users/isaacdaviet/Desktop/mason_igfold_models/masonIG.npy"
# labeled_seq_data = r"/Users/isaacdaviet/Desktop/mason_igfold_models/mason_sequences_label.csv"

# labeled_seq_data = pd.read_csv(labeled_seq_data)

# pt = np.load(ig_data) #load Ig data

# labels = labeled_seq_data.iloc[:,1].to_numpy()
# labels = np.array(labels)

# sequ = labeled_seq_data.iloc[:,0].to_numpy()
# sequ = np.array(sequ)

# print


# In[48]:


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


# # t-SNE Reduction

# In[50]:


def tsne_calculate(df, labels, n_components=2, perplexity=30, exaggeration = 12,metric = 'euclidean'):
    tsne = TSNE(n_components=n_components, perplexity=perplexity,early_exaggeration= exaggeration, random_state=42, metric = metric)
    tsne_transf = tsne.fit_transform(df.drop('Labels', axis=1))

    tsne_df = pd.DataFrame(tsne_transf, columns=[f'tSNE_Dim-{i+1}' for i in range(0, n_components)])

    labels_column = np.where(labels > 0.5, 'Binder', 'Non Binder')
    tsne_df['Labels'] = labels_column

    return tsne_df


# tsne_df = flatten_labeled_data(pt, labels)
# tsne_result = tsne_calculate(tsne_df, labels, n_components = 2)


# In[51]:


def tsne_plot(tsne_result, perplexity=30, component1=1, component2=2, show_pts='all', show_graph='y', save_graph=None, save_path=None, project_name=None):
    binder_color = 'red'
    non_binder_color = 'blue'


    if show_pts == 'binders':
        tsne_result = tsne_result[tsne_result['Labels'] == 'Binder']
        show_pts = 'BINDERS ONLY'
    elif show_pts == 'non binders':
        tsne_result = tsne_result[tsne_result['Labels'] == 'Non Binder']
        show_pts = 'NON BINDERS ONLY'

    plt.figure(figsize=(15, 10))

    sns.scatterplot(x=f'tSNE_Dim-{component1}', y=f'tSNE_Dim-{component2}', hue='Labels', data=tsne_result, palette={'Binder': 'red', 'Non Binder': 'blue'}, s=5)

    plt.title(f't-SNE - Dimensions {component1} & {component2} - Perplexity={perplexity} - {show_pts}')

    if save_graph == 'y':
        tsne_folder = os.path.join(save_path, 'tsne_analysis')
        if not os.path.exists(tsne_folder):
            os.makedirs(tsne_folder)

        tsne_subfolder = os.path.join(tsne_folder, f'{show_pts}_data_points')
        if not os.path.exists(tsne_subfolder):
            os.makedirs(tsne_subfolder)

        plt.savefig(os.path.join(tsne_subfolder, f'{project_name}_t-SNE_P{perplexity}_D{component1}&{component2}_{show_pts}.png'))

    if show_graph == 'y':
        plt.show()

    plt.close()


# Example usage:
# tsne_plot(tsne_result, perplexity=30, show_pts='all')


# In[52]:


def show_all_tsne_combos(tsne_result, perplexity = 30, show_pts=None, show_graph='y', save_graph=None, save_path=None, project_name=None):
    '''
    Computes and graphs all possible UMAP component combinations based on the number of components in umap_result.

    Only for quick investigation purposes. Does not return anything.
    '''
    combos = []
    n_components = tsne_result.shape[1] - 1  # Subtract 1 for the 'Labels' column
    
    for i in range(1, n_components + 1):
        for j in range (1, n_components + 1):
            if i != j and (i, j) not in combos and (j, i) not in combos:
                combos.append((i, j))

                tsne_plot(tsne_result, perplexity = perplexity, component1=i, component2=j, show_pts=show_pts, show_graph=show_graph, save_graph=save_graph, save_path=save_path, project_name=project_name)

# show_all_tsne_combos(tsne_result)


# In[60]:


def tnse_perplexity_iteration(tsne_df, labels, starting_perplexity=5, final_perplexity=50, step=5, show_pts='all', show_graph='y', save_graph=None, save_path=None, project_name=None, iterate_all_pt_types='n'):
    current_perplexity = starting_perplexity

    while current_perplexity <= final_perplexity:
        tsne_result = tsne_calculate(tsne_df, labels, perplexity=current_perplexity)

        tsne_plot(tsne_result, perplexity=current_perplexity, component1=1, component2=2, show_pts=show_pts, show_graph=show_graph, save_graph=save_graph, save_path=save_path, project_name=project_name)

        current_perplexity += step
        # if iterate_all_pt_types.lower() == 'y':
        #     for pt_type in ['binders', 'non binders']:
        #         tsne_plot(tsne_result, perplexity=current_perplexity, show_pts=pt_type, show_graph=show_graph, save_graph=save_graph, save_path=save_path, project_name=project_name)

# Example usage:
# tnse_perplexity_iteration(tsne_df, labels)

# In[61]
def tsne_parameter_comparison_calculation(tsne_df, start_perplx=500, end_perplx=1000, perplx_step=100, start_exag=0, end_exag=20, exag_step=5, metric = 'euclidean'):

    labels = pd.DataFrame(tsne_df['Labels'], columns=['Labels'])


    current_exag = start_exag
    current_perplx = start_perplx

    perplx_lst = []
    exag_lst = []

    compare_results = []

    while current_exag <= end_exag:
        current_exag += exag_step
        exag_lst.append(current_exag)
    while current_perplx <= end_perplx:
        current_perplx += perplx_step
        perplx_lst.append(current_perplx)


    for perp in perplx_lst:
        for ex in exag_lst:
            tsne_result = tsne_calculate(tsne_df, labels, perplexity = perp, exaggeration = ex, metric = metric)
      
            compare_results.append([perp, ex, tsne_result])

    return compare_results



def tsne_parameter_sub_plt(compare_results, metric='euclidean', show_pts='all', show_graph = 'y', save_graph=None, save_path=None, project_name=None, plt_size = [200, 150]):

    custom_palette = {'Binder': 'red', 'Non Binder': 'blue'}

    # Calculate the number of rows and columns based on the length of compare_results

    num_p = []
    num_e = []

    # Extract perplx and exageration to add to lists
    for i in compare_results:
        num_p.append(i[0])
        num_e.append(i[1])

    num_rows = int(len(set(num_p)))
    num_cols = int(len(set(num_e)))

    if save_graph == 'y':
        min_n = round(min(num_p), 2)
        min_d= round(min(num_e), 2)
        max_n = round(max(num_p), 2)
        max_d= round(max(num_e), 2)

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(nrows = num_rows, ncols=num_cols, figsize=(plt_size[0], plt_size[1]))

    # If there's only one subplot, axs is a single Axes object, not an array
    if num_rows == 1 and num_cols == 1:
        axs = np.array([[axs]])

    title_ext = ''
    counter = 0
    # Loop through each subplot
    for i in range(0, num_rows):
        for j in range(0, num_cols):
    
            result = compare_results[counter]
            p = result[0]
            e = result[1]
            df = result[2]

            if show_pts == 'binders':
                df = df[df['Labels'] == 'Binder']
                title_ext = ' - BINDERS ONLY'
            elif show_pts == 'non binders':
                df = df[df['Labels'] == 'Non Binder']
                title_ext = ' - NON BINDERS ONLY'
        
            sns.scatterplot(x='tsne-1', y='tsne-2',  hue='Labels',palette=custom_palette, data=df, s=5, ax=axs[i, j])

            axs[i, j].set_title(f'n_perplx={round(p,2)} - min_exag={round(e,2)} - {title_ext}')
            axs[i, j].legend()

            counter +=1

    plt.suptitle(f'{metric} tsne Parameter Comparison', fontsize=16)

    # Adjust layout for better spacing
    plt.tight_layout()

    if save_graph == 'y':
        tsne_folder = os.path.join(save_path, 'tsne_analysis')
        if not os.path.exists(tsne_folder):
            os.makedirs(tsne_folder)

        tsne_subfolder = os.path.join(tsne_folder, f'{metric}_exagance')
        if not os.path.exists(tsne_subfolder):
            os.makedirs(tsne_subfolder)

        plt.savefig(os.path.join(tsne_subfolder, f'{project_name}-tsne_{metric}-ParamCompar_Perplx_{min_n}-{max_d}_Exag_{min_d}-{max_d}_{title_ext}.png'))

    if show_graph == 'y':
        plt.show()
    plt.show()

def tsne_param_comp(tsne_df, start_perplx=5, end_perplx=20, perplx_step=5, start_exag=0, end_exag=0.1, exag_step=0.05, metric = 'euclidean', show_pts='all', show_graph='y', save_graph=None, save_path=None, project_name=None, iterate_all_pt_types='n', plt_size = [200, 150]):

    results = tsne_parameter_comparison_calculation(tsne_df, start_perplx=start_perplx, end_perplx=end_perplx, perplx_step=perplx_step, start_exag=start_exag, end_exag=end_exag, exag_step=exag_step, metric = metric)

    pt_list = [show_pts]

    if iterate_all_pt_types == 'y':
        pt_list = ['all', 'binders', 'non binders']

    for item in pt_list:
        tsne_parameter_sub_plt(results, metric=metric, show_pts=item, show_graph = show_graph, save_graph=save_graph, save_path=save_path, project_name=project_name, plt_size = plt_size)

# tsne_df = flatten_labeled_data(pt, labels)

# compare_results = tsne_parameter_comparison_calculation(tsne_df, start_perplx=5, end_perplx=10, perplx_step=5, start_exag=0, end_exag=0.05, dist_step=0.05)


# tsne_param_comp(tsne_df, save_graph='y', save_path='/Users/isaacdaviet/Desktop/thesis', project_name='tsne_test', iterate_all_pt_types='y')
