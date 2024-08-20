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
import seaborn as sns
import os
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import umap



#import logomaker #only to make a logo graph


# # Load Integrated Gradients (IG) labels and sequences into separate numpy arrays

# In[3]:


### Load data from numpy files
###Takes IG data from xyzIG.npy file and seq + label data from xyz_sequences_label.csv file


# #Mason
# ig_data =r"/Users/isaacdaviet/Desktop/mason_igfold_models/masonIG.npy"
# labeled_seq_data = r"/Users/isaacdaviet/Desktop/mason_igfold_models/mason_sequences_label.csv"



#import logomaker #only to make a logo graph


# # Load Integrated Gradients (IG) labels and sequences into separate numpy arrays

# In[3]:


### Load data from numpy files
###Takes IG data from xyzIG.npy file and seq + label data from xyz_sequences_label.csv file


# #Mason
# ig_data =r"/Users/isaacdaviet/Desktop/mason_igfold_models/masonIG.npy"
# labeled_seq_data = r"/Users/isaacdaviet/Desktop/mason_igfold_models/mason_sequences_label.csv"

# labeled_seq_data = pd.read_csv(labeled_seq_data)

# pt = np.load(ig_data) #load Ig data

# labels = labeled_seq_data.iloc[:,1].to_numpy()
# labels = np.array(labels)

# sequ = labeled_seq_data.iloc[:,0].to_numpy()
# sequ = np.array(sequ)

# print



# In[4]:


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

def flatten_from_files(integrated_gradients_file, labeled_sequences):

    labeled_seq_data = pd.read_csv(labeled_sequences)

    pt = np.load(integrated_gradients_file) #load Ig data

    labels = labeled_seq_data.iloc[:,1].to_numpy()
    labels = np.array(labels)

    df = flatten_labeled_data(pt, labels)
    df['Sequences'] = labeled_seq_data['sequence']

    return df


# # UMAP Reduction

# ### UMAP calculations

# In[5]:

def umap_reduction(umap_df, n_components=2, n_neighbors=55, min_dist=0.05, metric='euclidean', random_state=42, save_results_csv = 'n', save_folder = None, file_name = None):
    """
    Perform UMAP dimensionality reduction on the input DataFrame.

    Parameters:
    - umap_df (pandas.DataFrame): DataFrame containing flattened data and labels.
    - n_components (int): Number of components for the reduced representation.
    - n_neighbors (int): Number of neighbors to consider during UMAP construction.
    - min_dist (float): Minimum distance between points in the reduced space.
    - metric (str): Distance metric to use for UMAP.
    - random_state (int): Seed for reproducibility.

    Returns:
    pandas.DataFrame: DataFrame containing the UMAP-reduced data with labels.
    """
    # print(metric, n_components, n_neighbors, min_dist)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )

    umap_result = reducer.fit_transform(umap_df.iloc[:, :-1])  # Assuming the last column is 'Labels'

    umap_result = pd.DataFrame(umap_result, columns=[f'UMAP-{i+1}' for i in range(n_components)])

    labels = pd.DataFrame(umap_df['Labels'], columns=['Labels'])

    umap_result = pd.concat([umap_result, labels], axis = 1)

    if save_results_csv == 'y':
        file_name = f'{file_name}.csv'
        
        csv_folder = os.path.join(save_folder, 'csv_files') 
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)
        umap_result.to_csv(os.path.join(csv_folder, file_name), index = False)

    return umap_result

# umap_df = flatten_labeled_data(pt, labels)
# umap_result = umap_reduction(umap_df, n_components = 2)
# print(umap_result.head)


# ### 2D & 3D UMAP Plotting

# In[17]:


def plot_umap(umap_result, component1=1, component2=2, metric='euclidean', n_neighbors = None, min_dist = None, show_pts='all', show_graph='y', save_graph=None, save_path=None, project_name=None, iterate_all_pt_types = 'n', close_plot = 'y'):
    """
    Plot UMAP representation with optional filtering based on binders/non-binders.

    Parameters:
    - umap_result (pandas.DataFrame): DataFrame containing UMAP-reduced data with labels.
    - component1 (int): The first UMAP component.
    - component2 (int): The second UMAP component.
    - show_pts (str): Filter option for 'binders' or 'non-binders'.
    - show_graph (str): Display the graph or not.

    Returns:
    None, only prints the plot.
    """
        
    plt.figure(figsize=(10, 8))
    binder_color = 'red'
    non_binder_color = 'blue'
    title_ext = ''

    if show_pts == 'binders':
        umap_result = umap_result[umap_result['Labels'] == 'Binder']
        title_ext = 'BINDERS ONLY'
    elif show_pts == 'non-binders':
        umap_result = umap_result[umap_result['Labels'] == 'Non Binder']
        title_ext = 'NON BINDERS ONLY'

    all_dataframes = [umap_result]
    title_extensions = ['all_points']

    if iterate_all_pt_types == 'y':
        all_dataframes.append(umap_result[umap_result['Labels'] == 'Binder'])
        all_dataframes.append(umap_result[umap_result['Labels'] == 'Non Binder'])

        title_extensions.append('BINDERS_ONLY')
        title_extensions.append('NON-BINDERS_ONLY')
    
    for i in range(0, len(all_dataframes)):
        umap_result = all_dataframes[i]
        title_ext = title_extensions[i]
    
        plot = sns.scatterplot(x=f'UMAP-{component1}', y=f'UMAP-{component2}',  hue='Labels', data=umap_result,
                        palette={'Binder': binder_color, 'Non Binder': non_binder_color}, s=5)

        plt.title(f'{metric} UMAP Dimensionality Reduction - {n_neighbors} Neighbors - min_dist = {min_dist} - {title_ext}')

        if save_graph == 'y':
            umap_folder = os.path.join(save_path, 'UMAP_analysis')
            if not os.path.exists(umap_folder):
                os.makedirs(umap_folder)

            umap_subfolder = os.path.join(umap_folder, f'{metric}_distance')
            if not os.path.exists(umap_subfolder):
                os.makedirs(umap_subfolder)

            plt.savefig(os.path.join(umap_subfolder, f'{project_name}_{metric}_UMAP_{component1}-{component2}_{title_ext}.png'))

        if show_graph == 'y':
            plt.show()

        if close_plot == 'y':
            plt.close()

    return plt


# In[7]:


def show_all_umap_combos(umap_result, metric='euclidean', show_pts=None, show_graph='y', save_graph=None, save_path=None, project_name=None):
    '''
    Computes and graphs all possible UMAP component combinations based on the number of components in umap_result.

    Only for quick investigation purposes. Does not return anything.
    '''
    combos = []
    n_components = umap_result.shape[1] - 1  # Subtract 1 for the 'Labels' column
    
    for i in range(1, n_components + 1):
        for j in range (1, n_components + 1):
            if i != j and (i, j) not in combos and (j, i) not in combos:
                combos.append((i, j))

                plot_umap(umap_result, i, j, metric=metric, show_pts=show_pts, show_graph=show_graph, save_graph=save_graph, save_path=save_path, project_name=project_name)

# Examples
# plot_umap(umap_result, show_pts = 'binders', show_graph='n', save_graph='y', save_path='/Users/isaacdaviet/Desktop/thesis', project_name='test')
# plot_umap(umap_result, show_pts='binders')
# plot_umap(umap_result, show_pts='non-binders')
# plot_umap(umap_result, component1=1, component2=3)  # Example with different UMAP components
                
# show_all_umap_combos(umap_result, show_graph='n', save_graph='y', save_path='/Users/isaacdaviet/Desktop/thesis', project_name='umap_test')


# In[8]:


def umap_pair_plot(umap_result, metric='euclidean', n_components='2', n_neighbors = None, min_dist=None, show_pts=None, show_graph='y', save_graph=None, save_path=None, project_name=None, iterate_all_pt_types='n'):

    custom_palette = {'Binder': 'red', 'Non Binder': 'blue'}

    if iterate_all_pt_types == 'y':
        show_pts = ['all_pts', 'binders', 'non_binders']
    if show_pts == None:
        show_pts =['all_pts'] 

    all_dataframes = []
    for type in show_pts:
        if type == 'all_pts':
            all_dataframes.append(umap_result)
        if type == 'binders':
            all_dataframes.append(umap_result[umap_result['Labels'] == 'Binder'])
        if type == 'non_binders':
            all_dataframes.append(umap_result[umap_result['Labels'] == 'Non Binder'])

    for i in range(0, len(all_dataframes)):
        df = all_dataframes[i]
        title_ext = show_pts[i]

        plt.figure(figsize=(240, 180))
        sns.pairplot(df, hue='Labels', palette=custom_palette, plot_kws={'s': 3}, height=8, aspect=1.5)

        plt.title(f'{metric} UMAP Pair-Plot - N={n_neighbors} - MD={min_dist} - Components 1-{n_components} - {title_ext}')

        if save_graph == 'y':
            umap_folder = os.path.join(save_path, 'UMAP_analysis')
            if not os.path.exists(umap_folder):
                os.makedirs(umap_folder)

            umap_subfolder = os.path.join(umap_folder, f'{metric}_distance')
            if not os.path.exists(umap_subfolder):
                os.makedirs(umap_subfolder)

            pp_subfolder = os.path.join(umap_subfolder, 'pair_plts')
            if not os.path.exists(pp_subfolder):
                os.makedirs(pp_subfolder)

            plt.savefig(os.path.join(pp_subfolder, f'{project_name}_UMAP_{metric}_Comp-1-{n_components}_N-{n_neighbors}_MD-{min_dist}_{title_ext}.png'))

        if show_graph == 'y':
            plt.show()

        plt.close()



# umap_pair_plot(umap_result, 'euclidian', 3, show_graph='y', save_graph='n', save_path='/Users/isaacdaviet/Desktop/thesis', project_name='umap_test', iterate_all_pt_types = 'n')


# In[9]:


def umap_iterate_neighbors(umap_df, starting_neighbors=10, final_neighbors=200, step=5, min_dist = 0.1, metric = 'euclidean', show_pts='all', show_graph='y', save_graph=None, save_path=None, project_name=None, iterate_all_pt_types='y'):

    current_neighbors = starting_neighbors

    while current_neighbors <= final_neighbors:
        umap_result = umap_reduction(umap_df, n_neighbors = current_neighbors, min_dist = min_dist, metric = metric)

        if show_graph == 'y':
            plot_umap(umap_result, metric = metric, n_neighbors = current_neighbors, min_dist = min_dist, show_pts = show_pts, show_graph=show_graph, save_graph=save_graph, save_path=save_path, project_name=project_name, iterate_all_pt_types = iterate_all_pt_types)
            current_neighbors += step


# umap_df = flatten_labeled_data(pt, labels)
# umap_neighbor_iteration = umap_iterate_neighbors(umap_df)


# In[24]:


def umap_iterate_min_dist(umap_df, starting_dist=0, final_dist=1, step=0.05, n_neighbors = 15, metric = 'euclidean', show_pts='all', show_graph='y', save_graph=None, save_path=None, project_name=None, iterate_all_pt_types='n'):

    current_dist = starting_dist
    results = []

    while current_dist <= final_dist:
        umap_result = umap_reduction(umap_df, min_dist = current_dist, n_neighbors = n_neighbors, metric = metric)

        results.append(umap_result)


        if show_graph == 'y':
            plot_umap(umap_result, metric = metric, n_neighbors = n_neighbors,min_dist = current_dist, show_pts = show_pts, show_graph='y', save_graph=save_graph, save_path=save_path, project_name=project_name, iterate_all_pt_types = iterate_all_pt_types)
            current_dist += step


# umap_df = flatten_labeled_data(pt, labels)
# umap_neighbor_iteration = umap_iterate_min_dist(umap_df, final_dist=4)


# In[76]:


def umap_parameter_comparison_calculation(umap_df, start_neighbors=5, end_neighbors=20, neighbor_step=5, start_dist=0, end_dist=0.1, dist_step=0.05, metric = 'euclidean'):
    """
    Perform UMAP parameter comparison for different neighbors and minimum distances.

    Parameters:
    - umap_df (DataFrame): The input DataFrame containing UMAP data.
    - start_neighbors (int, optional): Starting number of neighbors for UMAP. Defaults to 5.
    - end_neighbors (int, optional): Ending number of neighbors for UMAP. Defaults to 20.
    - neighbor_step (int, optional): Step size for incrementing neighbors. Defaults to 5.
    - start_dist (float, optional): Starting minimum distance for UMAP. Defaults to 0.
    - end_dist (float, optional): Ending minimum distance for UMAP. Defaults to 0.1.
    - dist_step (float, optional): Step size for incrementing minimum distance. Defaults to 0.05.
    - metric (str, optional): The distance metric used for UMAP. Defaults to 'euclidean'.

    Returns:
    - compare_results (list): List of UMAP results for different combinations of neighbors and minimum distances.
    """
    current_dist = start_dist
    current_neighbor = start_neighbors

    neighbors_lst = []
    dist_lst = []

    compare_results = []

    while current_dist <= end_dist:
        dist_lst.append(current_dist)
        current_dist += dist_step

    while current_neighbor <= end_neighbors:
        neighbors_lst.append(current_neighbor)
        current_neighbor += neighbor_step

    for neigh in neighbors_lst:
        for dst in dist_lst:
            umap_result = umap_reduction(umap_df, min_dist = dst, n_neighbors = neigh, metric = metric)
      
            compare_results.append([neigh, dst, umap_result])

    return compare_results



def umap_parameter_sub_plt(compare_results, metric='euclidean', show_pts='all', show_graph = 'y', save_graph=None, save_path=None, project_name=None, plt_size = [200, 150]):
    """
    Visualize UMAP parameter comparison results in subplots.

    Parameters:
    - compare_results (list): List of UMAP results to be visualized.
    - metric (str, optional): The distance metric used for UMAP. Defaults to 'euclidean'.
    - show_pts (str, optional): Type of points to show ('all', 'binders', 'non binders'). Defaults to 'all'.
    - show_graph (str, optional): Whether to display the graph ('y' or 'n'). Defaults to 'y'.
    - save_graph (str, optional): Whether to save the graph ('y' or 'n'). Defaults to None.
    - save_path (str, optional): Path to save the graph. Defaults to None.
    - project_name (str, optional): Name of the project for saving the graph. Defaults to None.

    Returns:
    - None
    """
    custom_palette = {'Binder': 'red', 'Non Binder': 'blue'}

    # Calculate the number of rows and columns based on the length of compare_results

    num_n = []
    num_d = []



    # Extract neighbors and dist to add to lists
    for i in compare_results:
        num_n.append(i[0])
        num_d.append(i[1])

    num_rows = int(len(set(num_n)))
    num_cols = int(len(set(num_d)))

    if save_graph == 'y':
        
        min_n = round(min(num_n), 2)
        min_d= round(min(num_d), 2)
        max_n = round(max(num_n), 2)
        max_d= round(max(num_d), 2)

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(nrows = num_rows, ncols=num_cols, figsize=(plt_size[0], plt_size[1]))

    # If there's only one subplot, axs is a single Axes object, not an array
    if num_rows == 1 and num_cols == 1:
        axs = np.array([[axs]])

    title_ext = '_all'
    counter = 0
    # Loop through each subplot
    for i in range(0, num_rows):
        for j in range(0, num_cols):
    
            result = compare_results[counter]
            n = result[0]
            d = result[1]
            df = result[2]

            if show_pts == 'binders':
                df = df[df['Labels'] == 'Binder']
                title_ext = '_BINDERS ONLY'
            elif show_pts == 'non binders':
                df = df[df['Labels'] == 'Non Binder']
                title_ext = '_NON BINDERS ONLY'
        
            sns.scatterplot(x='UMAP-1', y='UMAP-2',  hue='Labels',palette=custom_palette, data=df, s=5, ax=axs[i, j])


            axs[i, j].legend()

            counter +=1

    plt.suptitle(f'{metric} UMAP Parameter Comparison', fontsize=16)

    # Adjust layout for better spacing
    plt.tight_layout()

    if save_graph == 'y':
        umap_folder = os.path.join(save_path, 'UMAP_analysis')
        if not os.path.exists(umap_folder):
            os.makedirs(umap_folder)

        umap_subfolder = os.path.join(umap_folder, f'{metric}_distance')
        if not os.path.exists(umap_subfolder):
            os.makedirs(umap_subfolder)

        plt.savefig(os.path.join(umap_subfolder, f'{project_name}-UMAP_{metric}-ParamCompar_N{min_n}-{max_n}_MD{min_d}-{max_d}{title_ext}.png'))

    if show_graph == 'y':
        plt.show()
    plt.show()

def umap_param_comp(umap_df, start_neighbors=5, end_neighbors=20, neighbor_step=5, start_dist=0, end_dist=0.1, dist_step=0.05, metric = 'euclidean', show_pts='all', show_graph='y', save_graph=None, save_path=None, project_name=None, iterate_all_pt_types='n', plt_size = [200, 150]):
    """
    Perform UMAP parameter comparison and generate subplots for visualization.

    Parameters:
    - umap_df (DataFrame): The input DataFrame containing UMAP data.
    - start_neighbors (int, optional): Starting number of neighbors for UMAP. Defaults to 5.
    - end_neighbors (int, optional): Ending number of neighbors for UMAP. Defaults to 20.
    - neighbor_step (int, optional): Step size for incrementing neighbors. Defaults to 5.
    - start_dist (float, optional): Starting minimum distance for UMAP. Defaults to 0.
    - end_dist (float, optional): Ending minimum distance for UMAP. Defaults to 0.1.
    - dist_step (float, optional): Step size for incrementing minimum distance. Defaults to 0.05.
    - metric (str, optional): The distance metric used for UMAP. Defaults to 'euclidean'.
    - show_pts (str, optional): Type of points to show ('all', 'binders', 'non binders'). Defaults to 'all'.
    - show_graph (str, optional): Whether to display the graph ('y' or 'n'). Defaults to 'y'.
    - save_graph (str, optional): Whether to save the graph ('y' or 'n'). Defaults to None.
    - save_path (str, optional): Path to save the graph. Defaults to None.
    - project_name (str, optional): Name of the project for saving the graph. Defaults to None.
    - iterate_all_pt_types (str, optional): Whether to iterate over all point types ('y' or 'n'). Defaults to 'n'.
    """
    results = umap_parameter_comparison_calculation(umap_df, start_neighbors=start_neighbors, end_neighbors=end_neighbors, neighbor_step=neighbor_step, start_dist=start_dist, end_dist=end_dist, dist_step=dist_step, metric = metric)

    pt_list = [show_pts]

    if iterate_all_pt_types == 'y':
        pt_list = ['all', 'binders', 'non binders']

    for item in pt_list:
        umap_parameter_sub_plt(results, metric=metric, show_pts=item, show_graph = show_graph, save_graph=save_graph, save_path=save_path, project_name=project_name, plt_size = plt_size)

# umap_df = flatten_labeled_data(pt, labels)

# compare_results = umap_parameter_comparison_calculation(umap_df, start_neighbors=5, end_neighbors=10, neighbor_step=5, start_dist=0, end_dist=0.05, dist_step=0.05)


# umap_param_comp(umap_df, save_graph='y', save_path='/Users/isaacdaviet/Desktop/thesis', project_name='umap_test', iterate_all_pt_types='y')

# In[]:
def umap_optimized_pair_plts_csv(ig_data, labeled_seqs, save_folder, selected_parameters_csv, n_components = 5, points_to_show = ['all_pts'], project_name = None):

    df = flatten_from_files(ig_data, labeled_seqs) 
    selection = pd.read_csv(selected_parameters_csv)
    results = {}

    print(selection)

    for index, row in selection.iterrows():
        metric = row['distance metric']
        n_neighbors = row['n_neighbors']
        min_dist = row['min_dist']

        print(f'\nCALCULATING:\n\tMetric: {metric} \n\tNeighbors: {n_neighbors} \n\tMin_Dist {min_dist} ') # Sanity check

        key = f'{metric}_Comp-{n_components}_N-{n_neighbors}_MD-{min_dist}'

        result = umap_reduction(df, n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)

        print(result.head()) # Sanity check

        results[key]= result

        umap_pair_plot(result, metric=metric, n_components=n_components, n_neighbors = n_neighbors, min_dist = min_dist,  show_pts=points_to_show, show_graph = 'n', save_graph='y', save_path=save_folder, project_name=project_name)

    return results

# In [ ]:
def umap_optimized_3d_pplt_csv(ig_data, labeled_seqs, save_folder, selected_parameters_csv, points_to_show = ['all_pts'], project_name=None):

    df = flatten_from_files(ig_data, labeled_seqs) 
    selection = pd.read_csv(selected_parameters_csv)
    results = {}

    print(selection)

    for index, row in selection.iterrows():
        metric = row['distance metric']
        n_neighbors = row['n_neighbors']
        min_dist = row['min_dist']

        print(f'\nCALCULATING:\n\tMetric: {metric} \n\tNeighbors: {n_neighbors} \n\tMin_Dist {min_dist} ') # Sanity check

        key = f'{metric}_Comp-3_N-{n_neighbors}_MD-{min_dist}'

        result = umap_reduction(df, n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)

        print(result.head()) # Sanity check

        results[key]= result

        for type in points_to_show:        
            plot_3d_umap(result, metric = metric, n_neighbors=n_neighbors, min_dist = min_dist, show_pts = type, graph_size=(1000, 700), point_size=2, fig_show = 'n', fig_save = 'y', save_path=save_folder, project_name=project_name)

        umap_pair_plot(result, metric=metric, n_components=3, n_neighbors = n_neighbors, min_dist = min_dist,  show_pts=points_to_show, show_graph = 'n', save_graph='y', save_path=save_folder, project_name=project_name)

    return results

# In[ ]:
def umap_final_plts_csv(ig_data, labeled_seqs, selected_parameters_csv, points_to_show = ['all_pts', 'binders', 'non_binders'], plt_size = [60, 20], pt_size = 2, fontsize=64, decimanl_axes = 'y', save_graph = 'y', save_folder = None, project_name=None, save_3d_graph = 'y', save_results_csv = 'y'):

    ### Set graphing colors
    custom_palette = {'Binder': 'red', 'Non Binder': 'blue'}


    graphing_results_df = pd.DataFrame(columns=['graph id','umap file', 'distance metric', 'n_components', 'n_neighbors', 'min_dist', 'component_1', 'component_2', '3D graph?'])


    df = flatten_from_files(ig_data, labeled_seqs)
    selection = pd.read_csv(selected_parameters_csv)

    print(selection)

    for index, row in selection.iterrows():

        ### Extract all parameters from cvs file
        comp_list = []
        graph_type = row['graph']
        metric = row['distance metric']
        n_components = int(round(row['n_components']))
        n_neighbors = int(round(row['n_neighbors']))
        min_dist = float(round(row['min_dist'], 2))

        comp1 = int(row['component_1'])
        comp_list.append(comp1)
        comp2 = int(row['component_2'])
        comp_list.append(comp2)

        ### to screen for 2D vs 3D entries
        if graph_type == '3D':
            comp3 = int(row['component_3'])
            comp_list.append(comp3)

        else:
            comp3 = None

        ### Calculate UMAP results
        print(f'\nCALCULATING:\n\tMetric: {metric} \n\tn_components: {n_components}\n\tNeighbors: {n_neighbors} \n\tMin_Dist: {min_dist} \n\tComponent 1: {comp1} \n\tComponent 2: {comp2}\n\tComponent 3: {comp3}') # Sanity check

        file_name = f'{project_name}-FinalSelect_UMAP-{metric}_nC{n_components}_Ne{n_neighbors}_MD{min_dist}'


        result = umap_reduction(df, n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42, save_results_csv = save_results_csv, save_folder = save_folder, file_name = file_name)


        print(result.head()) # Sanity check


        ### create list of all possible component combinations
        combos = []
        for i in comp_list:
            for j in comp_list:
                if i != j and (i,j) not in combos and (j,i) not in combos:
                    combos.append((i,j))



        ### Isolate binders/non-binders/all point dataframes
        all_dfs = [] 
        for type in points_to_show:
            if type == 'binders':
                df_type = result[result['Labels'] == 'Binder']

            elif type == 'non_binders':
                df_type = result[result['Labels'] == 'Non Binder']
            else:
                df_type = result

            all_dfs.append(df_type)



        ### Plot all combos at all the given point types
        for combo in combos:
            # print(combos, '\n',combo) # Sanity Check
            c1 = combo[0]
            c2 = combo[1]
            counter = 0

            fig, axs = plt.subplots(nrows = 1, ncols = len(points_to_show), figsize=(plt_size[0], plt_size[1]))

            for sel_df, ax in zip(all_dfs, axs):
                # print(sel_df)
                sns.scatterplot(x=f'UMAP-{c1}', y=f'UMAP-{c2}',  hue='Labels', palette=custom_palette, data=sel_df, s=pt_size, ax=ax)
                ax.set_title(points_to_show[counter])

                # Set custom tick locations and labels
                if decimanl_axes == 'y':
                    hold1 = f'UMAP-{c1}'
                    hold2= f'UMAP-{c2}'

                    x_range = np.arange(sel_df[hold1].min(), sel_df[hold1].max() + 0.1, 0.1)
                    if len(x_range) > 1:  # Check if the range is valid
                        ax.set_xticks(x_range)
                        ax.set_xticklabels(np.round(x_range, 1), rotation=90)
                    
                    y_range = np.arange(sel_df[hold2].min(), sel_df[hold2].max() + 0.1, 0.1)
                    if len(y_range) > 1:  # Check if the range is valid
                        ax.set_yticks(y_range)
                        ax.set_yticklabels(np.round(y_range, 1))


                counter += 1
            graphing_results_df = pd.concat([graphing_results_df, pd.DataFrame({
                'graph id': [f'{project_name}-{metric}-{n_components}-{n_neighbors}-{min_dist}-{c1}-{c2}'],
                'umap file': [f'{file_name}.csv'],
                'distance metric': [metric],
                'n_components': [n_components],
                'n_neighbors': [n_neighbors],
                'min_dist': [min_dist],
                'component_1': [c1],
                'component_2': [c2],
                '3D graph?': [graph_type]
                })])

            plt.suptitle(f'{metric} UMAP -- nComp = {n_components} -- nNeighbors = {n_neighbors} -- Min_Dist = {min_dist} -- Cmps {c1} & {c2}', fontsize=fontsize)
            plt.tight_layout()

            ### Add to results dataframe for easy analysis

        
            ### Saving the resulting graph
            if save_graph == 'y':
        
                umap_folder = os.path.join(save_folder, 'UMAP_analysis')
                if not os.path.exists(umap_folder):
                    os.makedirs(umap_folder)

                final_plt_subfolder = os.path.join(save_folder, 'final_graphs')
                if not os.path.exists(final_plt_subfolder):
                    os.makedirs(final_plt_subfolder)

                metric_subfolder = os.path.join(final_plt_subfolder, f'{metric}_distance')
                if not os.path.exists(metric_subfolder):
                    os.makedirs(metric_subfolder)


                plt.savefig(os.path.join(metric_subfolder, f'{project_name}-FinalSelect_UMAP-{metric}_nC{n_components}_Ne{n_neighbors}_MD{min_dist}_Cmp{c1}-{c2}.png'))

        if graph_type == '3D' and save_3d_graph == 'y':

            for type in points_to_show:
                final_plt_subfolder = os.path.join(save_folder, 'final_graphs')
                if not os.path.exists(final_plt_subfolder):
                    os.makedirs(final_plt_subfolder)

                plot_3d_umap(result, metric = metric, n_neighbors=n_neighbors, min_dist = min_dist, show_pts = type, graph_size=(1000, 700), point_size=2, fig_show = 'n', fig_save = 'y', save_path=final_plt_subfolder, project_name=project_name)

    if save_results_csv == 'y':
        results_csv_path = os.path.join(save_folder, 'results.csv')
        graphing_results_df.to_csv(results_csv_path, index=False)

# In []:
def plot_3d_umap(umap_result, metric = None, n_neighbors=None, min_dist = None, show_pts = None, graph_size=(1000, 700), point_size=5, fig_show = 'y', fig_save = 'y', save_path=None, project_name=None):
    """
    Plot 3D UMAP representation with optional filtering based on binders/non-binders.

    Parameters:
    - umap_result (pandas.DataFrame): DataFrame containing 3D UMAP-reduced data with labels.
    - show_pts (str): Filter option for 'binders' or 'non-binders'.
    - graph_size (tuple): Size of the plot in pixels.
    - point_size (int): Size of the markers in the plot.

    Returns:
    None
    """

    binders_color = 'red'
    non_binders_color = 'blue'

    if show_pts == 'binders':
        binders_opacity, non_binders_opacity = 1.0, 0.0
        title_ext = ' - BINDERS ONLY'
    elif show_pts == 'non-binders':
        binders_opacity, non_binders_opacity = 0.0, 1.0
        title_ext = ' - NON BINDERS ONLY'
    else:
        binders_opacity, non_binders_opacity = 1.0, 1.0
        title_ext = ''

    fig = go.Figure()


    for label, color, opacity in [('Binder', binders_color, binders_opacity), ('Non Binder', non_binders_color, non_binders_opacity)]:
        subset = umap_result[umap_result['Labels'] == label]
        fig.add_trace(go.Scatter3d(
            x=subset['UMAP-1'],
            y=subset['UMAP-2'],
            z=subset['UMAP-3'],
            mode='markers',
            marker=dict(size=point_size, color=color),
            opacity=opacity,
            name=label
        ))

    # Update layout for better visualization
    fig.update_layout(
        title=f'3D UMAP Dimensionality Reduction{title_ext}',
        scene=dict(
            xaxis_title='UMAP-1',
            yaxis_title='UMAP-2',
            zaxis_title='UMAP-3'
        ),
        width=graph_size[0],
        height=graph_size[1]
    )

    if fig_show == 'y':
        fig.show()

    if fig_save == 'y':
        umap_folder = os.path.join(save_path, 'UMAP_analysis')
        if not os.path.exists(umap_folder):
            os.makedirs(umap_folder)

        umap_subfolder = os.path.join(umap_folder, f'{metric}_distance')
        if not os.path.exists(umap_subfolder):
            os.makedirs(umap_subfolder)

        subfolder_3d = os.path.join(umap_subfolder, '3D_graphs')
        if not os.path.exists(subfolder_3d):
            os.makedirs(subfolder_3d)

        fig.write_html(os.path.join(subfolder_3d, f'{project_name}_UMAP_{metric}_Comp-1-3_N-{n_neighbors}_MD-{min_dist}_{title_ext}.html'))
        
    # fig.close




# umap_3d_default = umap_reduction(umap_df, n_components = 3)
# plot_3d_umap(umap_3d_default, point_size = 2)
# plot_3d_umap(umap_3d_default, point_size = 2)


# ### Exctracting Data Points from 2D UMAP graph

# In[ ]:


# test_vertices = np.asarray([(10, 0),#bottom left
#                        (15, 0),#bottom right
#                        (10, 2.5),#top right
#                        (15, 2.5)])#top left

test_vertices = np.asarray([(-2.5, 7.5),#bottom left
                       (2.5, 7.5),#bottom right
                       (-2.5, 13),#top right
                       (2.5, 13)])#top left

def umap_selection_2d(umap_result, vertices, component1=1, component2=2, metric='euclidean', show_pts = None, show_graph= 'y', save_graph = None, save_path = None, selection_name = None):

    binder_color = 'red'
    non_binder_color = 'blue'
    
    title_ext = ''

    if show_pts == 'binders':
        umap_result = umap_result[umap_result['Labels'] == 'Binder']
        title_ext = ' - BINDERS ONLY'
    elif show_pts == 'non binders':
        umap_result = umap_result[umap_result['Labels'] == 'Non Binder']
        title_ext = ' - NON BINDERS ONLY'

    hull = ConvexHull(vertices)
    vertices = vertices[hull.vertices]

    path = Path(vertices)
    points = umap_result[['UMAP-1', 'UMAP-2']].values
    mask = path.contains_points(points)

    umap_result['mask'] = mask
    
    if show_graph == 'y' or save_graph == 'y':
        plt.figure(figsize=(20, 15))
        
        ax = sns.scatterplot(x=f'UMAP-{component1}', y=f'UMAP-{component2}', hue='mask', data=umap_result, s=5)

        ax.set_frame_on(False)
        ax.locator_params(nbins=50, axis='x')
        ax.locator_params(nbins=50, axis='y')

        plt.title(f'{selection_name} - {metric} UMAP Reduction PC{component1} & {component2}{title_ext}')


        if save_graph == 'y':
            selection_folder = os.path.join(save_path, 'UMAP_selections')
            if not os.path.exists(selection_folder):
                os.makedirs(selection_folder)

            plt.savefig(os.path.join(selection_folder, f'{selection_name}_UMAP{component1}_{component2}.png'))

        if show_graph == 'y':
            plt.show()

    plt.close()

    all_selected = [i for i in umap_result.index[mask]]
    binder_rows = [i for i in umap_result.index[mask] if umap_result.loc[i, 'Labels'] == 'Binder']
    nonbinder_rows = [i for i in umap_result.index[mask] if umap_result.loc[i, 'Labels'] == 'Non Binder']

    print(f'Number of Points Selected: {len(all_selected)}    --    Number of Binders: {len(binder_rows)}    --    Number of Non Binders: {len(nonbinder_rows)}')

    return all_selected, binder_rows, nonbinder_rows, umap_result


# umap_selection_2d(umap_result, test_vertices)
# umap_selection_2d(umap_result, test_vertices, show_pts = 'binders')
# umap_selection_2d(umap_result, test_vertices, show_pts = 'non binders')

# umap_selection_2d(umap_result, test_vertices, component1=1, component2=3, metric='euclidean', show_pts = None, show_graph= 'n', save_graph = 'y', save_path = '/Users/isaacdaviet/Desktop/thesis', selection_name = 'test')

