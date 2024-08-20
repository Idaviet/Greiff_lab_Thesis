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
import plotly.express as px
import ast



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


# # PCA Reduction

# ## PCA Computing and 2D Graphing

# In[4]:
def compute_pca(pc_no):
    '''
    Computes PCA based on number of principle components provided
    '''
    pca = PCA(n_components=pc_no)
    return pca


def pca_transformation(pt, pca, pc1, pc2):
    pca_transf = pca.fit_transform(pt.reshape(-1, 20*10))
    pca_transf = pca_transf[:, [pc1-1, pc2-1]]
    return pca_transf


def pca_df(pt, labels, sequ, n_pc, save_csv_filepath = None, project = None):
    """
    Perform PCA on input data and create a DataFrame with transformed values.

    Parameters:
    - pt (numpy.ndarray): Input data array to be transformed using PCA.
    - labels (numpy.ndarray): Array of labels corresponding to each data point.
    - sequ (numpy.ndarray): Array of sequences corresponding to each data point.
    - n_pc (int): Number of principal components to use in PCA.
    - save_csv_filepath (str, optional): Path to the directory where the PCA DataFrame will be saved as a CSV file. Default is None.
    - project (str, optional): Name of the project for labeling the saved CSV file. Default is None.

    Returns:
    pd.DataFrame: DataFrame containing PCA-transformed data along with sequences and labels.

    If save_csv_filepath is provided, the DataFrame is also saved as a CSV file with a filename like "{project}-PCAdf{n_pc}.csv".
    """
       
    pca_df = compute_pca(n_pc)
    pca_transf = pca_df.fit_transform(pt.reshape(-1, 20*10))

    column_names = [f'PCA{i}_ExpVar:{pca_df.explained_variance_ratio_[i-1]*100:.2f}%' for i in range(1, n_pc + 1)]

    pplt_df = pd.DataFrame(pca_transf, columns=column_names)
    pplt_df['Sequences'] = pd.DataFrame(sequ, columns = ['Sequences'])
    pplt_df['Labels'] = pd.DataFrame(labels, columns=['Labels'])['Labels'].apply(lambda x: 'Binder' if x > 0.5 else 'Non Binder')

    if save_csv_filepath:
        file_name = os.path.join(save_csv_filepath, f'{project}-PCAdf{n_pc}.csv')
        pplt_df.to_csv(file_name, index = False)

    return pplt_df

def plotly_2d(pca_df, pc1, pc2, save_graph='y', save_path=None, project = None):
    """
    Generate a 2D scatter plot using Plotly based on selected principal components.

    Parameters:
    - pca_df (pd.DataFrame): DataFrame containing PCA-transformed data generated using 'pca_df' function
    - pc1 (int): Number of the first principal component.
    - pc2 (int): Number of the second principal component.
    - save_graph (str, optional): Specify whether to save the plot as an HTML file ('y' for yes, 'n' for no). Default is 'y'.
    - save_path (str, optional): Path to the directory where the saved HTML file will be stored. Required if save_graph is 'y'.
    - project (str, optional): Name of the project for labeling the plot. Default is None.

    Returns:
    None
    """

    # Get the column names
    x_column = pca_df.filter(like=f'PCA{pc1}_', axis=1)
    x_axes = x_column.columns.tolist()[0]
    y_column = pca_df.filter(like=f'PCA{pc2}_', axis=1)
    y_axes = y_column.columns.tolist()[0]

    X = pd.DataFrame(pca_df, columns=[x_axes, y_axes])
    X['Labels'] = pd.DataFrame(pca_df, columns=['Labels'])

    # Create a 2D scatter plot with Plotly
    fig = px.scatter(X,
                x=x_axes,
                y=y_axes, 
                color='Labels', 
                size_max=10,
                title=f'{project} -- PCs {pc1} & {pc2}')

    if save_graph == 'y':
        html_file_path = os.path.join(save_path, f'{project}_PCA{pc1}-{pc2}.html')
        fig.write_html(html_file_path)


def all_2d_plotlys_in_ranges(pca_df, ranges, save_path=None, project = None):
    """
    Generate 2D scatter plots for all combinations of principal components within specified ranges input as list of lists.

    Parameters:
    - pca_df (pd.DataFrame): DataFrame containing PCA-transformed data generated from pca_df function.
    - ranges (list): List of lists specifying the ranges for principal components combinations.
    - save_path (str, optional): Path to the directory where the saved HTML files will be stored. Required if save_graph is 'y'.
    - project (str, optional): Name of the project for labeling the plots. Default is None.

    Returns:
    None
    """
    for pc_range in ranges:
        combos = set()
        pc_start = pc_range[0]
        pc_end = pc_range[1]

        for i in range(pc_start, pc_end):
            for j in range(pc_start, pc_end):
                if i != j:
                    current_combo = frozenset({i, j})
                    if current_combo not in combos:
                        combos.add(current_combo)

                        plotly_2d(pca_df, i, j, save_graph='y', save_path=save_path, project=project)
    return 

def specific_2D_plotlys(pca_df, individual_pcas, save_path=None, project = None):
    """
    Generate 2D scatter plots for specific combinations of principal components.

    Parameters:
    - pca_df (pd.DataFrame): DataFrame containing PCA-transformed data generated from pca_df function
    - individual_pcas (list): List of lists specifying individual principal components combinations.
    - save_path (str, optional): Path to the directory where the saved HTML files will be stored. Required if save_graph is 'y'.
    - file_name (str, optional): Base name for the saved HTML files. Required if save_graph is 'y'.
    - project (str, optional): Name of the project for labeling the plots. Default is None.

    Returns:
    None
    """

    for combo in individual_pcas:
        pc1 = combo[0]
        pc2 = combo [1]
        file_name = f'{project}_PCA{pc1}&{pc2}_plotly.html'

        plotly_2d(pca_df, pc1, pc2, save_graph='y', save_path=save_path, project=project)

    return individual_pcas

def selected_2d_plotlys(pca_df, ranges, individual_pcas, save_path = None, project = None):
    """
    Generate selected 2D scatter plots based on specified ranges and individual combinations of principal components.

    Parameters:
    - pca_df (pd.DataFrame): DataFrame containing PCA-transformed data generated from pca_df function
    - ranges (list): List of lists specifying the ranges for principal components combinations.
    - individual_pcas (list): List of lists specifying individual principal components combinations.
    - save_path (str, optional): Path to the directory where the saved HTML files will be stored. Required if save_graph is 'y'.
    - file_name (str, optional): Base name for the saved HTML files. Required if save_graph is 'y'.
    - project (str, optional): Name of the project for labeling the plots. Default is None.

    Returns:
    None
    """

    all_2d_plotlys_in_ranges(pca_df, ranges, save_path=save_path, project= project)

    specific_2D_plotlys(pca_df, individual_pcas, save_path=save_path, project = project)





    
def graph_pca(pt, labels, sequ, pc1, pc2, pca, show_pts='all', show_graph='y', save_graph=None, save_path=None, project_name=None):
    '''
    Graphs PCA using IG, label and sequence numpy arrays based on selected principle components and the resulting 
    PCA transformation from the pca_transform function.

    Also returns list of rows containing binders/non-binders in separate lists for further use
    '''
    pca_transf = pca_transformation(pt, pca, pc1, pc2)

    x_axes = f'PCA{pc1} explained variance: {pca.explained_variance_ratio_[pc1-1]*100:.2f}%'
    y_axes = f'PCA{pc2} explained variance: {pca.explained_variance_ratio_[pc2-1]*100:.2f}%'

    X = pd.DataFrame(pca_transf, columns=[x_axes, y_axes])
    X['Labels'] = pd.DataFrame(labels, columns=['Labels'])['Labels'].apply(lambda x: 'Binder' if x > 0.5 else 'Non Binder')

    binder_color = 'red'
    non_binder_color = 'blue'

    title_ext = ''

    if show_pts == 'binders':
        X = X[X['Labels'] == 'Binder']
        title_ext = ' - BINDERS ONLY'
    elif show_pts == 'non binders':
        X = X[X['Labels'] == 'Non Binder']
        title_ext = ' - NON BINDERS ONLY'

    plt.figure(figsize=(20, 15))

    ax = sns.scatterplot(x=x_axes, y=y_axes, data=X, linewidth=0, hue='Labels', s=5, palette={'Binder': binder_color, 'Non Binder': non_binder_color})
    ax.set_frame_on(False)
    ax.locator_params(nbins=50, axis='x')
    ax.locator_params(nbins=50, axis='y')

    plt.title(f'PCA Reduction PC{pc1} & {pc2}{title_ext}')


    binders = X.index[X['Labels'] == 'Binder'].tolist()
    nonbinders = X.index[X['Labels'] == 'Non Binder'].tolist()

    if save_graph == 'y':
        pca_folder = os.path.join(save_path, 'pca_analysis')
        if not os.path.exists(pca_folder):
            os.makedirs(pca_folder)

        pca_subfolder = os.path.join(pca_folder, f'{show_pts}_data_points')
        if not os.path.exists(pca_subfolder):
            os.makedirs(pca_subfolder)

        plt.savefig(os.path.join(pca_subfolder, f'{project_name}_PCA_{pc1}-{pc2}_{show_pts}.png'))

    if show_graph == 'y':
        plt.show()
    plt.close()

    return binders, nonbinders, X



def compute_and_graph_pca(pt, labels, sequ, pc1, pc2, show_pts=None, show_graph='y', save_graph=None, save_path=None, project_name=None):
    '''
    Performs all computation steps of the PCA analysis according to the set principle components using the
    IG, labels and sequence data
    '''
    pca = compute_pca(pc2)
    pca_transformat = pca_transformation(pt, pca, pc1, pc2)

    binders_and_nonbinders = graph_pca(pt, labels, sequ, pc1, pc2, pca, show_pts, show_graph, save_graph, save_path, project_name)

    return binders_and_nonbinders




def show_all_pc_combos(pt, labels, sequ, pc_start, pc_end, show_pts=None, show_graph='n', save_graph=None, save_path=None, project_name=None):
    '''
    Computes and graphs all possible principle component combinations for the set pc number based on IG, label, 
    sequence.
    
    Only for quick investigation purposes. Does not return anything
    '''
    combos = []
    pca = compute_pca(pc_end)
    
    for i in range(pc_start, pc_end+1):
        for j in range (pc_start, pc_end+1):
            if i !=j and (i,j) not in combos and (j,i) not in combos:
                combos.append((i,j))
                compute_and_graph_pca(pt, labels, sequ, i, j, show_pts, show_graph, save_graph, save_path, project_name)

    

def aa_pair_plot(pt, labels, pc_no, show_pts='all', show_graph='y', save_graph=None, save_path=None, project_name=None):

    ### flatten IG data 
    pca = compute_pca(pc_no)
    reshaped_data = pt.reshape(-1, 20*10)
    # print(reshaped_data[1])

    ### Create column names (AA + Position in CDRH3)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    num_columns = 10
    column_names = [f'{aa}{i}' for i in range(1, num_columns + 1) for aa in amino_acids]
    # column_names.append('Labels')

    pplt_df = pd.DataFrame(reshaped_data, columns=column_names)
    pplt_df['Labels'] = pd.DataFrame(labels, columns=['Labels'])['Labels'].apply(lambda x: 'Binder' if x > 0.5 else 'Non Binder')

    # print(pplt_df.head())



    sns.pairplot(pplt_df)

    if show_graph == 'y':
        plt.show()

    plt.close()


# In[]:

def pca_pair_plot(pt, labels, pc_no, show_pts='all', show_graph='y', save_graph=None, save_path=None, project_name=None, iterate_all_pt_types='n'):
    """
    Generates a pair plot using principal component analysis (PCA) transformed data.

    Parameters:
    - pt (numpy.ndarray): Input data array to be transformed using PCA.
    - labels (numpy.ndarray): Array of labels corresponding to each data point.
    - pc_no (int): Number of principal components to use in PCA.
    - show_pts (str, optional): Specify which points to show in the pair plot ('all', 'binders', 'non binders'). Default is 'all'.
    - show_graph (str, optional): Specify whether to display the pair plot ('y' for yes, 'n' for no). Default is 'y'.
    - save_graph (str, optional): Specify whether to save the pair plot as an image ('y' for yes, 'n' for no). Default is None.
    - save_path (str, optional): Path to the directory where the saved images will be stored. Required if save_graph is 'y'.
    - project_name (str, optional): Name of the project for labeling the saved image. Required if save_graph is 'y'.

    Returns:
    None
    """
    pca = compute_pca(pc_no)
    pca_transf = pca.fit_transform(pt.reshape(-1, 20*10))

    column_names = [f'PCA{i} ExpVar: {pca.explained_variance_ratio_[i-1]*100:.2f}%' for i in range(1, pc_no + 1)]

    pplt_df = pd.DataFrame(pca_transf, columns=column_names)
    pplt_df['Labels'] = pd.DataFrame(labels, columns=['Labels'])['Labels'].apply(lambda x: 'Binder' if x > 0.5 else 'Non Binder')

    returned_df = pplt_df

    custom_palette = {'Binder': 'red', 'Non Binder': 'blue'}
    title_ext = ''
    if show_pts == 'binders':
        pplt_df = pplt_df[pplt_df['Labels'] == 'Binder']
        title_ext = ' - BINDERS ONLY'
    elif show_pts == 'non binders':
        pplt_df = pplt_df[pplt_df['Labels'] == 'Non Binder']
        title_ext = ' - NON BINDERS ONLY'

    all_dataframes = [pplt_df]
    title_extensions = ['_ALL POINTS']


    if iterate_all_pt_types == 'y':
        all_dataframes.append(pplt_df[pplt_df['Labels'] == 'Binder'])
        all_dataframes.append(pplt_df[pplt_df['Labels'] == 'Non Binder'])

        title_extensions.append('_BINDERS ONLY')
        title_extensions.append('_NON BINDERS ONLY')

    for i in range(0, len(all_dataframes)):
        plt.figure(figsize=(80, 60))
        df = all_dataframes[i]
        title_ext = title_extensions[i]

        sns.pairplot(df, hue='Labels', palette=custom_palette, plot_kws = {'s':7})

        plt.title(f'PCA Pair-Plot - PCs 1 through {pc_no}_{title_ext}')

        if save_graph == 'y':
            print('saving')
            # pca_folder = os.path.join(save_path, 'pca_analysis')
            # pca_subfolder = os.path.join(pca_folder, title_ext)
            # if not os.path.exists(pca_folder):
            #     os.makedirs(pca_folder)
            # if not os.path.exists(pca_subfolder):
            #     os.makedirs(pca_subfolder)

            plt.savefig(os.path.join(save_path, f'{project_name}_pair_plot_PCA_1-{pc_no}{title_ext}.png'))


        if show_graph == 'y':
            plt.show()
        plt.close()

    return returned_df


# In[5]:


# pca = compute_pca(10)

# pca_transf = pca_transformation(pt, pca, 1, 2)

# bvnb = graph_pca(pt, labels, sequ, 1, 3, pca)

# compute_and_graph_pca(pt, labels, sequ, 2, 3)

# show_all_pc_combos(pt, labels, sequ, 4, show_pts = 'binders', show_graph='n', save_graph='y', save_path='/Users/isaacdaviet/Desktop/thesis', project_name='test')

# aa_pair_plot(pt, labels, sequ, 10, save_graph=None, save_path=None, project_name=None)

# pca_pair_plot(pt, labels, 4, show_graph='y', save_graph='y', save_path='/Users/isaacdaviet/Desktop/thesis', project_name='pca_test', iterate_all_pt_types = 'y')


# ## PCA Explained Variance Bar Plot

# In[6]:


def pca_explained_variance_bar_plot(pt, n_components = 20, show_graph = 'y', save_graph = None, save_path = None, project_name = None):
    pca = compute_pca(n_components)
    pca_transf = pca_transformation(pt, pca, 1, n_components)

    total_ev = 0
    exp_var = []
    for ev in pca.explained_variance_ratio_:
        expv= round(ev * 100, 2)
        exp_var.append(expv)
        total_ev += expv


    # Bar plot for explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_components + 1), exp_var, color='blue')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance (%)')
    plt.title(f'Total PC Explained Variance: {total_ev}')
    if save_graph == 'y':
        selection_folder = os.path.join(save_path, 'pca_analysis')
        if not os.path.exists(selection_folder):
            os.makedirs(selection_folder)

        plt.savefig(os.path.join(selection_folder, f'{project_name}_{n_components}PCs_explained_variance.png'))

    if show_graph == 'y':
        plt.show()
    plt.close()

# pca_explained_variance_bar_plot(pt, show_graph = 'n', save_graph = 'y', save_path = '/Users/isaacdaviet/Desktop/thesis', project_name = 'test')


# ## 3-D PCA graphing

# In[ ]:


def graphing_3d_pca(pt, labels, sequ, pc1, pc2, pc3, show_pts = None):
        
    pca = compute_pca(pc3)
    pca_transf = pca.fit_transform(pt.reshape(-1, 20*10))
    pca_transf = pca_transf[:, [pc1-1, pc2-1, pc3-1]]

    x_axes = f'PCA{pc1} expl var: {pca.explained_variance_ratio_[pc1-1]*100:.2f}%'
    y_axes = f'PCA{pc2} expl var: {pca.explained_variance_ratio_[pc2-1]*100:.2f}%'
    z_axes = f'PCA{pc3} expl var: {pca.explained_variance_ratio_[pc3-1]*100:.2f}%'

    X = pd.DataFrame(pca_transf, columns = [x_axes, y_axes, z_axes])
    X['Labels'] = pd.DataFrame(labels, columns=['Labels'])['Labels'].apply(lambda x: 'Binder' if x > 0.5 else 'Non Binder')

    if show_pts == 'binders':
        X = X[X['Labels'] == 'Binder']

    elif show_pts == 'non binders':
        X = X[X['Labels'] == 'Non Binder']

    # Create a 3D scatter plot with Plotly
    fig = go.Figure(data=[
        go.Scatter3d(
            x=X[x_axes],
            y=X[y_axes],
            z=X[z_axes],
            mode='markers',
            marker=dict(
                size=2,
                color=X['Labels'].map({'Binder': 'red', 'Non Binder': 'blue'}),
                opacity=0.5
            )
        )
    ])

    # Set labels for each axis
    fig.update_layout(scene=dict(xaxis_title=x_axes, yaxis_title=y_axes, zaxis_title=z_axes))
    fig.update_layout(width=800, height=400)

    # Show the interactive plot
    fig.show()

# graphing_3d_pca(pt, labels, sequ, 1, 2, 3, show_pts = 'binders')
    

def show_all_pc_combos_3d(pt, labels, sequ, pc_no, show_pts = None):
    combos = set()
    pca = compute_pca(pc_no)
    
    for i in range(1, pc_no+1):
        for j in range (1, pc_no+1):
            for k in range (1, pc_no+1):
                if i !=j and i != k and j != k: 
                    current_combo = frozenset({i, j, k}) 

                    if current_combo not in combos:
                        combos.add(current_combo)
                        graphing_3d_pca(pt, labels, sequ, i, j, k, 'binders')

# show_all_pc_combos_3d(pt, labels, sequ, 10, show_pts = None)


# ## Extracting all data points from a given set of vertices

# ### Example 2D vertices provided by Robert

# In[8]:


# ### Vertices provided by Robert as examples

# #-0.0370, 0.0512#bottom right
# #-0.1263, 0.1838#bottom left
# #-0.0051, 0.1493#top right
# #-0.0779, 0.2759#top left

# # Mason negative extract
# vertices_1 = np.asarray([(-0.1263, 0.1838),#bottom left
#                        (-0.0370, 0.0512),#bottom right
#                        (-0.0051, 0.1493),#top right
#                        (-0.0779, 0.2759)])#top left

# # Mason spike right down extract
# vertices_2 = np.asarray([(-0.0735, -0.096),#bottom left
#                        (0.1508, -0.425),#bottom right
#                        (0.2074, -0.312),#top right
#                        (0.0211, -0.04)])#top left


# # Mason spike right down up
# vertices_3 = np.asarray([(0.0484, -0.008),#bottom left
#                        (0.3635, 0.15),#bottom right
#                        (0.3703, 0.295),#top right
#                        (0.0513, 0.121)])#top left

# # Brij spike down
# vertices_4 = np.asarray([(0.0578, -0.045),#bottom left
#                        (0.5968, 0.029),#bottom right
#                        (0.5942, 0.098),#top right
#                        (0.0628, 0.009)])#top left


# # Brij spike vertical 1
# vertices_5 = np.asarray([(-0.0263, 0.089),#bottom left
#                        (-0.0056, 0.093),#bottom right
#                        (-0.0784, 0.604),#top right
#                        (-0.0997, 0.588)])#top left

# # Brij spike diagonal
# vertices_6 = np.asarray([(0.1356, 0.127),#bottom left
#                        (0.2887, 0.264),#bottom right
#                        (0.2648, 0.298),#top right
#                        (0.1256, 0.147)])#top left

# vertices_selection = [vertices_1, vertices_2, vertices_3, vertices_4,vertices_5,vertices_6]


# ### Function to extract PCA rows/positions of the data points from 2D graph
# Maybe add a show_pts argument to visualize only the binders or non-binders? Although that can be done at a later date

# In[ ]:


### ploting selection captured in vertices´
def pca_selection(pt, labels, vertices, pc1, pc2, show_graph = None, save_graph = None, save_path = None, selection_name = None):
    """
    Returns the positions of the IG sequences entries that fall within the selected vertices
    Resulting list can be of all CDRH3 located within the vertices or just the binders/non-binders
    Also returns the final complete array for QC purposes
    
    Final 'show_graph' input can be left blank, or input 'y' or 'yes' to show the resulting graph and selected points
    
    Returned selection to be used in next functions to extract and convert the sequences to one letter amino acid code
    """
    pca = compute_pca(pc2)
    pca_transf = pca_transformation(pt, pca, pc1, pc2)

    hull = ConvexHull(vertices)
    vertices = vertices[hull.vertices]
    
    path = Path(vertices)
    mask = path.contains_points(pca_transf)
    
    
    x_axes = f'PCA{pc1} explained variance: {pca.explained_variance_ratio_[pc1-1]*100:.2f}%'
    y_axes = f'PCA{pc2} explained variance: {pca.explained_variance_ratio_[pc2-1]*100:.2f}%'
    
    X = pd.DataFrame(pca_transf, columns = [x_axes, y_axes])
    X['Labels'] = pd.DataFrame(labels, columns=['Labels'])['Labels'].apply(lambda x: 'Binder' if x > 0.5 else 'Non Binder')

    X['mask'] = mask
    
    if show_graph == 'y' or save_graph == 'y':
        plt.figure(figsize = (20,15))
        ax = sns.scatterplot(x=x_axes, y=y_axes, data=X, linewidth=0, hue='mask', s=5)
        ax.set_frame_on(False)
        ax.locator_params(nbins=50, axis='x')
        ax.locator_params(nbins=50, axis='y')


        if save_graph == 'y':
            selection_folder = os.path.join(save_path, 'PCA_selections')
            if not os.path.exists(selection_folder):
                os.makedirs(selection_folder)

            plt.savefig(os.path.join(selection_folder, f'{selection_name}_PCA{pc1}_{pc2}.png'))

        if show_graph == 'y':
            plt.show()
    plt.close()
        
    all_rows = [int(i) for i in X.index[mask]]
    binder_rows = [int(i) for i in X.index[mask] if X.loc[i, 'Labels'] == 'Binder']
    nonbinder_rows = [int(i) for i in X.index[mask] if X.loc[i, 'Labels'] == 'Non Binder']
    
    print(f'Number of Points Selected: {len(all_rows)}    --    Number of Binders: {len(binder_rows)}    --    Number of Non Binders: {len(nonbinder_rows)}')



    return all_rows, binder_rows, nonbinder_rows, X 
  
# all_rows, selected_binders, selected_nonbinders, xarray = pca_selection(pt, labels, vertices_1, 1, 2, show_ show_graph = 'n', save_graph = 'y', save_path = '/Users/isaacdaviet/Desktop/thesis', project_name = 'test')


# ### Extracting data points from a 3D PCA graph
# Completed but not tested. Difficult to figure out the 3D vertices. Might be more useful as an exploratory tool rather than a data extraction tool

# In[ ]:


def extract_manual_clusters_vertices(manual_clusters_df):
    vertices_dict = {}
    # print(manual_clusters_df)
    for index, row in manual_clusters_df.iterrows():
        # print(row)
        bottom_left, bottom_right, top_right, top_left = ast.literal_eval(row['bottom left']), ast.literal_eval(row['bottom right']), ast.literal_eval(row['top right']), ast.literal_eval(row['top left'])

        vertices = np.asarray([bottom_left, bottom_right, top_right, top_left])

        cluster_name = f"{row['PCA']}_{row['Cluster']}"
        vertices_dict[cluster_name] = vertices

    return vertices_dict
    
# vertices_dict= extract_manual_clusters_vertices(manual_clusters_df)
# print(vertices_dict)

def extract_manual_clusters(vertices_dict, pca_df, check_clusters = 'n'):
    clusters_by_iseq = {}
    clusters_by_sequence = {}

    for cluster_name, vertices in vertices_dict.items():
        clusters_by_iseq[cluster_name] = []
        clusters_by_sequence[cluster_name] = []
        hull = ConvexHull(vertices)
        vertices = vertices[hull.vertices]
        # print(vertices)
        path = Path(vertices)

        pcs = cluster_name.split('_')[1]
        pt_label = cluster_name.split('_')[-1]
        pt_type = 'Non Binder' if 'NB' in pt_label else 'Binder'
        
        pcs = pcs.replace('PCA', '')
        pcs = pcs.split('-')
        pc1, pc2 = pcs[0], pcs[1]

        search_x = f'PCA{pc1}_'
        search_y = f'PCA{pc2}_'


        for column_name, column_data in pca_df.items():
            if search_x in column_name:
                x_axes = column_name
            if search_y in column_name:
                y_axes = column_name

        X = pca_df[[x_axes, y_axes, 'label', 'iseq', 'Sequences']]

        # print(X[[x_axes, y_axes]].head())

        X['mask'] = False

        for index, row in X.iterrows():
            if row['label'] == pt_type:
                point = row[[x_axes, y_axes]].values
                X.at[index, 'mask'] = path.contains_point(point)

        for index, row in X[X['mask']].iterrows():
            clusters_by_iseq[cluster_name].append(row['iseq'])
            clusters_by_sequence[cluster_name].append(row['Sequences'])

        if check_clusters == 'y':
            Y =  X[X['label'] == 'Non Binder']

            plt.scatter(Y[x_axes], Y[y_axes], c=Y['mask'], cmap='coolwarm', s = 5)
            plt.xlabel(x_axes)
            plt.ylabel(y_axes)
            plt.title(f'Scatter plot of {cluster_name}')
            plt.colorbar(label='Mask')
            plt.show()

        
            
    return clusters_by_iseq, clusters_by_sequence

# clusters_by_iseq = extract_manual_clusters(vertices_dict, pca_df)

def clusters_by_filename_csv(clusters_by_iseq, filepath, binders_igfold_filename_format, nonbinders_igfold_filename_format, replace = 'ISEQ'):
    clusters_by_filename= {}

    for cluster_name, iseqs in clusters_by_iseq.items():
        for iseq in iseqs:
            
            file = nonbinders_igfold_filename_format.replace(replace, str(iseq))if 'NB' in cluster_name else binders_igfold_filename_format.replace(replace, str(iseq))

            if cluster_name in clusters_by_filename:
                clusters_by_filename[cluster_name].append(file)
            else:
                clusters_by_filename[cluster_name] = [file]
                

    max_length = max(len(lst) for lst in clusters_by_filename.values())
    clusters_by_filename_padded = {key: lst + [None] * (max_length - len(lst)) for key, lst in clusters_by_filename.items()}

    # Create DataFrame
    clusters_by_filename_df = pd.DataFrame(clusters_by_filename_padded)

    # Save to CSV
    clusters_by_filename_df.to_csv(f'{filepath}/mason_PCA_manual_clusters_igfoldfiles.csv', index=False)

    return clusters_by_filename_df

def extract_manual_pca_clusters_for_space2(manual_clusters_df, pca_df, filepath, binders_igfold_filename_format, nonbinders_igfold_filename_format, replace = 'ISEQ', check_clusters = 'n'):

    vertices_dict= extract_manual_clusters_vertices(manual_clusters_df)
    clusters_by_iseq, clusters_by_sequence = extract_manual_clusters(vertices_dict, pca_df, check_clusters = check_clusters)
    clusters_by_filename_df = clusters_by_filename_csv(clusters_by_iseq, filepath, binders_igfold_filename_format, nonbinders_igfold_filename_format, replace = replace)

    return clusters_by_filename_df, clusters_by_sequence



# In[]:
vertices_cube = np.array([
    (0, 0, 0),   # Vertex 1: bottom front left
    (0, 0, 0),   # Vertex 2: bottom front right
    (0, 0, 0),   # Vertex 3: bottom back right
    (0, 0, 0),   # Vertex 4: bottom back left
    (0, 0, 0),   # Vertex 5: top front left
    (0, 0, 0),   # Vertex 6: top front right
    (0, 0, 0),   # Vertex 7: top back right
    (0, 0, 0)])    # Vertex 8: top back left

def extract_3d_pca(pt, labels, sequ, pc1, pc2, pc3, vertices, show_pts=None):
        
    # Compute PCA and transform the data
    pca = compute_pca(pc3)
    pca_transf = pca.fit_transform(pt.reshape(-1, 20*10))
    pca_transf = pca_transf[:, [pc1-1, pc2-1, pc3-1]]

    # Create axis labels
    x_axes = f'PCA{pc1} expl var: {pca.explained_variance_ratio_[pc1-1]*100:.2f}%'
    y_axes = f'PCA{pc2} expl var: {pca.explained_variance_ratio_[pc2-1]*100:.2f}%'
    z_axes = f'PCA{pc3} expl var: {pca.explained_variance_ratio_[pc3-1]*100:.2f}%'

    # Create a Convex Hull using provided vertices
    hull = ConvexHull(vertices)
    vertices = vertices[hull.vertices]

    # Create a path to check points within the Convex Hull
    path = Path(vertices)
    mask = path.contains_points(pca_transf)

    # Create a DataFrame with PCA-transformed data, labels, and mask
    X = pd.DataFrame(pca_transf, columns=[x_axes, y_axes, z_axes])
    X['Labels'] = pd.DataFrame(labels, columns=['Labels'])['Labels'].apply(lambda x: 'Binder' if x > 0.5 else 'Non Binder')
    X['mask'] = mask

    # Filter points based on the 'show_pts' parameter
    if show_pts == 'binders':
        X = X[X['Labels'] == 'Binder']
    elif show_pts == 'non binders':
        X = X[X['Labels'] == 'Non Binder']

    # Create a 3D scatter plot with Plotly
    fig = go.Figure(data=[
        go.Scatter3d(
            x=X[x_axes],
            y=X[y_axes],
            z=X[z_axes],
            mode='markers',
            marker=dict(
                size=2,
                color=X['mask'].map({True: 'red', False: 'blue'}),
                opacity=0.5
            )
        )
    ])

    # Set labels for each axis
    fig.update_layout(scene=dict(xaxis_title=x_axes, yaxis_title=y_axes, zaxis_title=z_axes))
    fig.update_layout(width=1200, height=800)

    # Show the interactive plot
    fig.show()

# Example usage:
# extract_3d_pca(pt, labels, sequ, 1, 2, 3, vertices, show_pts=None)
# extract_3d_pca(pt, labels, sequ, 1, 2, 3, vertices, show_pts='binders')
# extract_3d_pca(pt, labels, sequ, 1, 2, 3, vertices, show_pts='non binders')