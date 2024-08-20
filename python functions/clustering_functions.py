#!/usr/bin/env python
# coding: utf-8
# In[1]:
# # Imports

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import plotly.express as px

# In[0]:
# # Functions

# In[2]:
# # Clustering Calculation and Graphing Functions

def agglomerative_clustering(select_umap_df, n_clusters, linkage, affinity):
    # Initialize AgglomerativeClustering model
    agglomerative_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity)

    # Perform clustering
    select_umap_df['Cluster'] = agglomerative_model.fit_predict(select_umap_df.iloc[:, :-2])

    select_umap_df['Cluster'] = select_umap_df['Labels'].astype(str) + select_umap_df['Cluster'].astype(str)

    return select_umap_df

def dbscan_clustering(select_umap_df, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    select_umap_df['Cluster'] = dbscan.fit_predict(select_umap_df.iloc[:, :-3])
    select_umap_df['Cluster'] = select_umap_df['Labels'].astype(str) + select_umap_df['Cluster'].astype(str)
    return select_umap_df

def add_iseq_to_labels_file(labeled_seq_csv_file, unique_positive_fv, unique_negative_fv):

    labeled_df = pd.read_csv(labeled_seq_csv_file)
    pos_fv_df = pd.read_csv(unique_positive_fv)
    neg_fv_df = pd.read_csv(unique_negative_fv)

    iseqs = []

    for index, row in labeled_df.iterrows():
        label = row['label']
        sequence = row['sequence']

        if label == 'Binder':
            fv_df = pos_fv_df
        if label == 'Non Binder':
            fv_df = neg_fv_df

        match_row = fv_df[fv_df['hseq'].str.contains(sequence)]
        if not match_row.empty:
            iseq = match_row['iseq'].values[0]
            iseqs.append(iseq)
        else:
            iseqs.append('Not found')

    labeled_df['iseq'] = iseqs

    labeled_df.to_csv(labeled_seq_csv_file, index=False)

    return labeled_df

def add_seqs_umap_df(reduction_df_csv, labeled_seqs):
    ### in the future, look to integrate this option when the umap reductions are generated.
    df = pd.read_csv(reduction_df_csv)

    sequ = pd.read_csv(labeled_seqs)
    df['Sequences'] = sequ['sequence']
    df['iseq'] = sequ['iseq']

    return df

# test = r'/Users/isaacdaviet/Desktop/graphs/Mason_umap_final_graphs/csv_files/Mason-FinalSelect_UMAP-correlation_nC3_Ne25_MD0.0.csv'

# test = add_seqs_umap_df(test, labeled_seqs)
# print(test.head())

def cluster_extract_and_plot(selected_graphs_csv, labeled_seqs, umap_reductions_folder, clusters = 'both', algorithm = 'dbscan', generate_indicator = 'y', plt_size = [60, 20], pt_size = 2, fontsize=64, show_unclustered_pts = 'y', save_graph = 'y', save_folder = None, project_name=None, save_clusters_csv = 'y', save_plotly = 'y'):

    ### extract parameters from results csv file
    selection = pd.read_csv(selected_graphs_csv)
    # print(selection.head())

    for index, row in selection.iterrows():
        ### iterate each row in results csv file and extract parameter data
        id, umap_file, metric, n_components, n_neighbors, min_dist, comp1, comp2, generate, db_eps,	db_min_s = row['graph id'], row['umap file'], row['distance metric'], row['n_components'], row['n_neighbors'], row['min_dist'], row['component_1'], row['component_2'], row['generate'], row['dbscan_eps'], row['dbscan_min_samples']


        if generate_indicator == generate:
            print(f'Clustering {id}')
            db_min_s = int(db_min_s)

            ### locate and extract relevant umap reduction csv file
            umap_csv = os.path.join(umap_reductions_folder, umap_file)

            ### add 1 letter sequences to umap data frame and remove columns of components not to be graphed
            umap_df = add_seqs_umap_df(umap_csv, labeled_seqs)
            
            umap_df = umap_df[[f'UMAP-{round(comp1)}', f'UMAP-{round(comp2)}', 'Sequences', 'iseq', 'Labels']]

            # print(umap_df.head()) # sanity check

            ### create separate frames for binders/non binders
            all_umap_dfs = []
            if clusters == 'binders' or clusters == 'both':
                binders_df = umap_df[umap_df['Labels'] == 'Binder']
                all_umap_dfs.append(binders_df)
                print(f'\tgenerated binders data frame')
            if clusters == 'non binders' or clusters == 'both':
                non_binders_df = umap_df[umap_df['Labels'] == 'Non Binder']
                all_umap_dfs.append(non_binders_df)
                print(f'\tgenerated non binders data frame')

            # print(all_umap_dfs) # Sanity check
            fig, axs = plt.subplots(nrows = 1, ncols = len(all_umap_dfs)+1, figsize = (plt_size[0], plt_size[1]))

            sns.scatterplot(x=f'UMAP-{round(comp1)}',
                        y=f'UMAP-{round(comp2)}',
                        hue='Labels',
                        palette={'Binder': 'red', 'Non Binder': 'blue'}, 
                        data=umap_df, 
                        s=pt_size, 
                        ax=axs[0])
            axs[0].set_title('Binders Vs Non binders')

            ### Perform clustering on specific components
            # Perform clustering on specific components
            for i, select_umap_df in enumerate(all_umap_dfs):
                df_type = select_umap_df['Labels'].iloc[0]

                if algorithm == 'dbscan':
                    select_umap_df = dbscan_clustering(select_umap_df, db_eps, db_min_s)
                    param_ext = f'{db_eps}-{db_min_s}'

                if save_plotly == 'y':
                    fig = px.scatter(select_umap_df,
                                  x=f'UMAP-{round(comp1)}', y=f'UMAP-{round(comp2)}', color='Cluster', 
                                  size_max=10,
                                  title=f'{project_name} -- {metric} UMAP {algorithm} Clusters -- nComp = {n_components} -- nNeighbors = {n_neighbors} -- Min_Dist = {min_dist} -- Cmps {comp1} & {comp2} -- {df_type}')

                    fig.write_html(os.path.join(save_folder, f'UMAP_{id}_{algorithm}Clusters-{param_ext}_{df_type}.html'))

                # print(select_umap_df.head())

                # Plot the clusters
                if show_unclustered_pts == 'n':
                    filtered_df = select_umap_df[select_umap_df['Cluster'] != 'Binder-1']

                else: 
                    filtered_df = select_umap_df

                sns.scatterplot(x=f'UMAP-{round(comp1)}',
                                y=f'UMAP-{round(comp2)}',
                                hue='Cluster',
                                data=filtered_df,
                                palette = 'colorblind',
                                s=pt_size,
                                ax=axs[i + 1])  # Use the appropriate axis from the subplot grid
                axs[i + 1].set_title(f'{df_type} clusters')

            # Set the main title
            title = f'{project_name} -- {metric} UMAP {algorithm} Clusters -- nComp = {n_components} -- nNeighbors = {n_neighbors} -- Min_Dist = {min_dist} -- Cmps {comp1} & {comp2}'
            plt.suptitle(title, fontsize=fontsize)
            plt.tight_layout()

            print(f'Finished caclulating {id} clusters\n\n')

            if save_graph == 'y':   
                # graph_folder = os.path.join(save_folder, 'graphs')
                # if not os.path.exists(graph_folder):
                #     os.makedirs(graph_folder)

                plt.savefig(os.path.join(save_folder, f'UMAP_{id}_{algorithm}Clusters-{param_ext}.png'))

            if save_clusters_csv == 'y':   
                ### create txt folder to save files
                # txt_folder = os.path.join(save_folder, 'clusters_csv')
                # if not os.path.exists(txt_folder):
                #     os.makedirs(txt_folder)

                ### Rename clusters to 'Label + Cluster #' and combine binder and non binder cluster df into single df
                final_cluster_df = pd.DataFrame()
                # print(all_umap_dfs[1])
                for clustered_df in all_umap_dfs:
                    # print(clustered_df)
                    # print(clusters_df.head()) # Sanity check
                    final_cluster_df = pd.concat([final_cluster_df, clustered_df], ignore_index=True)
                
                # print(final_cluster_df)


                file = os.path.join(save_folder, f'UMAP_{id}_{algorithm}Clusters-{param_ext}.csv')
                final_cluster_df.to_csv(file, index = False)
# In[3]:

# # QC Functions:
                
def cluster_priority(clusters_df, high_threshold = 10, low_threshold = 1, upper_limit = None, print_percentages = 'n'):
    binders_df = clusters_df[clusters_df['Labels'] == 'Binder']
    non_binders_df = clusters_df[clusters_df['Labels'] == 'Non Binder']

    n_binders = len(binders_df)
    n_nonbinders = len(non_binders_df)

    b_cluster_counts = binders_df['adjusted_clusters'].value_counts()   
    nb_cluster_counts = non_binders_df['adjusted_clusters'].value_counts()

    b_cluster_percentages = round(((b_cluster_counts / n_binders) * 100), 2)
    nb_cluster_percentages = round(((nb_cluster_counts / n_nonbinders) * 100), 2)

    combined_percentages = b_cluster_percentages.add(nb_cluster_percentages, fill_value=0)

    high_clusters = combined_percentages[combined_percentages > high_threshold].index.tolist()
    low_clusters = combined_percentages[combined_percentages < low_threshold].index.tolist()
    med_clusters = combined_percentages[
        (combined_percentages >= low_threshold) & (combined_percentages <= high_threshold)
    ].index.tolist()

    out_of_bounds = combined_percentages[combined_percentages > upper_limit].index.tolist()

    print(combined_percentages) if print_percentages == 'both' else None
    print(b_cluster_percentages) if print_percentages == 'binders'else None
    print(nb_cluster_percentages) if print_percentages == 'non binders' else None
    
    return high_clusters, med_clusters, low_clusters, out_of_bounds

def extract_clusters_to_combine(clusters_csv, analysis_csv, point_type = 'Binder'):

    ### load cluster and analysis csv files
    analysis_df =pd.read_csv(analysis_csv)

    ### extract graph id from cluster csv file name
    cluster_id = clusters_csv.split('/')
    cluster_id = cluster_id[-1]
    cluster_id = cluster_id.split('_')
    cluster_id = cluster_id[1]

    combine_clusters = None
    
    ### find the row associated with the cluster csv
    for index, row in analysis_df.iterrows():
        id = row['graph id']

        if id == cluster_id:
            groups = []

            combine_clusters_row = str(row['combine_clusters'])
            ### screen for files that do need to have clusters adjusted
            if combine_clusters_row != None:

                combine_clusters = combine_clusters_row.split('/') # split different clusters if needed

                # generate list of lists containing 
                for group in combine_clusters:
                    new_group_name = f'Binder{group}'
                    group = group.split('+')

                    for cluster in group:
                        groups.append([f'{point_type}{cluster}', new_group_name])


    return groups

def add_adjusted_clusters_column(row, groups):
    for entry in groups:
        if row['Cluster'] in entry:
            return entry[1]
    return row['Cluster']

def add_priority_column(row, high, med, low, out_of_bounds):
    if row['adjusted_clusters'] == 'Binder-1':
        return 'unclustered'
    elif row['adjusted_clusters'] in out_of_bounds:
        return 'out_of_bounds'
    elif row['adjusted_clusters'] in high:
        return 'high'
    elif row['adjusted_clusters'] in med:
        return 'med'
    elif row['adjusted_clusters'] in low:
        return 'low'
    
def adjust_clusters(clusters_csv, analysis_csv, point_type = 'Binder', high_threshold = 10, low_threshold = 1, upper_limit = None):
    groups = extract_clusters_to_combine(clusters_csv, analysis_csv, point_type = point_type)

    cluster_df = pd.read_csv(clusters_csv)

    cluster_df['adjusted_clusters'] = cluster_df.apply(add_adjusted_clusters_column, axis=1, groups = groups)

    high, med, low, out_of_bounds = cluster_priority(cluster_df, high_threshold = high_threshold, low_threshold = low_threshold, upper_limit=upper_limit)

    cluster_df['priority'] = cluster_df.apply(add_priority_column, axis=1, high = high, med=med, low=low, out_of_bounds = out_of_bounds)

    cluster_df.to_csv(clusters_csv, index = False)


def adjust_clusters_folder(clusters_csv_folder, analysis_csv, point_type = 'Binder', high_threshold = 10, low_threshold = 1, upper_limit = None):

    csv_list = [f for f in os.listdir(clusters_csv_folder) if os.path.isfile(os.path.join(clusters_csv_folder, f)) and f.endswith('.csv')]

    for file in csv_list:
        print(f'Adjusting {file}')
        file = os.path.join(clusters_csv_folder, file)
        adjust_clusters(file, analysis_csv, point_type = point_type, high_threshold = high_threshold, low_threshold = low_threshold, upper_limit=upper_limit)