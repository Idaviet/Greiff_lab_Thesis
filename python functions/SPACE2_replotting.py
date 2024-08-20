#!/usr/bin/env python
# coding: utf-8
# In[1]:
# # Imports

python_files_folder = '/Users/isaacdaviet/Desktop/thesis/python_versions'

import sys
sys.path.append(python_files_folder)
# replace with directory containing the .py calculation files below
import SPACE2_analysis as sp2
import pdb_extraction as pdb
import pandas as pd 
import os
import SPACE2
from SPACE2 import reg_def
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:

### Functions to associate SPACE2 clusters with sequences and clusters from UMAP reductions. Ultimate output is excel file containing separate sheet for each separate reduction with the x-y coordinates of the specific sequence as well as a summary sheet indicating each SPACE2 cluster for each sequence and the total number of unique SPACE2 clusters

def list_space2_clusters(row):
    """
    Counts the number of unique values in a row, excluding the first four columns.

    Parameters:
    - row (Series): A pandas Series representing a row in a DataFrame.

    Returns:
    - list: A list of the unique values in the row (or None if no values in row)
    """
    # print(row)

    unique_values = row.unique().tolist()
    
    if 'unclustered' in unique_values:
        unique_values.remove('unclustered')

    unique_values = None if len(unique_values) == 0 else unique_values
    # print('\t',unique_values)
    return unique_values 

def count_unique_cell_values(row):
    """
    Counts the number of unique values in a row, excluding the first four columns.

    Parameters:
    - row (Series): A pandas Series representing a row in a DataFrame.

    Returns:
    - int: The number of unique values in the row.
    """
    # print(row)

    if row is None:
        return 0
    
    row_values = row.values.tolist()[4:]

    row_values = [value for value in row_values if value != 'unclustered']

    n_unique = len(set(row_values))


    return n_unique

def all_unique_clusters(row):
    """
    Extracts all unique clusters from a list of lists.

    Parameters:
    - row (list): A list of lists containing clusters.

    Returns:
    - list: A list of all unique clusters.
    """
    all_list = []
    for sublist in row:
        if sublist is not None:
            for i in sublist:
                if i not in all_list:
                    all_list.append(i)
    all_list = None if len(all_list) == 0 else all_list
    return all_list

def generate_unique_clusters_df(reduction_type, excel_file_or_df_dict):
    """
    Processes SPACE2 clusters data and adds unique clusters to the UMAP DataFrame.

    Parameters:
    - reduction_type (str): the reduction method by which SPACE2 clusters were isolated ('UMAP' or 'PCA') 
    - excel_file_or_df_dict (str or dict): Either the path to the Excel file containing the results/Summary of the SPACE2 clusters or a dictionary containing DataFrames derived from said excel.

    Returns:
    - DataFrame: The updated DataFrame with unique clusters added.
    """
    excel = False

    if isinstance(excel_file_or_df_dict, str):
        df = pd.read_excel(excel_file_or_df_dict, sheet_name='all_clusters')
        excel = True

    if isinstance(excel_file_or_df_dict, dict):
        df = excel_file_or_df_dict['all_clusters']

    clusters_df = df.iloc[:,4:]
    reductions = clusters_df.columns.tolist()

    metrics = pd.Series([reduction.split('-')[1] for reduction in reductions]).unique().tolist() if reduction_type == 'UMAP' else None

    pcs = None
    if reduction_type == 'PCA':
        pc_list = []
        for reduction in reductions:
            pc1 = reduction.split('-')[0].replace('PCA', '')
            pc2 = reduction.split('-')[1]
            pc_list.append(pc1)
            pc_list.append(pc2)
        pcs = pd.Series(pc_list).unique().tolist()

    new_df = df[['Sequences', 'iseq', 'Labels']]

    cluster_cols_list = []

    reduction_metrics = metrics if reduction_type == 'UMAP' else pcs

    for metric in reduction_metrics:
        if reduction_type == 'UMAP':
            matching_columns = [column for column in clusters_df.columns if metric in column]
        elif reduction_type == 'PCA':
            matching_columns = [column for column in clusters_df.columns if f'PCA{metric}-' or f'-{metric}'in column]
            
        metrics_df = df[matching_columns]
        
        metric_name = f'PC{metric}' if reduction_type == 'PCA' else metric

        cluster_list = f'{metric_name}_unique_clusters'
        cluster_cols_list.append(cluster_list)

        new_df[cluster_list] =metrics_df.apply(list_space2_clusters, axis = 1)

        new_df[f'n_{metric_name}_unique_clusters'] = new_df[f'{metric_name}_unique_clusters'].apply(lambda x: len(list(set(x))) if x is not None else 0)

    new_df['all_SPACE2_clusters'] = new_df[cluster_cols_list].apply(all_unique_clusters, axis = 1)
    new_df['total_SPACE2_clusters'] =  new_df['all_SPACE2_clusters'].apply(lambda x: len(list(set(x))) if x is not None else 0)

    # if excel == True:
    #     with pd.ExcelWriter(excel_file_or_df_dict, mode='a', engine='openpyxl') as writer:
    #     # Write the DataFrame to a new sheet
    #         new_df.to_excel(writer, sheet_name='unique_clusters_by_sequence', index=False)

        # new_df.to_csv('/Users/isaacdaviet/Desktop/test.csv', index = False)

    # if excel == False:
    #     excel_file_or_df_dict['unique_clusters_by_sequence'] = new_df

    return new_df

def add_file_paths_to_all_summaries(list, igfold_outfile):
    """
    Generates full file paths for all cluster files.

    Parameters:
    - cluster_files (list): A list of cluster file names.
    - igfold_outfile (str): The path to the directory containing the igfold pdb files (or whatever folder filepath to add).

    Returns:
    - list: A list of full file paths for all clusters.
    """
    full_file_paths = []
    for cluster_file in list:
        file_path = os.path.join(igfold_outfile, cluster_file)
        full_file_paths.append(file_path)

    return full_file_paths

def find_superclusters(clusters_by_seq_df, igfold_outfile, cdr_selection, chain_selection, rmsd_threshold, n_jobs):
    """
    Finds superclusters based on given parameters.

    Parameters:
    - clusters_by_seq_df (DataFrame): DataFrame containing clusters by sequence.
    - igfold_outfile (str): The directory containing the igfold pdb files.
    - cdr_selection (list): List of CDR selections used as input for SPACE2 algorithm (see SPACE2 documentation for details).
    - chain_selection (list): List of chain selections used as input for SPACE2 algorithm (see SPACE2 documentation for details).
    - rmsd_threshold (float): RMSD threshold used as input for SPACE2 algorithm (see SPACE2 documentation for details).
    - n_jobs (int): Number of jobs for processing used as input for SPACE2 algorithm (see SPACE2 documentation for details).

    Returns:
    - DataFrame: DataFrame where each sequence has their associated superclusters stored in the same order as the 'labels.csv' dataframe.
    """
    n_seqs = len(clusters_by_seq_df)

    cluster_columns = clusters_by_seq_df.columns.tolist()[3:]
    sub_columns = cluster_columns.copy()
    
    super_df = clusters_by_seq_df.copy()
    labels_df = super_df[['Sequences', 'iseq', 'Labels']]

    for column in cluster_columns:
        if 'n_' in column or 'total_' in column:
            cluster_columns.remove(column)

    clusters_df = clusters_by_seq_df[cluster_columns]

    selection = []
    anchor = []
    for i in cdr_selection:
        selection.append(reg_def[i])
    for i in chain_selection:
        anchor.append(reg_def[i])
    cdr_selection = selection
    chain_selection=anchor

    rows = []
    count = 0
    for index, row in clusters_df.iterrows():
        count +=1
        print(f'Calculated {count} of {n_seqs} sequences') if count % 100 == 0 else None

        new_row = []
        for file_list in row:
            full_path_list = add_file_paths_to_all_summaries(file_list, igfold_outfile) if file_list is not None else None
            # print(full_path_list)

            if full_path_list is not None and len(full_path_list) > 1:
                structure_df = SPACE2.agglomerative_clustering(full_path_list, selection = cdr_selection, anchors=chain_selection, cutoff=rmsd_threshold, n_jobs = n_jobs) 

                superclusters = structure_df['cluster_by_rmsd'].unique().tolist()

                clusters = []
                for pdb_path in superclusters:
                    cluster_file = pdb_path.split('/')[-1]
                    clusters.append(cluster_file)
                    superclusters = clusters
                    n_superclusters = len(set(superclusters))


                if len(superclusters) == 1:
                    superclusters = superclusters[0]
                    n_superclusters = 1

            elif full_path_list is not None and len(full_path_list) == 1:
                superclusters =full_path_list[0].split('/')[-1]
                n_superclusters =1

            else:
                superclusters = None
                n_superclusters = 0

            new_row.append(superclusters)
            new_row.append(n_superclusters)
        
        rows.append(new_row)

    sub_df = pd.DataFrame(rows, columns = sub_columns)
    super_df = pd.concat([labels_df, sub_df], axis = 1)
    return super_df

def assign_space2_cluster(row, space2_dfs, space2_algorithm):
    """
    Assigns SPACE2 clusters to sequence data points based on iseq values. Use '.apply' on pd.dataframe

    Parameters:
    - row (pd.Series): A row from the UMAP DataFrame containing iseq, priority, and Cluster information.
    - space2_dfs (dict): A dictionary of DataFrames where keys are SPACE2 cluster sheet names and values are DataFrames.
    - space2_algorithm (str): The algorithm used for SPACE2 clustering.

    Returns:
    - str: The assigned SPACE2 cluster for the given sequence data point in, or 'unclustered' if no match is found.
    """
    
    # Get values required to ID associated SPACE2_
    iseq = int(row['iseq'])
    priority = row['priority']
    umap_cluster = row['Cluster']

    # Get the associated umap cluster sheet in the space2_df dictionary and extract the associated df
    full_cluster_name = f"{space2_algorithm}_{priority}_{umap_cluster}"
    sp2_cluster_df = space2_dfs.get(full_cluster_name, 'unclustered')

    # Get SPACE2 cluster based on iseq
    if isinstance(sp2_cluster_df, pd.DataFrame):
        sp2_iseqs = sp2_cluster_df['ID'].str.split('_').str[5].astype(int)
        mask = sp2_iseqs == iseq
        
        if mask.any():
            return sp2_cluster_df.loc[mask, 'cluster_by_rmsd'].iloc[0]
        else:
            return 'unclustered'
    else:
        return 'unclustered'

def create_umap_space2_df(dbscan_cluster_csv, space2_cluster_xl):
    """
    Creates a DataFrame with assigned SPACE2 clusters based on UMAP data points. Resulting dataframe contains the associated sequences, labels, iseqs, dbscan clusters, and SPACE2 clusters 

    Parameters:
    - dbscan_cluster_csv (str): Path to the CSV file containing UMAP data points.
    - space2_cluster_xl (str): Path to the Excel file containing SPACE2 cluster data.

    Returns:
    - pd.DataFrame: UMAP DataFrame with SPACE2 cluster assignments.
    """
    # Extract reduction (graph)
    reduction = dbscan_cluster_csv.split('/')[-1].split('_')[0:-1]
    reduction = ''.join(reduction)

    # Extract umap df to which space2 clusters will be added and add columns
    umap_df = pd.read_csv(dbscan_cluster_csv)
    umap_df['SPACE2_cluster'] = None

    # Read space2 cluster excel file and create a dictionary of DataFrames
    space2_xl = pd.ExcelFile(space2_cluster_xl, engine = 'openpyxl')
    space2_dfs = {sheet: pd.read_excel(space2_cluster_xl, sheet_name=sheet) 
                  for sheet in space2_xl.sheet_names[1:]}

    space2_algorithm = list(space2_dfs.keys())[0].split('_')[0]

    # Iterate through umap_df rows and perform vectorized operations
    umap_df['SPACE2_cluster'] = umap_df.apply(lambda row: assign_space2_cluster(row, space2_dfs, space2_algorithm), axis=1)

    return umap_df

# test_dbscan = r'/Users/isaacdaviet/Desktop/results/clustering/UMAP_dbscan_clusters/cosine_clusters/csv_files/UMAP_Mason-cosine-3-35-0.0-1-3_dbscanClusters-0.125-30.csv'
# test_space2 = r'/Users/isaacdaviet/Desktop/results/SPACE2_analysis/UMAP_reductions/structural_clusters/cosine/UMAP_Mason-cosine-3-35-0.0-1-3_DBsc-0.125-30_SPACE2_agglomerative_1.25.xlsx'
# test_save = r'/Users/isaacdaviet/Desktop'

# umap_df = create_umap_space2_df(test_dbscan, test_space2)

# umap_df.to_csv(os.path.join(test_save, 'test.csv'))

def get_xl_files(folder_path):
    """
    Retrieves a list of Excel files (with extension .xlsx or .xl) from a given folder path and its subfolders.

    Parameters:
    - folder_path (str): The path to the folder containing the Excel files.

    Returns:
    - list: A list of file paths to the Excel files.
    """
    xl_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.xlsx') or file.endswith('.xl'):
                xl_files.append(os.path.join(root, file))
    return xl_files

def get_csv_files(folder_path):
    """
    Retrieves a list of CSV files (with extension .csv) from a given folder path and its subfolders.

    Parameters:
    - folder_path (str): The path to the folder containing the CSV files.

    Returns:
    - list: A list of file paths to the CSV files.
    """
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def remove_files_from_list(file_list, files_to_exclude_list):
    """
    Removes files specified in the 'files_to_exclude_list' from the 'file_list'.

    Parameters:
    - file_list (list): The list of file paths from which files will be removed.
    - files_to_exclude_list (list): The list of file paths to be excluded.

    Returns:
    - list: The updated list of file paths after removing the excluded files.
    """
    for i in file_list:
        for j in files_to_exclude_list:
            updated_list = file_list.remove(i) if j in i else file_list
    return updated_list

def order_dataframe_by_original_sequence_order(template_df_or_csv, original_column_name, df_or_csv_to_reorder, reordered_df_column_name, print_heads = False):
    # Load or use the template DataFrame
    if isinstance(template_df_or_csv, str) and '.csv' in template_df_or_csv:
        df_template = pd.read_csv(template_df_or_csv, dtype={original_column_name: str})
    else:
        df_template = template_df_or_csv.copy()

    # Load or use the DataFrame to reorder
    if isinstance(df_or_csv_to_reorder, str) and '.csv' in df_or_csv_to_reorder:
        df_to_reorder = pd.read_csv(df_or_csv_to_reorder)
    else:
        df_to_reorder = df_or_csv_to_reorder.copy()

    # Ensure the column names match for merging
    # df_template = df_template[[original_column_name]].drop_duplicates()  # Ensure no duplicate sequences
    df_template.rename(columns={original_column_name: reordered_df_column_name}, inplace=True)

    # Merge based on sequence to get the order from df_template
    df_ordered = pd.merge(df_template, df_to_reorder, on=reordered_df_column_name, how='left')

    # Debugging output
    if print_heads is not False:
        print("Template Head:", df_template[reordered_df_column_name].head())
        print("To Reorder Head:", df_to_reorder[reordered_df_column_name].head())
        print("Ordered DF Head:", df_ordered[reordered_df_column_name].head())

    return df_ordered

def generate_dict_of_all_umap_reductions(space2_clusters_folder, umap_csv_folder, exclude_files = []):   
    """
    Generates a dictionary of DataFrames where keys are reduction names and values are DataFrames containing SPACE2 cluster information.

    Parameters:
    - space2_clusters_folder (str): The path to the folder containing SPACE2 cluster Excel files.
    - umap_csv_folder (str): The path to the folder containing UMAP CSV files.
    - exclude_files (list): A list of file names to be excluded from processing.

    Returns:
    - dict: A dictionary where keys are reduction names and values are DataFrames containing SPACE2 cluster information.
    """    

    df_dict = {}

    space2_files = get_xl_files(space2_clusters_folder)
    space2_files = remove_files_from_list(space2_files, exclude_files)
    umap_files = get_csv_files(umap_csv_folder)
    umap_files = remove_files_from_list(umap_files, exclude_files)

    for sp2_file in space2_files:
        reduction = sp2_file.split('/')[-1]
        reduction = reduction.split('_')[0:2]
        reduction = '_'.join(reduction)

        for u_file in umap_files:
            if reduction in u_file:
                df_dict[reduction] = create_umap_space2_df(u_file, sp2_file)

    return df_dict

# space2_clusters_folder = r'/Users/isaacdaviet/Desktop/results/SPACE2_analysis/UMAP_reductions/structural_clusters'
# umap_csv_folder = r'/Users/isaacdaviet/Desktop/results/clustering/UMAP_dbscan_clusters'

# exclude_files = ['umap_all_summaries.xlsx', 'mason_umap_clustering_analysis_binders.csv']

# df_dict = generate_dict_of_all_umap_reductions(space2_clusters_folder, umap_csv_folder, exclude_files = exclude_files)


# def list_space2_clusters(row):
#     """
#     Counts the number of unique values in a row, excluding the first four columns.

#     Parameters:
#     - row (Series): A pandas Series representing a row in a DataFrame.

#     Returns:
#     - int: The number of unique values in the row.
#     """
#     unique_values = row[4:].nunique()  # Exclude the first four columns
#     return unique_values    

def add_all_clusters_df(df_dict, umap_df):
    """
    Adds SPACE2 cluster information to a DataFrame and counts the number of unique clusters.

    Parameters:
    - df_dict (dict): A dictionary containing DataFrames with SPACE2 cluster information.
    - umap_df (DataFrame): The DataFrame containing UMAP data.

    Returns:
    - dict: The updated dictionary with a new key 'all_clusters' and its associated DataFrame.
    """
    all_clusters_df = umap_df[['Sequences', 'iseq', 'Labels']]
    all_clusters_df['n_SPACE2_clusters'] = None
    # all_clusters_df['SPACE2_clusters'] = None
    
    for reduction, df in df_dict.items():
        all_clusters_df[reduction] = df['SPACE2_cluster']

    all_clusters_df['n_SPACE2_clusters'] = all_clusters_df.apply(count_unique_cell_values, axis=1)

    # all_clusters_df['SPACE2_clusters'] = all_clusters_df.apply(list_space2_clusters, axis=1)

    df_dict['all_clusters'] = all_clusters_df

    return df_dict

def save_dict_to_excel(dict_of_dfs, save_file):
    """
    Saves a dictionary of DataFrames to an Excel file, with each DataFrame as a separate sheet.

    Parameters:
    - dict_of_dfs (dict): A dictionary where keys are sheet names and values are DataFrames.
    - save_file (str): The file path to save the Excel file.

    Returns:
    - None
    """
    ordered_keys = ['all_clusters'] + [key for key in dict_of_dfs if key != 'all_clusters']
    writer = pd.ExcelWriter(save_file, engine='xlsxwriter')
    
    for key in ordered_keys:
        if key != 'all_clusters' and 'PCA' not in key:
            shortened_name = key.split('-')[1:8] # Limiting the length of the sheet name
            shortened_name = '-'.join(shortened_name)

        elif key != 'all_clusters' and 'PCA' in key:
            shortened_name = key

        elif key == 'all_clusters':
            shortened_name = 'all_clusters'
        dict_of_dfs[key].to_excel(writer, sheet_name=shortened_name, index=False)

    writer.close()  # Use close() to save the Excel file

def assign_space2_cluster_to_pca(row, space2_df):
    """
    Assign SPACE2 and PCA clusters to a given sequence in dataframe with original labels.csv file order based on iseq and label.

    Parameters:
        row (pandas.Series): A row from labels.csv df DataFrame containing 'iseq' and 'Labels'.
        space2_df (pandas.DataFrame): DataFrame containing SPACE2 cluster information.

    Returns:
        tuple: A tuple containing SPACE2 and PCA cluster values.
    """
    # Get values required to ID associated 
    iseq = int(row['iseq'])
    label = 'AgPos' if row['Labels'] == 'Binder' else 'AgNeg'
    iseqlabel = f'{iseq}_{label}'

    # Get SPACE2 cluster based on iseq
    if isinstance(space2_df, pd.DataFrame):
        sp2_iseqs = space2_df['iseq_labels']
        mask = sp2_iseqs == iseqlabel
        
        if mask.any():
            sp2cl =space2_df.loc[mask, 'SPACE2_cluster'].iloc[0]
            pcacl =space2_df.loc[mask, 'PCA_cluster'].iloc[0]
            return sp2cl, pcacl
        else:
            return 'unclustered', 'unclustered'
    else:
        return 'unclustered', 'unclustered'

def generate_dict_of_all_pca_reductions(pca_space2_results_xl, pca_coord_csv):
    """
    Generates a dictionary of all PCA reductions based on given Excel and CSV files.
    
    Parameters:
        pca_space2_results_xl (str): Path to the Excel file containing PCA results.
        pca_coord_csv (str): Path to the CSV file containing PCA coordinates.
        
    Returns:
        dict: A dictionary containing PCA reductions, where keys are specific reduction name and values are dataframes of each reduction with sequences,labels, iseqs, and reduction specific cluster names and SPACE2 clusters.
    """

    coord_df = pd.read_csv(pca_coord_csv)
    coord_df = coord_df.rename(columns={'label':'Labels'})

    coord_columns = coord_df.columns.tolist()


    xl = pd.ExcelFile(pca_space2_results_xl, engine = 'openpyxl')
    pca_space2_dict = {}
    pca_clusters = []
    pc_combos = []
    
    for sheet_name in xl.sheet_names:
        pca_space2_dict[sheet_name] = xl.parse(sheet_name)
        
        if sheet_name != 'summary':
            pca_clusters.append(sheet_name)
            pcs = sheet_name.split('_')[1] if 'PCA' in sheet_name.split('_')[1] else sheet_name.split('_')[2]

            if pcs not in pc_combos:
                pc_combos.append(pcs)

    pca_df_dict = {}    

    for combo in pc_combos:
        pca_df = pd.DataFrame()
        hold = combo.replace('PCA', '')
        hold = hold.split('-')
        pc1, pc2 = hold[0], hold[1]
        print(pc1, pc2)

        x_col, y_col = f'PCA{pc1}_', f'PCA{pc2}_'

        for column in coord_columns:
            if x_col in column:
                x_col = column
            if y_col in column:
                y_col = column

        pca_df[x_col] = coord_df[x_col]
        pca_df[y_col] = coord_df[y_col]

        pca_df[['Sequences', 'iseq', 'Labels']] = coord_df[['Sequences', 'iseq', 'Labels']]


        new_df = pd.DataFrame(columns=['iseq', 'Labels', 'PCA_cluster', 'SPACE2_cluster'])

        for key, space2_df in pca_space2_dict.items():
            if combo in key and key != 'summary':

                sub_df = pd.DataFrame(columns=['iseq', 'Labels', 'PCA_cluster', 'SPACE2_cluster'])
                
                sub_df['iseq'] = space2_df['ID'].apply(lambda x: str(x.split('_')[5]))

                sub_df['Labels'] = space2_df['ID'].apply(lambda x: x.split('_')[2])

                sub_df['iseq_labels'] = sub_df['iseq'] + '_' + sub_df['Labels']

                sub_df['SPACE2_cluster'] = space2_df['cluster_by_rmsd']

                # Assigning PCA_cluster value to each row in sub_df
                sub_df['PCA_cluster'] = key

                new_df = pd.concat([new_df, sub_df], ignore_index=True)

        pca_df[['SPACE2_cluster', 'PCA_cluster']] = pca_df.apply(lambda row: pd.Series(assign_space2_cluster_to_pca(row, new_df)), axis=1)

        pca_df_dict[combo] = pca_df

    return pca_df_dict


# df_dict = generate_dict_of_all_umap_reductions(pca_space2_results_xl, pca_coord_csv)


def generate_all_cluster_excel(reduction_type, space2_clusters_folder_or_pca_space2_results_xl, umap_csv_folder_or_pca_coord_csv, save_file, exclude_files = None):
    """Generates an Excel file containing combined data from UMAP SPACE2 clusters folder and UMAP CSV files OR from single PCA SPACE2 result xl file and single PCA reduction csv file.

    Args:
        reduction_type (str): the reduction method by which SPACE2 clusters were isolated ('UMAP' or 'PCA') 

        space2_clusters_folder_or_pca_space2_results_xl (str): Path to the folder containing SPACE2 cluster Excel files OR Excel file containing PCA/SPACE2 clusters
        
        umap_csv_folder (str): Path to the folder containing UMAP CSV files OR single PCA reduction csv file

        save_file (str): Path to save the generated Excel file.
        exclude_files (list of str): List of file names to exclude from processing.

    Returns:
        dict: A dictionary containing DataFrames with SPACE2 cluster information combined with UMAP data.

    """
    if reduction_type == 'UMAP':
        df_dict = generate_dict_of_all_umap_reductions(space2_clusters_folder_or_pca_space2_results_xl, umap_csv_folder_or_pca_coord_csv, exclude_files = exclude_files) 
          
    elif reduction_type== 'PCA':
        df_dict = generate_dict_of_all_pca_reductions(space2_clusters_folder_or_pca_space2_results_xl, umap_csv_folder_or_pca_coord_csv)

    dimred_df = next(iter(df_dict.values()))
    df_dict = add_all_clusters_df(df_dict, dimred_df)
    save_dict_to_excel(df_dict, save_file)
    return df_dict


def get_df_dict_from_excel(file_path):
    """
    Extracts DataFrames from the all summaries Excel file. To be used if df_dict returned from generate_dict_of_all_umap_reductions or generate_all_cluster_excel was saved to excel but lost before could proceed to next steps.

    Parameters:
    - file_path (str): The path to the Excel file.

    Returns:
    - dict: A dictionary of DataFrames.
    """
    # Create an empty dictionary to store DataFrames
    sheet_dict = {}

    # Open the Excel file
    xls = pd.ExcelFile(file_path, engine = 'openpyxl')

    # Iterate through each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        # Store the DataFrame in the dictionary with the sheet name as the key
        sheet_dict[sheet_name] = df

    return sheet_dict

def replot_by_supercluster(filtered_df,space2_cluster_column_name, n_reduction_column_ext, n_red_ext, colored_by_ext, n_params, title, save_file):
    """
    Replot UMAP scatterplots colored by superclusters or clusters, based on cluster_column input. Generates png file with 12 graphs (4 rows by 3 columns). Columns 1-3 are All - Binders - Non Binders. Rows for UMAP to UMAP are All Sequences - SPACE2 clusters at n_metric >1 - UMAP clusters at n_metric > input - SPACE2 clusters at n_metric > the input. Rows for UMAP to PCA are SPACE2 clusters at n_metric >1 - SPACE2 clusters at n_metric > input 
    
    Args:
    - supercluster_df (DataFrame): DataFrame containing supercluster information.
        Columns: [Sequences	iseq, Labels, cosine_unique_clusters, n_cosine_unique_clusters, euclidean_unique_clusters, n_euclidean_unique_clusters, correlation_unique_clusters, n_correlation_unique_clusters, manhattan_unique_clusters	n_manhattan_unique_clusters, hamming_unique_clusters, n_hamming_unique_clusters, all_SPACE2_clusters, total_SPACE2_clusters, n_metrics]

    - umap_df (DataFrame): DataFrame containing UMAP coordinates.
        Columns:[X-coord, Y-coord, Sequences, iseq, Labels, Cluster, reduction_clusters_column, priority] 

    - cluster_column (str): Name of the column containing cluster information.
    - n_metric (int): Minimum number of metrics confirming the cluster required to plot point.
    - title (str): Title of the plot.
    - save_file (str): File path to save the plot.
    - subtitle_size (int, optional): Size of subtitle font. Default is 60.
    - plot_to (str, optional): Plotting destination, either 'UMAP' or 'PCA'. Default is 'UMAP'. Will generate slightly different graphs 
    
    Returns:
    - None
    """
    
    ### set color pallette for b/nb
    labels_colors = {'Binder': 'red', 'Non Binder': 'blue'}

    ### Set 'replot_to' variable
    replot_to = 'UMAP' if 'UMAP' in colored_by_ext else 'PCA'


    ### create full copy of filtered_df and extract copies containing binders or non binders only
    full_df = filtered_df.copy()
    full_bdf = full_df[full_df['Labels'] == 'Binder']
    full_nbdf = full_df[full_df['Labels'] == 'Non Binder']



    ### Remove any data points unclustered in reduction graphs from b/nb and reduction  dataframes.
    unclustered_value = ['Binder-1', 'Non Binder-1'] if 'UMAP' in colored_by_ext else ['unclustered']

    for value in unclustered_value:
        full_bdf =  full_bdf[full_bdf['reduction_clusters'] != value]
        full_nbdf =  full_nbdf[full_nbdf['reduction_clusters'] != value]
        reduct_df = filtered_df[filtered_df['reduction_clusters']!= value]


    ### Remove any data points unclustered by SPACE2 in reductions_df and create separate binder/non binder dataframes
    reduct_df = reduct_df[reduct_df[space2_cluster_column_name] != 'unclustered']
    reduct_df = reduct_df[reduct_df[space2_cluster_column_name] != None]
    reduct_df = reduct_df[reduct_df[space2_cluster_column_name] != 0]

    bdf = reduct_df[reduct_df['Labels'] == 'Binder']

    nbdf = reduct_df[reduct_df['Labels'] == 'Non Binder']


    ext = ['All Points', 'Binders', 'Non Binders']
    fig_size = (100,100)

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=fig_size)


    ### Create lists of dataframes to be plotted
    all_full_dfs = [full_df, full_bdf, full_nbdf]
    all_filtered_dfs = [reduct_df, bdf, nbdf]

        
    for j in range(0, 12):

        ### Row 1: Plot the full df of binders vs non binders + full binder by reduction cluster + full non binder by reduction cluster
        if j <= 2:
            df = all_full_dfs[j%3]

            n_points = [len(full_df), len(full_bdf), len(full_nbdf)]

            n_reduct_clusters = [full_df['reduction_clusters'].nunique(), full_bdf['reduction_clusters'].nunique(), full_nbdf['reduction_clusters'].nunique()]

            palette = labels_colors if j%3 == 0 else 'colorblind'
            hue = 'Labels' if j%3 == 0 else 'reduction_clusters'
            # print(hue)

            full_title = f'All {n_points[0]} Sequence Data Points\n{n_points[1]} Binders (Red) vs {n_points[2]} Non Binders (Blue) ' if j%3 == 0 else f'All {ext[j%3]} Colored by {colored_by_ext}\n {len(df)} points - {n_reduct_clusters[j%3]} {replot_to} Clusters'

            scatterplot = sns.scatterplot(data=df, x=df['x'], y=df['y'], hue=hue, palette=palette, legend=True, s=20, ax=axs[0, j%3])
            scatterplot.set_title(full_title, fontsize=60)

            if j == 0:
                x_lim, y_lim = axs[0, 0].get_xlim(),axs[0, 0].get_ylim()

            scatterplot.set(xlim=x_lim, ylim=y_lim)


        ### Row 2: Plot binders vs non binders + binder SPACE2 clusters + non binder SPACE2 clusters at n_metric >= 1
        elif 2 < j <= 5:
            row = 1

            df = all_filtered_dfs[j%3]

            n_points = [len(all_filtered_dfs[0]), len(all_filtered_dfs[1]), len(all_filtered_dfs[2])]

            n_superclusters = [all_filtered_dfs[0]
            [space2_cluster_column_name].nunique(), all_filtered_dfs[1][space2_cluster_column_name].nunique(), all_filtered_dfs[2][space2_cluster_column_name].nunique()]

            palette = labels_colors if j%3 == 0 else 'colorblind'
            hue = 'Labels' if j%3 == 0 else space2_cluster_column_name

            full_title = f'All {n_points[0]} Sequences in SPACE2 Superclusters confirmed by >1 {n_red_ext}\n{n_points[1]} Binders (Red) vs {n_points[2]} Non Binders (Blue)' if j%3 == 0 else f'{ext[j%3]} with SPACE2 Superclusters confirmed by >1 {n_red_ext}\nColored by SPACE2 Supercluster - {len(df)} Points - {n_superclusters[j%3]} SPACE2 Clusters'

            scatterplot = sns.scatterplot(data=df, x=df['x'], y=df['y'], hue=hue, palette=palette, legend=False, s=20, ax=axs[row, j%3])
            scatterplot.set_title(full_title, fontsize=60)
            scatterplot.set(xlim=x_lim, ylim=y_lim)

      
        ### Row 3+4: Filter dfs to exclude points <= n_metric 
        elif 5 < j <= 11:
            new_filtered_dfs = []
            for df in all_filtered_dfs:
                new_filtered_dfs.append(df[df['n_param'] >= n_params])

            all_filtered_dfs=new_filtered_dfs

            n_points = [len(all_filtered_dfs[0]), len(all_filtered_dfs[1]), len(all_filtered_dfs[2])]

            n_reduct_clusters = [all_filtered_dfs[0]['reduction_clusters'].nunique(), all_filtered_dfs[1]['reduction_clusters'].nunique(), all_filtered_dfs[2]['reduction_clusters'].nunique()]

            n_superclusters = [all_filtered_dfs[0][space2_cluster_column_name].nunique(), all_filtered_dfs[1][space2_cluster_column_name].nunique(), all_filtered_dfs[2][space2_cluster_column_name].nunique()]

            ### Row 3: Plot binders vs non binder reduction clusters + binder reduction clusters + non reduction clusters at n_metric = input
            if j<=8:
                df = all_filtered_dfs[j%3]

                palette = labels_colors if j%3 == 0 else 'colorblind'
                hue = 'Labels' if j%3 == 0 else 'reduction_clusters'

                full_title = f'All {n_points[0]} Sequences in SPACE2 Superclusters confirmed by >{n_params-1} {n_red_ext}\nColored by Label - {n_points[1]} Binders (Red) vs {n_points[2]} Non Binders (Blue)' if j%3 == 0 else f'{ext[j%3]} With SPACE2 Superclusters confirmed by >{n_params-1} {n_red_ext}\nColored by {colored_by_ext} - {len(df)} Points - {n_reduct_clusters[j%3]} {replot_to} Clusters'

                scatterplot = sns.scatterplot(data=df, x=df['x'], y=df['y'], hue=hue, palette=palette, legend=True, s=20, ax=axs[2, j%3])
                scatterplot.set_title(full_title, fontsize=60)

                scatterplot.set(xlim=x_lim, ylim=y_lim)

            ### Row 4: Plot binders vs non binder space2 clusters + binder space2 clusters + non space2 clusters at n_metric = input
            else:
                row = 3

                df = all_filtered_dfs[j%3]

                palette =  'colorblind'
                hue = space2_cluster_column_name

                full_title = f'All {n_superclusters[0]} SPACE2 Superclusters confirmed by >{n_params-1} {n_red_ext}\nColored by SPACE2 Supercluster - {n_superclusters[1]} Binder vs {n_superclusters[2]} Non Binder Clusters' if j%3 == 0 else f'{ext[j%3]} With SPACE2 Superclusters confirmed by >{n_params-1} {n_red_ext}\nColored by SPACE2 Supercluster - {len(df)} Points - {n_superclusters[j%3]} SPACE2 Superclusters'

                scatterplot = sns.scatterplot(data=df, x=df['x'], y=df['y'], hue=hue, palette=palette, legend=True, s=20, ax=axs[row, j%3])
                scatterplot.set_title(full_title, fontsize=60)
                scatterplot.set(xlim=x_lim, ylim=y_lim)



    plt.suptitle(title, fontsize=80, y = 0.02)
    plt.subplots_adjust(top=10)
    rt = [0, 0.03, 1, 1]
    plt.tight_layout(rect=rt)
    # plt.show()
    plt.savefig(save_file)
    plt.close()

# supercluster_csv = '/Users/isaacdaviet/Desktop/results/SPACE2_analysis/UMAP_reductions/SPACE2_cluster_replotting/mason_umap-space2_superclusters.csv'
# spcl_df = pd.read_csv(supercluster_csv)
# test_umap_csv= '/Users/isaacdaviet/Desktop/results/clustering/UMAP_dbscan_clusters/euclidean_clusters/csv_files/UMAP_Mason-euclidean-3-120-0.1-2-3_dbscanClusters-0.08-20.csv'
# test_umap_df = pd.read_csv(test_umap_csv)
# n_metric = 3
# cluster_column = 'all_SPACE2_clusters'
# title = 'test'
# save_file =(os.path.join('/Users/isaacdaviet/Desktop/', f'all_scatterplots_{n_metric}metrics.png'))
# subtitle_size= 60
    
# replot_by_supercluster(spcl_df, test_umap_df, n_metric, title, subtitle_size, save_file)
import plotly.graph_objects as go
import plotly.express as px


def plotly_replot_by_supercluster(filtered_df, reduction_type, replot_to, point_type, reduction_name, dataset_name, space2_cluster_column_name, save_folder):
    """
    Replot UMAP scatterplots colored by superclusters or clusters, based on cluster_column input. Generates plotly html file with 12 graphs (4 rows by 3 columns). Columns 1-3 are All - Binders - Non Binders. Rows for UMAP to UMAP are All Sequences - SPACE2 clusters at n_metric >1 - UMAP clusters at n_metric > input - SPACE2 clusters at n_metric > the input. Rows for UMAP to PCA are SPACE2 clusters at n_metric >1 - SPACE2 clusters at n_metric > input 

    Args:
    - supercluster_df (DataFrame): DataFrame containing supercluster information.
        Columns: [Sequences iseq, Labels, cosine_unique_clusters, n_cosine_unique_clusters, euclidean_unique_clusters, n_euclidean_unique_clusters, correlation_unique_clusters, n_correlation_unique_clusters, manhattan_unique_clusters n_manhattan_unique_clusters, hamming_unique_clusters, n_hamming_unique_clusters, all_SPACE2_clusters, total_SPACE2_clusters, n_metrics]

    - umap_df (DataFrame): DataFrame containing UMAP coordinates.
        Columns:[X-coord, Y-coord, Sequences, iseq, Labels, Cluster, reduction_clusters_column, priority] 

    - cluster_column (str): Name of the column containing cluster information.
    - n_metric (int): Minimum number of metrics confirming the cluster required to plot point.
    - title (str): Title of the plot.
    - save_file (str): File path to save the plot.
    - subtitle_size (int, optional): Size of subtitle font. Default is 60.
    - plot_to (str, optional): Plotting destination, either 'UMAP' or 'PCA'. Default is 'UMAP'. Will generate slightly different graphs 
    
    Returns:
    - None
    """

    uncluster_values = ['Binder-1', 'Non Binder-1', 'unclustered', None, 0, "['mHER_H3_AgNeg_unique_fv_20955_igfold.pdb', 'mHER_H3_AgNeg_unique_fv_5306_igfold.pdb']", "['mHER_H3_AgNeg_unique_fv_11892_igfold.pdb', 'mHER_H3_AgNeg_unique_fv_20955_igfold.pdb']"]

    filtered_df['non binder'] = filtered_df['Labels'] == 'Non Binder'
    filtered_df['binder'] = filtered_df['Labels'] == 'Binder'


    for value in uncluster_values:
        # Check if the value is in the DataFrame column
        if value in filtered_df[space2_cluster_column_name].values:
            # Filter the DataFrame by excluding rows with the current value
            filtered_df = filtered_df[filtered_df[space2_cluster_column_name] != value]

        # Check if the value is in the 'reduction_clusters' column
        if value in filtered_df['reduction_clusters'].values:
            # Filter the DataFrame by excluding rows with the current value
            filtered_df = filtered_df[filtered_df['reduction_clusters'] != value]

    # print(filtered_df[filtered_df['Labels']== 'Non Binder'].head())


    spcl_fig = go.Figure()


    n_spcl = filtered_df[space2_cluster_column_name].nunique()

    spcl_title = f'Mapping of {reduction_type} Superclusters to {reduction_name} Graph\n {n_spcl} {point_type} clusters'

    spcl_save_file = f'{dataset_name}_REPLOTS_{reduction_type}spcl_to_{reduction_name}_{point_type}.html'

    spcl_fig = px.scatter(filtered_df,
                        x='x', 
                        y='y', 
                        color=space2_cluster_column_name, 
                        size_max=10,
                        title=spcl_title)
    
    show_all_button = dict(label="Show All", method="update", args=[{"visible": [True] * len(filtered_df)}])

    binder_visibility = filtered_df['binder'].tolist()
    non_binder_visibility = filtered_df['non binder'].tolist()

    binder_button = [dict(label='Binders', method="update", args=[{"visible": [vis] for vis in binder_visibility}])]
    non_binder_button = [dict(label='Non Binders', method="update", args=[{"visible": [vis] for vis in non_binder_visibility}])]


    # Add buttons for 'priority'
    priority_buttons = [dict(label=str(pri), method="update", args=[{"visible": [pri == priority for priority in filtered_df['priority']]}]) for pri in filtered_df['priority'].unique()] if replot_to == 'UMAP' else None

    # Add buttons for 'reduction_clusters'
    redcl_buttons = [dict(label=str(cl), method="update", args=[{"visible": [cl == cluster for cluster in filtered_df['reduction_clusters']]}]) for cl in filtered_df['reduction_clusters'].unique()]

    spcl_buttons = [dict(label=str(cl), method="update", args=[{"visible": [cl == cluster for cluster in filtered_df[space2_cluster_column_name]]}]) for cl in filtered_df[space2_cluster_column_name].unique()]




    # Combine all buttons
    all_spcl_buttons = [show_all_button] + binder_button + non_binder_button + priority_buttons + redcl_buttons if replot_to == 'UMAP' else [show_all_button] + binder_button + non_binder_button + redcl_buttons

    # all_spcl_buttons = [show_all_button] + binder_button

    # for button in all_spcl_buttons:
    #     print(f"Button: {button['label']}, Args: {button['args']}")


    # Update the layout with buttons
    spcl_fig.update_layout(updatemenus=[dict(buttons=all_spcl_buttons, direction="down", x=1.2, xanchor="left", y=1.15, yanchor="top")])


    # for button in all_spcl_buttons:
    #     visibility = button["args"][0]["visible"]
    #     print(f"Label: {button['label']}, Visibility: {visibility}")

    x_min, x_max = filtered_df['x'].min(), filtered_df['x'].max()
    y_min, y_max = filtered_df['y'].min(), filtered_df['y'].max()

    x_range = [x_min+(x_min/10), x_max+(x_max/10)]
    y_range = [y_min+(y_min/10), y_max+(y_max/10)]
    spcl_fig.update_layout(xaxis=dict(range=x_range), yaxis=dict(range=y_range))


    spcl_fig.write_html(os.path.join(save_folder, spcl_save_file))

    # redcl_fig = go.Figure()
    # Add buttons for 'reduction_clusters'
    # redcl_buttons = [dict(label=str(cl), method="update", args=[{"visible": [filtered_df['reduction_clusters'] == cl]}]) for cl in filtered_df['reduction_clusters'].unique()]

    # n_b_redcl, n_nb_redcl = bdf['reduction_clusters'].nunique(), nbdf['reduction_clusters'].nunique()

    # redcl_title = f' Replotting {reduction_type} Reduction Clusters to {reduction_name} Graph\n {n_b_redcl} Binder clusters vs {n_nb_redcl} Non Binder Clusters'

    # redcl_save_file = f'{dataset_name}_REPLOTS_{reduction_type}_{reduction_name}_redcls.html'

    # redcl_fig = px.scatter(filtered_df,
    #             x='x', 
    #             y='y', 
    #             color='reduction_clusters', 
    #             size_max=10,
    #             title=redcl_title)
    

    # redcl_fig.write_html(os.path.join(save_folder, redcl_save_file))


def replot_all_to_plotly(supercluster_reduction_type, replot_to, supercluster_df, clusters_by_reduction_dict, n_params, space2_cluster_column_name, save_folder, dataset_name = 'Mason',SPACE2_algorithm = 'agglomerative', min_rmsd = 1.25, n_jobs = 1):
    """
    Generate UMAP scatterplot replots for all files in a folder using replot_by_supercluster function.
    
    Args:
    - supercluster_df (str or pd.Dataframe): File path to the DataFrame containing supercluster information.
    - clusters_by_reduction_dict (str or dictionary): File path of excel containing reductions dataframes with coordinates and reduction clusters.
    - color_column (str): Name of the column containing cluster information by which points will be colored.
    - save_folder (str): Folder path to save the generated plots.
    - n_metric (int): Minimum number of metrics required to confirm supercluster.
    - SPACE2_algorithm (str, optional): SPACE2 clustering algorithm. Default is 'agglomerative'.
    - min_rmsd (float, optional): Minimum RMSD value. Default is 1.25.
    
    Returns:
    - None
    """

    spcl_df = pd.read_csv(supercluster_df) if type(supercluster_df) == str else supercluster_df

    df_dict = get_df_dict_from_excel(clusters_by_reduction_dict) if type(clusters_by_reduction_dict) == str else clusters_by_reduction_dict

    for reduction, df in df_dict.items():
        if reduction != 'all_clusters':
            df = order_dataframe_by_orginal_sequence_order(supercluster_df, 'Sequences', df, 'Sequences')
            print(f'Plotting:\t{reduction}')

            name_list = reduction.split('-')

            if replot_to == 'UMAP':
                metric = name_list[0]
                n_component = name_list[1]
                n_neighbors = name_list[2]
                min_dist = name_list[3]
                comp1 = name_list[4]
                comp2 = name_list[5]


                title = f'{dataset_name} data -- Binders Only -- {supercluster_reduction_type} Cluster to {replot_to} Reductions Replotting -- UMAP-{metric}-{n_component} Components - Components {comp1}&{comp2}\nNeighbors: {n_neighbors}  - Minimum Distance: {min_dist}\nSPACE2-{SPACE2_algorithm} - RMSD Cutoff: {min_rmsd} - Jobs: {n_jobs} -- {n_params} Metric Min.'
                
                binders_file_name = f'{dataset_name}_REPLOT-{supercluster_reduction_type}cl-to-{replot_to}-{metric}-{n_component}nC-{n_neighbors}nN-{min_dist}MD-{comp1}c1-{comp2}c2_SPACE2-{SPACE2_algorithm}-{min_rmsd}rmsd_{space2_cluster_column_name.split("_")[0]}-Min{n_params}Metric_binders.html'

                file_name = f'{dataset_name}_REPLOT-{supercluster_reduction_type}cl-to-{replot_to}-{metric}-{n_component}nC-{n_neighbors}nN-{min_dist}MD-{comp1}c1-{comp2}c2_SPACE2-{SPACE2_algorithm}-{min_rmsd}rmsd_{space2_cluster_column_name.split("_")[0]}-Min{n_params}Metric_nonbinders.html'
                
                metric_folder = os.path.join(save_folder, metric)
                if not os.path.exists(metric_folder):
                    os.makedirs(metric_folder)
                save_file = os.path.join(metric_folder, file_name)

                new_save_folder = metric_folder

            elif replot_to == 'PCA':
                pc1, pc2 = name_list[0].split('A')[1], name_list[1]

                ext_title = f'Components {pc1} & {pc2}' if replot_to == 'PCA' else f'{reduction} Reduction'

                ext_file = f'{pc1}-{pc2}' if replot_to == 'PCA' else f'-{reduction}'

                metric_cluster = space2_cluster_column_name.split('_')[0]


                file_name = f"{dataset_name}_REPLOTS-{supercluster_reduction_type}spcl-to-{replot_to}{ext_file}_SPACE2-{SPACE2_algorithm}-{min_rmsd}rmsd_min{n_params}Metrics-{metric_cluster}Clusters.html" 

                new_save_folder = save_folder


            spcl_col_list = ['Sequences', 'iseq', 'Labels', space2_cluster_column_name]

            
            new_df = pd.DataFrame(columns = spcl_col_list)

            reduction_clusters_column = 'adjusted_clusters' if replot_to == 'UMAP' else 'PCA_cluster'


            new_df[spcl_col_list] = spcl_df[spcl_col_list]
            new_df['n_param'] = spcl_df['n_metrics']
            new_df['reduction_clusters'] = df[reduction_clusters_column]
            if replot_to == 'UMAP':
                new_df['priority'] = df['priority'] 
            new_df['x'] = df.iloc[:, 0] 
            new_df['y'] = df.iloc[:, 1]

            new_bdf = new_df[new_df['Labels'] == 'Binder']
            new_nbdf =new_df[new_df['Labels'] == 'Non Binder']

            
            plotly_replot_by_supercluster(new_bdf, supercluster_reduction_type, replot_to, 'Binder', reduction, dataset_name, space2_cluster_column_name, new_save_folder)

            plotly_replot_by_supercluster(new_nbdf, supercluster_reduction_type, replot_to, 'Non Binder', reduction, dataset_name, space2_cluster_column_name, new_save_folder)


    return df_dict
    
def generate_all_umap_replots(supercluster_reduction_type, replot_to, supercluster_df, clusters_by_reduction_dict,  n_metric, space2_cluster_column_name, save_folder, dataset_name = 'Mason',SPACE2_algorithm = 'agglomerative', min_rmsd = 1.25, n_jobs = 1):
    """
    Generate UMAP scatterplot replots for all files in a folder using replot_by_supercluster function.
    
    Args:
    - supercluster_df (str or pd.Dataframe): File path to the DataFrame containing supercluster information.
    - clusters_by_reduction_dict (str or dictionary): File path of excel containing reductions dataframes with coordinates and reduction clusters.
    - color_column (str): Name of the column containing cluster information by which points will be colored.
    - save_folder (str): Folder path to save the generated plots.
    - n_metric (int): Minimum number of metrics required to confirm supercluster.
    - SPACE2_algorithm (str, optional): SPACE2 clustering algorithm. Default is 'agglomerative'.
    - min_rmsd (float, optional): Minimum RMSD value. Default is 1.25.
    
    Returns:
    - None
    """

    spcl_df = pd.read_csv(supercluster_df) if type(supercluster_df) == str else supercluster_df

    df_dict = get_df_dict_from_excel(clusters_by_reduction_dict) if type(clusters_by_reduction_dict) == str else clusters_by_reduction_dict

    for reduction, df in df_dict.items():
        if reduction != 'all_clusters':
            df = order_dataframe_by_orginal_sequence_order(supercluster_df, 'Sequences', df, 'Sequences')

            print(f'Plotting:\t{reduction}')

            name_list = reduction.split('-')

            if replot_to == 'UMAP':
                metric = name_list[0]
                n_component = name_list[1]
                n_neighbors = name_list[2]
                min_dist = name_list[3]
                comp1 = name_list[4]
                comp2 = name_list[5]


                title = f'{dataset_name} data -- {supercluster_reduction_type} Cluster to {replot_to} Reductions Replotting -- UMAP-{metric}-{n_component} Components - Components {comp1}&{comp2}\nNeighbors: {n_neighbors}  - Minimum Distance: {min_dist}\nSPACE2-{SPACE2_algorithm} - RMSD Cutoff: {min_rmsd} - Jobs: {n_jobs} -- {n_metric} Metric Min.'
                
                file_name = f'{dataset_name}_REPLOT-{supercluster_reduction_type}cl-to-{replot_to}-{metric}-{n_component}nC-{n_neighbors}nN-{min_dist}MD-{comp1}c1-{comp2}c2_SPACE2-{SPACE2_algorithm}-{min_rmsd}rmsd_{space2_cluster_column_name.split("_")[0]}-Min{n_metric}Metric.png'
                
                metric_folder = os.path.join(save_folder, metric)
                if not os.path.exists(metric_folder):
                    os.makedirs(metric_folder)
                save_file = os.path.join(metric_folder, file_name)

            elif replot_to == 'PCA':
                pc1, pc2 = name_list[0].split('A')[1], name_list[1]

                ext_title = f'Components {pc1} & {pc2}' if replot_to == 'PCA' else f'{reduction} Reduction'

                ext_file = f'{pc1}-{pc2}' if replot_to == 'PCA' else f'-{reduction}'

                metric_cluster = space2_cluster_column_name.split('_')[0]

                title = f'{dataset_name} data -- Replotting {supercluster_reduction_type}/SPACE2 Superclusters to {replot_to} {ext_title} \nSPACE2-{SPACE2_algorithm} - RMSD Cutoff: {min_rmsd} - Jobs: {n_jobs} -- {n_metric} Metric Min.'

                file_name = f"{dataset_name}_REPLOTS-{supercluster_reduction_type}spcl-to-{replot_to}{ext_file}_SPACE2-{SPACE2_algorithm}-{min_rmsd}rmsd_min{n_metric}Metrics-{metric_cluster}Clusters.png" 


                save_file = os.path.join(save_folder, file_name)


            spcl_col_list = ['Sequences', 'iseq', 'Labels', space2_cluster_column_name]
            
            new_df = pd.DataFrame(columns = spcl_col_list)

            reduction_clusters_column = 'adjusted_clusters' if replot_to == 'UMAP' else 'PCA_cluster'


            new_df[spcl_col_list] = spcl_df[spcl_col_list]
            new_df['n_param'] = spcl_df['n_metrics']
            new_df['reduction_clusters'] = df[reduction_clusters_column]
            new_df['x'] = df.iloc[:, 0] 
            new_df['y'] = df.iloc[:, 1]

            n_reduction_column = 'n_metric'
            n_red_ext = 'Distance Metrics'
            colored_by = 'UMAP Cluster' if replot_to == 'UMAP' else 'PCA Cluster'


            # print(new_df.head())
            
            replot_by_supercluster(new_df, space2_cluster_column_name,n_reduction_column, n_red_ext, colored_by, n_metric, title, save_file)

    return df_dict


def generate_all_PCA_replots(replot_to, supercluster_df, clusters_by_reduction_dict, space2_cluster_column_name, save_folder, n_components, dataset = 'Mason', SPACE2_algorithm = 'agglomerative', min_rmsd = 1.25, n_jobs = 1):
    """
    Plot UMAP/SPACE2 clusters to PCA scatterplot using replot_by_supercluster.
    
    Args:
    - supercluster_csv (str): File path to the DataFrame containing supercluster information.
    - pca_csv (str): File path to the PCA DataFrame.
    - selected_pcas_csv (str): File path to the selected PCAs DataFrame.
    - cluster_column (str): Name of the column containing cluster information.
    - save_folder (str): Folder path to save the generated plots.
    - pca_superclusters_csv (int): Minimum number of metrics or components required.
    - SPACE2_algorithm (str, optional): SPACE2 clustering algorithm. Default is 'agglomerative'.
    - min_rmsd (float, optional): Minimum RMSD value. Default is 1.25.
    
    Returns:
    - None
    """
    print('loading data')

    reduction_clusters_column = 'adjusted_clusters' if replot_to == 'UMAP' else 'PCA_cluster'

    supercluster_df = pd.read_csv(supercluster_df) if type(supercluster_df) == str else supercluster_df

    df_dict = get_df_dict_from_excel(clusters_by_reduction_dict) if type(clusters_by_reduction_dict) == str else clusters_by_reduction_dict

    print('data succesfully extracted')
    for reduction, df in df_dict.items():
        if reduction != 'all_clusters':
            pc1, pc2 = reduction[3], reduction[5]

            ext_title = f'Components {pc1} & {pc2}' if replot_to == 'PCA' else f'{reduction} Reduction'

            ext_file = f'{pc1}-{pc2}' if replot_to == 'PCA' else f'-{reduction}'

            metric_cluster = space2_cluster_column_name.split('_')[0]

            title = f'{dataset} data -- Replotting PCA/SPACE2 Superclusters to {replot_to} {ext_title} \nSPACE2-{SPACE2_algorithm} - RMSD Cutoff: {min_rmsd} - Jobs: {n_jobs} -- {n_components} Metric Min.'

            file_name = f"{dataset}_REPLOTS-PCAspcl-to-{replot_to}{ext_file}_SPACE2-{SPACE2_algorithm}-{min_rmsd}rmsd_min{n_components}Metrics-{metric_cluster}Clusters.png" 

            print(file_name)

            save_file = os.path.join(save_folder, file_name)
            
            spcl_col_list = ['Sequences', 'iseq', 'Labels', space2_cluster_column_name]
            new_df = pd.DataFrame(columns = spcl_col_list)

            reduction_clusters_column = 'adjusted_clusters' if replot_to == 'UMAP' else 'PCA_cluster'

            # print(df.head())

            new_df[spcl_col_list] = supercluster_df[spcl_col_list]
            new_df['n_param'] = supercluster_df['n_PCs']
            new_df['reduction_clusters'] = df[reduction_clusters_column]
            new_df['x'] = df.iloc[:, 0] 
            new_df['y'] = df.iloc[:, 1]

            n_reduction_column = 'n_PCs'
            n_red_ext = 'Components'
            colored_by = 'UMAP Cluster' if replot_to == 'UMAP' else 'PCA Cluster'


            print(new_df.head())
            
            replot_by_supercluster(new_df, space2_cluster_column_name,n_reduction_column, n_red_ext, colored_by, n_components, title, save_file)

    return df_dict



# supercluster_df = pd.read_csv(supercluster_csv)
# pca_df = pd.read_csv(pca_csv)
# plot_df = pd.DataFrame()
# plot_df['pc1'] = pca_df['PCA1_ExpVar:8.89%']
# plot_df['pc2'] = pca_df['PCA2_ExpVar:4.47%']
# plot_df['Labels'] = pca_df['label']
# plot_df['priority'] = None
# test= 'test'
# test_save= '/Users/isaacdaviet/Desktop/test.png'

# replot_by_umap_supercluster(supercluster_df, plot_df, cluster_column, n_metric, test, test_save, plot_to = 'PCA')

# %%
