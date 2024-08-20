# %% [markdown]
# ## Imports and setups

# %% [markdown]
# Substitute variables with appropriate file paths/selections

# %%
import sys
sys.path.append('/Users/isaacdaviet/Desktop/thesis/python_versions')
# replace with directory containing the .py calculation files below
import SPACE2_analysis as sp2
import pdb_extraction as pdb
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display, clear_output
import time
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB import PDBParser



# %% [markdown]
# ## Extract CDRH3 structure from igfold pdb files into separate pdb_files

# %%
def save_cdrh3_pdb_files(pdb_file, output_file):
    cdrh3, cdrh3_pos, atoms_lines = pdb.parse_pdb(pdb_file)
    
    with open(output_file, 'w') as pdb_output:
        for line in atoms_lines:

            line_list = line.split(' ')
            line_list = [item for item in line_list if item != '']
            line_list = [item for item in line_list if item !='\n']

            position = line_list[5]
            if position in cdrh3_pos:
                pdb_output.write(line)

# test_in = r'/Users/isaacdaviet/Desktop/mason_igfold_models/mason_igfold_models/igfold_outfile/mHER_H3_AgNeg_unique_fv_1_igfold.pdb'

# test_out = r'/Users/isaacdaviet/Desktop/test2_CDRH3.pdb'

# save_cdrh3_pdb_files(test_in, test_out)

def add_all_cdrh3_pdb_files(pdb_folder, output_folder):
    all_files = os.listdir(pdb_folder)
    pdb_files = [file for file in all_files if file.endswith('.pdb')]

    for file in pdb_files:
        output_file = file.replace('.pdb', 'CDRH3_ONLY.pdb')
        output_file = os.path.join(output_folder, output_file)
        pdb_file = os.path.join(pdb_folder, file)

        save_cdrh3_pdb_files(pdb_file, output_file)

# pdb_folder = r'/Users/isaacdaviet/Desktop/mason_igfold_models/mason_igfold_models/igfold_outfile'
# output_folder = r'/Users/isaacdaviet/Desktop/mason_igfold_models/mason_igfold_models/igfold_CDRH3_pdbs'

# add_all_cdrh3_pdb_files(pdb_folder, output_folder)

# %% [markdown]
# ## UMAP multicluster RMSD calculation

# %% [markdown]
# #### This section calculates the RMSD of the SPACE2 clusters contained within a single UMAP cluster, when applicable

# %%
def calculate_rmsd(pdb_file1, pdb_file2):
    # Parse the PDB files
    parser = PDBParser()
    structure1 = parser.get_structure('1', pdb_file1)
    structure2 = parser.get_structure('2', pdb_file2)

    # Extract atom coordinates
    atoms1 = []
    atoms2 = []
    for model1, model2 in zip(structure1, structure2):
        for chain1, chain2 in zip(model1, model2):
            for residue1, residue2 in zip(chain1, chain2):
                for atom1, atom2 in zip(residue1, residue2):
                    atoms1.append(atom1.get_coord())
                    atoms2.append(atom2.get_coord())

    # Convert atom coordinates to numpy arrays
    atoms1_array = np.array(atoms1)
    atoms2_array = np.array(atoms2)

    # Initialize SVDSuperimposer
    super_imposer = SVDSuperimposer()

    # Apply rotation and translation to atoms
    super_imposer.set(atoms1_array, atoms2_array)
    super_imposer.run()

    # Get RMSD
    rmsd = super_imposer.get_rms()
    return rmsd

# Example usage
# pdb2 = r'/Users/isaacdaviet/Desktop/test1_CDRH3.pdb'
# pdb1 = r'/Users/isaacdaviet/Desktop/test2_CDRH3.pdb'

# rmsd = calculate_rmsd(pdb1, pdb2)
# print("RMSD between the structures:", rmsd)



# %%
def generate_multi_clusters_df(all_summaries_xl):
    summaries_df = pd.read_excel(all_summaries_xl, sheet_name='all_summaries')

    multi_summary_df = summaries_df[summaries_df['n_structural_clusters'] > 1]

    return multi_summary_df

# all_summaries_xl = r'/Users/isaacdaviet/Desktop/results/SPACE2_analysis/UMAP_reductions/structural_clusters/all_summaries.xlsx'

# filtered_summaries_df =generate_multi_clusters_df(all_summaries_xl)

# filtered_summaries_df.to_csv('/Users/isaacdaviet/Desktop/test_filtered_df.csv')



# %%
def generate_multi_umaps_dict(filtered_summaries_df, type):

    multi_umaps = {}
    for index, row in filtered_summaries_df.iterrows():
        umap_cluster = f"{row['reduction_file']}__{row['cluster']}__{row['n_structural_clusters']}SPcl" if type == 'UMAP' else row['cluster_name']
        sp2_cluster = row['structural_cluster']
        n_abspc = row['n_abs%']
        # print(umap_cluster)
        # print(sp2_cluster)
        # print(n_abspc)
        # print('\n')

        if umap_cluster not in multi_umaps.keys():
            multi_umaps[umap_cluster] = [[sp2_cluster], [n_abspc]]

        else:
            multi_umaps[umap_cluster][0].append(sp2_cluster)
            multi_umaps[umap_cluster][1].append(n_abspc)

    return multi_umaps


# multi_umaps_dict = generate_multi_umaps_dict(filtered_summaries_df)
# for cluster, list in multi_umaps_dict.items():
#     print(cluster)
#     print('\t', list)
#     print('\n')

# %%
def add_rmsd_to_multi_umaps_dict(multi_umaps_dict, igfold_folder):
    count = 1

    for umap_cluster, list in multi_umaps_dict.items():
        sp2_clusters_lst = list[0]
        rmsd_dict = {}

        for i in range(0, len(sp2_clusters_lst)-1):
            cluster_1 = sp2_clusters_lst[i]
            for j in range(i+1, len(sp2_clusters_lst)):
                cluster_2 = sp2_clusters_lst[j]

                if cluster_1 != cluster_2:
                    file_2 = cluster_2.replace('.pdb', 'CDRH3_ONLY.pdb')
                    file_1 = cluster_1.replace('.pdb', 'CDRH3_ONLY.pdb')

                    name1 = cluster_1.split('_')
                    id1 = name1[-2]
                    type1= name1[2]
                    name1 = type1+'_'+id1

                    name2 = cluster_2.split('_')
                    id2 = name2[-2]
                    type2= name2[2]
                    name2 = type2+'_'+id2

                    pdb_file1 = os.path.join(igfold_folder, file_1)
                    pdb_file2 = os.path.join(igfold_folder, file_2)
                    # print(pdb_file1)
                    
                    rmsd = calculate_rmsd(pdb_file1, pdb_file2)

                    rmsd_dict[f'{i}vs{j}'] = rmsd

                    multi_umaps_dict[umap_cluster].append(rmsd_dict)
        # print(count)
        count +=1

    return multi_umaps_dict


# multi_umap_rmsd_dict = add_rmsd_to_multi_umaps_dict(multi_umaps_dict, igfold_folder)


    


# %%
def generate_rmsd_df(multi_umap_rmsd_dict, type):
    # Initialize an empty list to store row dictionaries
    rows = []

    lengths = []
    for key, lst in multi_umap_rmsd_dict.items():
        lengths.append(len(lst[0]))
    max_length = max(lengths)

    column_names = ['reduction', 'cluster', 'label','n_SPACE2_clusters', 'SPACE2_clusters+n_abs%']

    for i in range(0, max_length - 1):
        for j in range(i + 1, max_length):
            new_column = f'{i + 1}_vs_{j + 1}'
            column_names.append(new_column)

    column_names.append('avg_rmsd')

    for umap_cluster, lst in multi_umap_rmsd_dict.items():
        if type == 'UMAP':
            umap_cluster_lst = umap_cluster.split('__')

            reduction, umap_cluster, n_sp2cl = umap_cluster_lst[0], umap_cluster_lst[1], umap_cluster_lst[2]
            n_sp2cl = n_sp2cl.replace('SPcl', '')

            label = 'Non Binder' if 'Non Binder' in umap_cluster else 'Binder'

            row = {'reduction': reduction,'label':label, 'cluster': umap_cluster, 'n_SPACE2_clusters': n_sp2cl}

        elif type == 'PCA':
            label = 'Non Binder' if '_NB' in umap_cluster else 'Binder'
            pca_ls =umap_cluster.split('_')
            cluster = pca_ls[-1]
            pcas = pca_ls[-2].replace('PCA', '')
            pcas = pcas.split('-')
            pc1 = pcas[0]
            pc2= pcas[1]
            n_sp2cl = len(lst[0])

            row = {'reduction': umap_cluster, 'pc1': pc1, 'pc2': pc2,'label':label, 'cluster': cluster, 'n_SPACE2_clusters': n_sp2cl}


        sp2_cluster_ls, n_abspc_ls, rmsd_dict = lst[0], lst[1], lst[2]

        sp2_clusters_cell = ''
        for i in range(0, len(sp2_cluster_ls)):
            sp2_cluster = sp2_cluster_ls[i]
            n_abs = n_abspc_ls[i]

            cluster_str = f'{i + 1}: {sp2_cluster} ({n_abs}%)'
            sp2_clusters_cell += cluster_str

            if i != len(sp2_cluster_ls) - 1:
                sp2_clusters_cell += ' + '

        row['SPACE2_clusters+n_abs%'] = sp2_clusters_cell

        rmsds = []
        for key, rmsd in rmsd_dict.items():
            new_key = key.split('vs')
            n_1 = int(new_key[0])+1
            n_2 = int(new_key[1])+1
            new_key = f'{n_1}_vs_{n_2}'
            row[new_key] = round(rmsd, 2)
            rmsds.append(rmsd)
        avg_rmsd = round((sum(rmsds)/(len(rmsds))), 2)
        row['avg_rmsd'] =avg_rmsd

        # Append the row dictionary to the list
        rows.append(row)

    # Convert the list of row dictionaries to a DataFrame
    rmsd_df = pd.DataFrame(rows, columns=column_names)

    return rmsd_df


# rmsd_df = generate_rmsd_df(multi_umap_rmsd_dict)
# print(rmsd_df.head())


# %%
def add_rmsds_to_all_summaries(all_summaries_xl, rmsd_df):

    with pd.ExcelWriter(all_summaries_xl, engine = 'openpyxl', mode='a') as writer:
        rmsd_df.to_excel(writer, sheet_name='multi_cluster_rmsds', index = False)

# add_rmsds_to_all_summaries('/Users/isaacdaviet/Desktop/all_summaries.xlsx', rmsd_df)



# %%
def rmsds_summary_sheet(all_summaries_xl, igfold_folder, type):
    filtered_summaries_df = generate_multi_clusters_df(all_summaries_xl)
    multi_umaps_dict = generate_multi_umaps_dict(filtered_summaries_df, type)
    multi_umap_rmsd_dict = add_rmsd_to_multi_umaps_dict(multi_umaps_dict, igfold_folder)
    rmsd_df = generate_rmsd_df(multi_umap_rmsd_dict, type)
    add_rmsds_to_all_summaries(all_summaries_xl, rmsd_df)


# igfold_folder = r'/Users/isaacdaviet/Desktop/mason_igfold_models/mason_igfold_models/igfold_CDRH3_pdbs'

# rmsds_summary_sheet('/Users/isaacdaviet/Desktop/results/SPACE2_analysis/PCA_manual_clusters/all_summaries.xlsx', '/Users/isaacdaviet/Desktop/mason_igfold_models/mason_igfold_models/igfold_CDRH3_pdbs', 'PCA')

# %%
def calculate_avg_rmsd_of_dataset(iseq_incl_file, column, column_filter, igfold_pdb_file_name_format, title, n_sequences = 'all', pc_selected_seqs = 1, save_final_graph_file = None, show_updating_graph = 'y'):
    
    df = pd.read_csv(iseq_incl_file) if type(iseq_incl_file) == str else iseq_incl_file
    filtered_df = df[df[column] == column_filter] if column_filter != 'all' else df

    # 3: Randomize selection of iseq and plug them into the igfold_pdb_file_name_format
    n_seq = int(len(filtered_df)) if n_sequences == 'all' else n_sequences
    n_seq = int(round(n_seq * pc_selected_seqs)) if pc_selected_seqs is not None else n_seq

    random_df = filtered_df.sample(n=n_seq)

    tested_iseqs = []
    rmsd_calcs = []
    rmsd_means = []
    rmsd_medians = []

    count = 0
    fig_width = 8

    for index, row in random_df.iterrows():
        # Get the value in a specific column for the current row
        iseq_1 = row['iseq']
        label_1 = 'AgNeg' if 'Non Binder' in  row['Labels'] else 'AgPos'
        
        # Get the index of the next row
        next_index = index + 1
        
        copy_format = igfold_pdb_file_name_format

        # Check if the next index is within the DataFrame's index range
        if next_index < len(random_df) and iseq_1 not in tested_iseqs:
            tested_iseqs.append(iseq_1)

            # Get the value in a specific column for the next row
            next_row = random_df.iloc[next_index]
            iseq_2 = next_row['iseq']
            tested_iseqs.append(iseq_2)

            # print(iseq_1)

            label_2 = 'AgNeg' if 'Non Binder' in next_row['Labels'] else 'AgPos'

            pdb_1 = copy_format.replace('LABEL', str(label_1))
            pdb_1 = pdb_1.replace('ISEQ', str(iseq_1))


            pdb_2 = copy_format.replace('LABEL', str(label_2))
            pdb_2 = pdb_2.replace('ISEQ', str(iseq_2))

            # print(f"label_1: {label_1}, label_2: {label_2}\n")
            # print(f"pdb_1: {pdb_1}, pdb_2: {pdb_2}\n\n")


            rmsd = calculate_rmsd(pdb_1, pdb_2)
            new_mean = round(np.mean(rmsd_calcs), 4)
            new_median = round(np.median(rmsd_calcs), 4)
            # print(rmsd)

            rmsd_calcs.append(rmsd)
            rmsd_means.append(new_mean)
            rmsd_medians.append(new_median)

            count +=1

            fig_width = fig_width + 2 if count % 50 == 0 else fig_width


            if show_updating_graph == 'y':
                plt.clf()

                plt.figure(figsize=(fig_width, 6))

                # Plot rmsd_calcs
                plt.plot(rmsd_calcs, color='blue', label = f'RMSD Values')
                # Plot rmsd_means
                plt.plot(rmsd_means, color='green', label = f'RMSD Averages')
                # Plot rmsd_medians
                plt.plot(rmsd_medians, color='red', label = f'RMSD Medians')

                
                plt.xlabel(f'Iterations 1-{count} - {n_seq} Sequences')
                plt.ylabel(f'RMSD Values')
                plt.title(f'{title}\nFinal Calculations: Max RMSD= {round(max(rmsd_calcs), 4)} -- Min RMSD= {round(min(rmsd_calcs), 4)} -- Mean RMSD = {new_mean} -- Median RMSD = {new_median}')
                plt.legend()
                plt.xticks(np.arange(0, count, 10))

                # Display the updated plot
                plt.tight_layout()

                display(plt.gcf())
                clear_output(wait=True)
                time.sleep(1)  # Adjust the sleep time as needed
                
    plt.clf()

    plt.figure(figsize=(fig_width, 6))

    # Plot rmsd_calcs
    plt.plot(rmsd_calcs, color='blue', label = f'RMSD Values')
    # Plot rmsd_means
    plt.plot(rmsd_means, color='green', label = f'RMSD Averages')
    # Plot rmsd_medians
    plt.plot(rmsd_medians, color='red', label = f'RMSD Medians')

    
    plt.xlabel(f'Iterations 1-{count} - {n_seq} Sequences')
    plt.ylabel(f'RMSD Values')
    plt.title(f'{title}\nFinal Calculations: Max RMSD= {round(max(rmsd_calcs), 4)} -- Min RMSD= {round(min(rmsd_calcs), 4)} -- Mean RMSD = {new_mean} -- Median RMSD = {new_median}')
    plt.legend()
    plt.xticks(np.arange(0, count, 10))


    # Display the updated plot
    plt.tight_layout()
    plt.savefig(save_final_graph_file) if save_final_graph_file is not None else None
    plt.show() if show_updating_graph != 'y' else None
    plt.close()
            
    return rmsd_calcs, rmsd_means[1:], rmsd_medians[1:], n_seq # first value in means/medians list is nan. [1:] removes it

def calculate_rmsd_averages(iseq_incl_file, column, column_filter, igfold_pdb_file_name_format, n_sequences = 'all', pc_selected_seqs = 1):
    
    df = pd.read_csv(iseq_incl_file) if type(iseq_incl_file) == str else iseq_incl_file
    filtered_df = df[df[column] == column_filter] if column_filter != 'all' else df

    # 3: Randomize selection of iseq and plug them into the igfold_pdb_file_name_format
    n_seq = int(len(filtered_df)) if n_sequences == 'all' else n_sequences
    n_seq = int(round(n_seq * pc_selected_seqs)) if pc_selected_seqs is not None else n_seq

    random_df = filtered_df.sample(n=n_seq)
    random_df = random_df.reset_index(drop=True)

    tested_iseqs = []
    rmsd_calcs = []
    rmsd_means = []
    rmsd_medians = []

    count = 0

    for index, row in random_df.iterrows():
        # Get the value in a specific column for the current row
        iseq_1 = row['iseq']
        label_1 = 'AgNeg' if 'Non Binder' in  row['Labels'] else 'AgPos'
        
        # Get the index of the next row
        next_index = index + 1
        
        copy_format = igfold_pdb_file_name_format


        # Check if the next index is within the DataFrame's index range


        if next_index < len(random_df) and iseq_1 not in tested_iseqs:
            tested_iseqs.append(iseq_1)

            # Get the value in a specific column for the next row
            next_row = random_df.iloc[next_index]
            iseq_2 = next_row['iseq']
            tested_iseqs.append(iseq_2)

            # print(iseq_1)

            label_2 = 'AgNeg' if 'Non Binder' in next_row['Labels'] else 'AgPos'

            pdb_1 = copy_format.replace('LABEL', str(label_1))
            pdb_1 = pdb_1.replace('ISEQ', str(iseq_1))



            pdb_2 = copy_format.replace('LABEL', str(label_2))
            pdb_2 = pdb_2.replace('ISEQ', str(iseq_2))

            # print(f"label_1: {label_1}, label_2: {label_2}\n")
            # print(f"pdb_1: {pdb_1}, pdb_2: {pdb_2}\n\n")


            rmsd = calculate_rmsd(pdb_1, pdb_2)
            
            

            rmsd_calcs.append(rmsd)


            count +=1

           
    return rmsd_calcs



# %% [markdown]
# ### Violin Plots of RMSD values
# 

# %%
def summaries_violin_plot(x_axes, y_axes, data_set, title, file_name, font_size, save_path, inner_plot='box'):
    """
    Generate a violin plot with optional point cloud and standard deviation annotation.

    Args:
    - x_axes (str): Name of the column containing the x-axis data.
    - y_axes (str): Name of the column containing the y-axis data.
    - data_set (DataFrame): DataFrame containing the dataset.
    - title (str): Title of the plot.
    - font_size (int): Font size of the title.
    - save_path (str): File path to save the plot.
    - inner_plot (str, optional): Type of inner plot. Default is 'box'.

    Returns:
    - None
    """
    # Calculate standard deviation, median, and mean for each unique value in x_axes
    summary_stats = data_set.groupby(x_axes)[y_axes].agg(['std', 'median', 'mean', 'count']).reset_index()
    summary_stats.columns = [x_axes, 'Std_Dev', 'Median', 'Mean', 'Count']

    # Create a square figure
    plt.figure(figsize=(10, 10))

    # Create a violin plot with adjusted x-axis position
    sns.violinplot(x=x_axes, y=y_axes, data=data_set, inner=inner_plot, position=1)

    # Add point cloud
    sns.swarmplot(x=x_axes, y=y_axes, data=data_set, color='k', alpha=0.5, s=3)

    # Set the title with the optimal font size
    plt.title(title, fontsize=font_size)

    # Set x-axis labels with summary statistics included
    x_labels = [f'{row[x_axes]}\nCount: {row["Count"]}\nStd Dev: {row["Std_Dev"]:.2f}\n Median: {row["Median"]:.2f}\nMean: {row["Mean"]:.2f}' for index, row in summary_stats.iterrows()]
    plt.xticks(ticks=range(len(summary_stats)), labels=x_labels)

    # Adjust aspect ratio to make the plot square
    # plt.gca().set_aspect('equal')

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()  # Close the figure after saving




def get_metric(reduction):
    # Example transformation, you can replace this with your custom logic
    if isinstance(reduction, str):
        parts = reduction.split('_')
        if len(parts) > 1:
            metric = parts[1].split('-')[1]
            return metric

# %%
# # Example data
# all_summaries_file = '/Users/isaacdaviet/Desktop/results/SPACE2_analysis/PCA_manual_clusters/all_summaries.xlsx'
# save_path = '/Users/isaacdaviet/Desktop/results/SPACE2_analysis/PCA_manual_clusters'

# umap_all_summaries_file ='/Users/isaacdaviet/Desktop/results/SPACE2_analysis/UMAP_reductions/structural_clusters/umap_all_summaries.xlsx'
# pca_all_summaries_file = '/Users/isaacdaviet/Desktop/results/SPACE2_analysis/PCA_manual_clusters/pca_all_summaries.xlsx'

# umap_save_path = '/Users/isaacdaviet/Desktop/results/SPACE2_analysis/UMAP_reductions/structural_clusters/all_summaries graphs'
# pca_save_path = '/Users/isaacdaviet/Desktop/results/SPACE2_analysis/PCA_manual_clusters'

# reduction_type = 'UMAP'
# labels_filter = 'all'
# x = 'metric'
# y = '1_vs_2'
# plt_title = f'{y} RMSD of {reduction_type} reductions {x} - {labels_filter}'
# inner_plot_format = 'box'
# title_size = 9
# project_name = 'Mason'

# list of RMSD violins = [sheet, reduction_type, labels_filter, x_axes, y_axes]

# a = ['multi_cluster_rmsd', 'UMAP', 'all', 'label', 'avg_rmsd']
# b = ['multi_cluster_rmsd', 'UMAP', 'all', 'label', '1_vs_2']
# c = ['multi_cluster_rmsd', 'UMAP', 'all', 'metric', 'avg_rmsd']
# d = ['multi_cluster_rmsd', 'UMAP', 'Binder', 'metric', 'avg_rmsd']
# e = ['multi_cluster_rmsd', 'UMAP', 'Non Binder', 'metric', 'avg_rmsd']
# f =['multi_cluster_rmsd', 'UMAP', 'all', 'metric', '1_vs_2']
# g =['multi_cluster_rmsd', 'UMAP', 'Binder', 'metric', '1_vs_2']
# h =['multi_cluster_rmsd', 'UMAP', 'Non Binder', 'metric', '1_vs_2']

# i =['multi_cluster_rmsd', 'PCA', 'all', 'label', 'avg_rmsd']
# j =['multi_cluster_rmsd', 'PCA', 'all', 'n_SPACE2_clusters', 'avg_rmsd']
# k =['multi_cluster_rmsd', 'PCA', 'all', 'label', '1_vs_2']
# l =['multi_cluster_rmsd', 'PCA', 'Binder', 'n_SPACE2_clusters', 'avg_rmsd']
# m =['multi_cluster_rmsd', 'PCA', 'Non Binder', 'n_SPACE2_clusters', 'avg_rmsd']


# n = ['all_summaries', 'UMAP', 'all', 'label', 'n_abs%']
# o = ['all_summaries', 'UMAP', 'all', 'metric', 'n_abs%']
# p = ['all_summaries', 'UMAP', 'Binder', 'metric', 'n_abs%']
# q = ['all_summaries', 'UMAP', 'Non Binder', 'metric', 'n_abs%']

# r = ['all_summaries', 'PCA', 'all', 'label', 'n_abs%']
# s = ['all_summaries', 'PCA', 'all', 'component_1', 'n_abs%']
# t = ['all_summaries', 'PCA', 'Binder', 'component_1', 'n_abs%']
# u = ['all_summaries', 'PCA', 'Non Binder', 'component_1', 'n_abs%']
# v = ['all_summaries', 'PCA', 'all', 'component_2', 'n_abs%']
# w = ['all_summaries', 'PCA', 'Binder', 'component_2', 'n_abs%']
# x = ['all_summaries', 'PCA', 'Non Binder', 'component_2', 'n_abs%']

# rmsd_dict = {'UMAP': [a,b,c,d,e,f,g,h], 'PCA': [i,j,k,l,m]}
# n_abs_dict = {'UMAP': [n, o, p, q], 'PCA': [r, s, t, u, v, w, x]}


# for key, items in rmsd_dict.items():
#     all_summaries_file = umap_all_summaries_file if key == 'UMAP' else pca_all_summaries_file

#     input_df = pd.read_excel(all_summaries_file, sheet_name='multi_cluster_rmsds')
#     save_path = umap_save_path if key == 'UMAP' else pca_save_path

#     for sheet, reduction_type, labels_filter, x, y in items:
#         ext = f'{labels_filter}s only' if labels_filter != 'all' else 'all points'

#         averages = True if y == 'avg_rmsd' else False

#         plt_title = f'Average RMSD of {reduction_type} Clusters Containing Multiple SPACE2 Clusters\n{x} - {ext}' if y == True else f'RMSD of {y[0]} and {y[-1]} Largest SPACE2 Clusters Contained in Single {reduction_type} Cluster \n {x} - {ext}'

#         file_name = f'{project_name}_{reduction_type}_RMSD-{y}_{x}_{labels_filter}.png'

#         data = input_df[input_df['label'] == labels_filter] if labels_filter != 'all' else input_df
#         labels_filter

#         if reduction_type != 'PCA':
#             data['metric'] = data['reduction_file'].apply(get_metric)

#         summaries_violin_plot(x, y, data, plt_title, file_name, title_size, save_path, inner_plot=inner_plot_format)



# for key, items in n_abs_dict.items():
#     all_summaries_file = umap_all_summaries_file if key == 'UMAP' else pca_all_summaries_file

#     input_df = pd.read_excel(all_summaries_file, sheet_name='all_summaries')
#     save_path = umap_save_path if key == 'UMAP' else pca_save_path

#     for sheet, reduction_type, labels_filter, x, y in items:

#         ext = f'{labels_filter}s only' if labels_filter != 'all' else 'all points'

#         plt_title = f'Percentages of {reduction_type} Cluster Contained Within Associated SPACE2 Clusters Separated\n{x}' if labels_filter == 'all' else f'Percentages of {reduction_type} Cluster Contained Within Associated SPACE2 Clusters Separated\n{x} - {ext}'

#         file_name = f'{project_name}_{reduction_type}_nAbsPC-{y}_{x}_{labels_filter}.png'

#         data = input_df[input_df['label'] == labels_filter] if labels_filter != 'all' else input_df
#         labels_filter

#         if reduction_type != 'PCA':
#             data['metric'] = data['reduction_file'].apply(get_metric)

#         summaries_violin_plot(x, y, data, plt_title, file_name, title_size, save_path, inner_plot=inner_plot_format)



