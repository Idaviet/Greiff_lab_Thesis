#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import argparse


# # Object Class Definitions

# ### Define PDB_Atom object class

# In[2]:


class PDB_Atom:
    '''
    Creates new class of object based on the atomic entry of a .pdb file.
    
    ATTRIBUTES:
        - atom # in file (.atom_id)
        - residue position in file (.res_position)
        - position within the CDRH3 (.cdrh3_pos)
        - associated residue type (.res_type)
        - full CDRH3 sequence associated with the atom (.cdrh3)
        - x/y/z coordinates (in separate attributes, .x/.y/.z)
        - chain it is located in (.chain)
        - element of the atom (.element)
        - calculated occupancy (.occupancy)
        - temperature factor (.temp_factor)
        - Summary list containing position in CDRH3, atom #, amino acid type, element and x/y/z coordinates (.element_coord)    
    '''
    
    def __init__(self, atom_number, amino_acid_position, pos_in_cdrh3, amino_acid_type, x, y, z, chain, element, occupancy, temp_factor, CDRH3):
        self.atom_id = atom_number #gives atom number as it appears in the pdb file
        self.res_position = amino_acid_position #gives the residue position ID as it appears in the PDB file
        self.cdrh3_pos = pos_in_cdrh3 #Gives numerical position of atom/residue in CDRH3, from 1-10
        self.res_type = amino_acid_type #gives amino acid associated with this atom
        self.cdrh3 = CDRH3 #Gives full CDRH3 sequence associated with this atom
        self.x = x #x-coordinate of the atom
        self.y = y #y-coordinate of the atom
        self.z = z #z-coordinate of the atom
        self.chain = chain #Gives chain the atom is a part of
        self.element = element #gives element of atom
        self.occupancy = occupancy #occupancy as it appears in pdb file
        self.temp_factor = temp_factor #temperature factor as it appears in pdb file
        self.element_coord = [pos_in_cdrh3, atom_number, amino_acid_type, element, x, y, z] #all the important structural information (at least for now)
    
    def __str__(self):
        return (
            f"CDRH3: {self.cdrh3} Position #{self.cdrh3_pos}- Atom #{self.atom_id} - {self.res_type}{self.res_position} "
            f"({self.x}, {self.y}, {self.z}) Chain: {self.chain} Element: {self.element} "
            f"Occupancy: {self.occupancy} Temp Factor: {self.temp_factor}"
        )


# ### Define PDB_CDRH3 object class

# In[18]:


class PDB_CDRH3:
    '''
    Compiles all atoms (of class PDB_atoms) of a given CDRH3 from pdb file. The data of the individual atoms can 
    be accessed using the attributes of PDB_Atom objects (see above)
    
    ATTRIBUTES:
     - .full_atom_list: returns full PDB_atom list used as input when object is created
     - .atoms_number: returns total number of atoms in the object
     - .atoms_range: returns the start and end of the CDRH3's atoms within the original pdb file
     - .binding: returns the binding as either 0 (non binder), 1 (binder) or 'Unknown' (if the labeling data has 
                 not been added yet; see binding_and_qc function fo more details)
     - .sequence: returns sequence of CDRH3 as 1-letter AA code
     - .atoms_coord: returns list containing all .element_coord summaries of the PDB_atom objects of CDRH3
    
    '''
    
    def __init__(self, pdb_atoms_list, binding = None):
        self.full_atom_list = pdb_atoms_list #returns full list of PDB_Atom objects
        self.atoms_number = len(self.full_atom_list) #returns numbmer of atoms inf given CDRH3 
        self.atoms_range = (pdb_atoms_list[0].atom_id, pdb_atoms_list[-1].atom_id) #returns positions of CDRH3 atoms in pdb file
        self.binding = 'Unknown' if binding is None else binding
        self.sequence = pdb_atoms_list[0].cdrh3
        
        #Resulting List: [Atom #, Position in CDRH3, Associated Residue, Element, X, Y, Z coordinates]
        self.atoms_coord = [atom.element_coord for atom in pdb_atoms_list] 
        
        
    def __str__(self):
        return (
            f"Sequence: {self.sequence} -- {self.atoms_number} atoms at positions {self.atoms_range} -- Binding: {self.binding} "
        )
        
# test = PDB_CDRH3(pdb_atoms)
# test.atoms_range


# # Function Definitions

# ### Function to convert strings into floats when possible

# In[3]:


def convert_to_float(value):
    '''
    converts strings into floats when possible, leaving as string if not
    '''
    try:
        return float(value)
    except ValueError:
        return value
    


# ### Function to convert residue into alphabetic code

# In[4]:


def three_letter_to_one_letter(three_letter_code):
    '''
    Quick converter to switch from 3 to 1 letter amino acid code. Only takes one amino acid at a time
    '''
    aa_mapping = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
 


    return aa_mapping.get(three_letter_code, three_letter_code)


# ### Function to extract all atom entries + CDRH3 sequence/positions from PDB file

# In[5]:


def parse_pdb(file_path):
    """
    Parses PDB file to extract CDRH3 sequences and positions as well as structural information for ALL atoms contained
    therein. 
    
    CDRH3 sequence is returned as a single string using 1 letter AA code
    CDRH3 positions are returned as a list of strings. Assumes that the 98th unique residue position is the start
            of the sequence and the 108th unique position is the end
    Atoms list contains all data points for a given atom, saved as a list of strings
    """
    atoms_lines = []

    positions = []

    aa_seq = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith ("ATOM") or line.startswith("HETATM"):
                atoms_lines.append(line)
                
                line_list = line.split(' ')
                line_list = [item for item in line_list if item != '']
                line_list = [item for item in line_list if item !='\n']
                
                aa_pos = line_list[5]
                aa = line_list[3]

                if aa_pos not in positions:
                        aa = three_letter_to_one_letter(aa)
                        aa_seq.append(aa)
                        positions.append(aa_pos)

        cdrh3, cdrh3_pos = ''.join(aa_seq[98:108]), positions[98:108]
        
        return cdrh3, cdrh3_pos, atoms_lines

### TESTING

# file_path = r"C:\Users\isaac\Desktop\MTLS Masters Program\Thesis\Brijj-Mason_data\test\mHER_H3_AgNeg_unique_fv_12_igfold.pdb"

# parsed_pdb_file = parse_pdb(file_path) 
# cdrh3_sequence = parsed_pdb_file[0]
# #cdrh3_position_ids = parsed_pdb_file[1]
# all_atoms = parsed_pdb_file[2]
# print(cdrh3_sequence)


# ### Function to extract atom information of CDRH3 and store in PDB_Atom class object

# In[6]:


def find_cdrh3_atoms(parsed_pdb_file):
    '''
    Extracts the information for each atom in the PDB file using the outputs of the parse_pbd function
    
    Input: output from parse_pdb function 
    ['<CDRH3 sequence>', [<positions of CCDRH3 in PDB file>], [<list of all atom entries as a list>] )
    
    Output: list containing all atoms in CDRH3, stored as objects of PDB_Atom type
    '''
    
    cdrh3 = parsed_pdb_file[0]
    cdrh3_pos = parsed_pdb_file[1]
    atoms = parsed_pdb_file[2]
    
    pdbatoms = []
    hold = []

    for atom in atoms:
        chain = atom[4]
        aa_pos = atom[5]
        aa = atom[3]


        if aa_pos in cdrh3_pos and chain == 'H':
            atom_id = atom[1]
            atom_type = atom[2]
            aa = three_letter_to_one_letter(atom[3])
            x = atom[6]
            y = atom[7]
            z = atom[8]
            occ = atom[9]
            temp = atom[10]
            element = atom[11]

            if aa_pos not in hold: #finds actual position in CDRH3 from 1-10
                hold.append(aa_pos)
                pos_in_cdrh3 = len(hold)

            entry = PDB_Atom(atom_id, aa_pos, pos_in_cdrh3, aa, x, y, z, chain, element, occ, temp, cdrh3)
            pdbatoms.append(entry)
            
    cdrh3_object = PDB_CDRH3(pdbatoms)
    
    return cdrh3_object

###TESTING
# cdrh3_object = find_cdrh3_atoms(parsed_pdb_file)
# print(cdrh3_object, '\n\n')
# for atom in cdrh3_object.full_atom_list:
#     print(atom)


# ### Full parsing function

# In[7]:


def full_pdb_parsing(file_path):
    '''
    Combines previous functions to fully parse all atoms in a given pdb file and return the CDRH3 structural data
    as a PDB_CDRH3 type object.
    
    See object class definitions above for resulting attributes
    '''
    parsed_pdb_file = parse_pdb(file_path)
    cdrh3_object = find_cdrh3_atoms(parsed_pdb_file)
    return cdrh3_object

### TESTING
# file_path = r"C:\Users\isaac\Desktop\MTLS Masters Program\Thesis\Brijj-Mason_data\test\mHER_H3_AgNeg_unique_fv_12_igfold.pdb"

# cdrh3_object = full_pdb_parsing(file_path)
# print(cdrh3_object.atoms_range)
# print(cdrh3_object.full_atom_list[0].res_type)


# ### Function to extract all CDRH3 structural information from a folder

# In[8]:


def parse_all_pdb_files(folder):
    '''
    Uses previous functions to parse all pdb files from a given folder and extract the CDRH3 structural data into
    PDB_Atom and PDB_CDRH3 class objects. 
    
    Returns the data as a PDB_CDRH3 type object, one for each file parsed 
    '''
    
    cdrh3_all = []
    all_sequences = []

    for filename in os.listdir(folder):
        if filename.endswith('.pdb'):
            file_path = os.path.join(folder, filename)
            cdrh3_object = full_pdb_parsing(file_path)
            cdrh3_all.append(cdrh3_object)
            
            if cdrh3_object.sequence not in all_sequences:
                all_sequences.append(cdrh3_object.sequence)
            
    return cdrh3_all, all_sequences

### TESTING
# cdrh3_all, all_sequences = parse_all_pdb_files(folder)

# print(cdrh3_all[0].atoms_range)
# print(cdrh3_all[0].sequence)
# print(all_sequences)

# test_sequences = all_sequences[:-3] #testing set for the next section


# ### Function to extract selected CDRH3 sequences from folder of pdb files

# In[9]:


def parse_selected_cdrh3(folder, cdrh3_list = None):
    '''
    Uses previous functions to parse pdb files from a given folder containing the CDRH3 sequences specified in a given 
    list and extract the CDRH3 structural data into PDB_Atom and PDB_CDRH3 class objects. 
    
    NOTE: input list of CDRH3 must be a sequence in the 1-letter code. Output will also be in this format 
    
    Returns the data as a PDB_CDRH3 type object, one for each file parsed, and a list of all the CDRH3 sequences 
    extracted
    '''
    
    cdrh3_selected = []
    selected_sequences = []

    for filename in os.listdir(folder):
        if filename.endswith('.pdb'):
            file_path = os.path.join(folder, filename)
            cdrh3_object = full_pdb_parsing(file_path)
            
            if cdrh3_object.sequence in cdrh3_list:
                cdrh3_selected.append(cdrh3_object)

                if cdrh3_object.sequence not in selected_sequences:
                    selected_sequences.append(cdrh3_object.sequence)
            
    return cdrh3_selected, selected_sequences

### TESTING
# cdrh3_selected, selected_sequences = parse_selected_cdrh3(folder, test_sequences)

# print(cdrh3_selected[0].atoms_range)
# print(cdrh3_selected[0].sequence)
# print(cdrh3_selected[0].binding)
# print(selected_sequences)


# ### Function to add binding data to PDB_CDRH3 objects

# In[10]:


def binding_and_qc(pdb_cdrh3_list, binding_file):
    '''
    Parses the starting labeled sequence csv file to add missing binding attribute to PDB_CDRH3 objects in a given list
    
    Also performs a simple QC to make sure that the sequences in the PDB_CDRH3 list match the original labeled 
    sequences of the labeled file
    '''
    
    missing_seqs = []
    with open(binding_file, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        
        cdrh3_index = 0
        binding_index = 1
        cdrh3_binding_dict = {row[cdrh3_index]: row[binding_index] for row in csv_reader}

        for object in pdb_cdrh3_list:
            seq = object.sequence
            binding_value = cdrh3_binding_dict.get(seq, 'not found')
            
            if binding_value == 'not found':
                missing_seqs.append(seq)
            else:
                object.binding = binding_value
    
    
    # if len(missing_seqs) > 0:
    #     print(f'{missing_seqs} could not be found in binding file\nAll {len(pdb_cdrh3_list)} other CDRH3 objects were found and annotated.')
    # else:
    #     print(f'{len(pdb_cdrh3_list)} CDRH3 sequences found. Binding data successfully added\n')




### TESTING        
# binding_and_qc(cdrh3_all, binding_file)
# for i in cdrh3_all:
#     print(i.binding)

### TEST CDRH3 SEQUENCES TAKEN FROM PCA EXTRACTION PYTHON FILE ('pca.ipnyb')
#vert6_cdrh3s= ['WSGAGFYEFA', 'WGSRGFYEFA', 'WSNSSLYEFP', 'WLGDGFYEFA', 'YLPFSFYEFI', 'WRGAGMYEFT', 'FGSSAFFEFR', 'YDVGGLYEFA', 'WRGAGFYEFA', 'WLSGSMYEFA', 'YRVGSHYEFR', 'YDKTSFYEFN', 'FQKCALFEFS', 'YGAGGMYEFL', 'YPAFGMYEFQ', 'YNKSALFEFS', 'WKRAAMFEFS', 'YPEGGMYEFR', 'WINNSMYEFL', 'YSDNGFYEFA', 'FLVRALFEFG', 'WVPSGFYEFK', 'YPLSSLYEFA', 'YGQISHYEFQ', 'WQGPSFYEFV', 'WGDAGYFEFT', 'FTLCAFFEFP', 'FNISSFYEFN', 'YENCSFYEFL', 'YRGCSFYEFT', 'YVENGLFEFS', 'FDNPSHFEFP', 'FRPFSMYEFA', 'FRMNSFYEFP', 'YTDCSLYEFK', 'FDGGGFYEFP']


# # Final Interactive Function
# Final function to flexibly perform all analyses possible. Arguments can be input directly when function is called or after function is called blank

# In[11]:


def interactive_function(analysis_type = None, file_or_folder_path = None, labels_file = None, cdrh3_seq_list = None):
    """
    Perform various analyses on PDB files based on user input.

    Parameters:
    - analysis_type (str): Type of analysis to perform. 
                          Options: 'single' or '1' for analysis of single pdb file, 
                                    'full' or '2' for analysis of all pdb files in a given folder, 
                                    'selection' or '3' for analysis pdb files in folder containing 
                                        only list of selected CDRH3 sequences.
    - file_or_folder_path (str): Path of the file or folder to be parsed.
    - labels_file (str): Path of the file containing labeled sequences.
                        Optional; leave blank if adding labels/QC not wanted.
    - cdrh3_seq_list (list): List of 1-letter amino acid sequences for CDRH3 analysis options.

    Returns:
    - result (list): List of results from the analysis contained in PDB_CDRH3 and PDB_Atoms type objects.

    If labels_file is not blank, binding data will be added to the analysis results.
    If an error occurs (e.g., incorrect file format for labels_file), an error message will be printed.

    The results of the analysis will be printed to the console.
    """
    
    analysis_type = input('\nSelect analysis type:\n\
    [1] - Single PDB file parsing\n\
    [2] - Parsing all PDB files in a folder\n\
    [3] - Parsing selection of CDRH3 sequences in a specific folder\n\n') if analysis_type is None else analysis_type
        
    file_or_folder_path = str(input('\nInput path of file or folder to be parsed:\n')) if file_or_folder_path is None else str(file_or_folder_path)
    
    labels_file = str(input('\nInput path of file containing labeled sequences.(Optional; leave blank if adding labels/QC not wanted):\n')) if labels_file is None else str(labels_file)


    if analysis_type == 'single' or analysis_type == '1' or analysis_type == 1:
        result = [full_pdb_parsing(file_or_folder_path)]

    if analysis_type == 'full' or analysis_type == '2' or analysis_type == 2 :
        result = parse_all_pdb_files(file_or_folder_path)[0]

    if analysis_type == 'selection' or analysis_type == '3' or analysis_type == 3:
        cdrh3s = str(input('Input CDRH3 sequences as a list of 1-letter amino acid sequences:\n')) if cdrh3_seq_list is None else None
        cdrh3_seq_list = [item.strip() for item in cdrh3s.split(',')] if cdrh3s is not None else cdrh3_seq_list
        result = parse_selected_cdrh3(file_or_folder_path, cdrh3_seq_list)[0]
    
    error = None
    if labels_file != '':
        if '.csv' in labels_file:
            binding_and_qc(result, labels_file)
        elif labels_file is not None:
            error = '\n\n\nPlease load a .csv file in the proper format to add binding data'
    
    print('\n\n\nResults of Analysis:\n\n')
    for i in result:
        print(i)

    if error is not None:
        print(error)     
    
    if len(result) == 1:
        result = result[0]

    return result



# # Practical applications

# ##### Example files and variables used in testing
# Change to desired values before using code in next section

# In[16]:


pdb_file = r"C:\Users\isaac\Desktop\MTLS Masters Program\Thesis\Brijj-Mason_data\test\mHER_H3_AgNeg_unique_fv_12_igfold.pdb"
folder = r"C:\Users\isaac\Desktop\MTLS Masters Program\Thesis\Brijj-Mason_data\test"
labeled_file = r"C:\Users\isaac\Desktop\MTLS Masters Program\Thesis\Brijj-Mason_data\mason_sequences_label.csv"
CDRH3_sequences = ['YPLDGLFLFD', 'FNATAMYAFT', 'YGVNSSYVLN']


# ##### Demonstration of how the final interactive_function can be used within this notebook.
# Uncomment to use and adapt as necessary

# In[23]:


#Single File Parsing: Specificy which individual pdb file with pdb files to be analyzed as well as the labeled sequences csv file distinguishing binders from non-binders 
# cdrh3_object = interactive_function(1, pdb_file, labeled_file) 

# # Parse All PDB files in Folder: Specify folder containing pdb files, labeled sequences csv file
# cdrh3_object_list = interactive_function(2, folder, labeled_file)

# # Parse PDB files containing selected list of CDRH3 sequences: Specify folder containing pdb files, labeled sequences csv file and list of CDRH3 1-letter amino acid sequences.
# cdrh3_object_list = interactive_function(3, folder, labeled_file, CDRH3_sequences)

# # Call parsing function and input arguments through terminal
# cdrh3_interactive = interactive_function()


# ##### Argparse function, callable from terminal ONLY in .py version of this code
# Keep commented in .ipynb file so that the system can run smoothly

# In[14]:


###Argparse function to parse pdb files from terminal. Commentted in this file as a python notebook can't properly execute it, however it is callable in the .py version of this code 

# if __name__ == "__main__":
#     """
#     Command-line interface for the extraction of CDRH3 structural information from a Fv model PDB file.
#     Assumes the CDRH3 sequence of interest is at positions 99-109 of the heavy chain.

#     Usage:
#     python script_name.py --analysis_type TYPE --file_or_folder PATH --labeled_file PATH --cdrh3_seq_list LIST
    
#     NOTE: It is not necessary to specify all arguments in the command line. The interactive function used contains 
#         input prompts for any missing arguments

#     Arguments:
#     - TYPE (str): Analysis type. Options: 'single' or '1', 'full' or '2', 'selection' or '3'.
#     - PATH (str): Path of the file or folder to be parsed.
#     - PATH (str): Path of the file containing labeled sequences. Optional; leave blank if adding labels/QC not wanted.
#     - LIST (list): List of 1-letter amino acid sequences for CDRH3 analysis. Separate entries by comma

#     Example:
#     python script_name.py --analysis_type single --file_or_folder path/to/file --labeled_file path/to/labels.csv --cdrh3_seq_list ACGT

#     The script uses argparse to parse command-line arguments and calls the interactive_function with the provided inputs.
#     """

#     parser = argparse.ArgumentParser(description='Allows the extraction of CDRH3 structural information\
#                                       from a Fv model pdb file, assuming the sequence of interest is at\
#                                       positons 99-109 of the heavy chain')
    
#     parser.add_argument('--analysis_type', type = str, help = 'Analysis type:\n\
#                                                                 [1 or "single"] - Single PDB file parsing\n\
#                                                                 [2 or "full"] - Parsing all PDB files in a folder\n\
#                                                                 [3 or "selection"] - Parsing selection of CDRH3 sequences\
#                                                                  in a specific folder\n\n')
    
#     parser.add_argument('--file_or_folder', type = str, help = 'Path of file or folder to be parsed:')
    
#     parser.add_argument('--labeled_file', type = str, help = 'Path of file containing labeled sequences.(Optional\
#                                                             leave blank if adding labels/QC not wanted)')
    
#     parser.add_argument('--cdrh3_seq_list', type = list, help = ' CDRH3 sequences as a list of 1-letter amino acid sequences. Separate by comma')

#     args = parser.parse_args()
    
#     interactive_function(args.analysis_type, args.file_or_folder, args.labeled_file, args.cdrh3_seq_list) 


