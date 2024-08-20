import csv
import numpy as np

class ABentry:
    """Creates object class for input sequences:
            number id = self.id
            amino acid sequence = self.sequence
            binder status = self.is_binder (original dataset used 0 for non and 1 for binder)
            expected length of sequence = self.expected_size (expected_size parameter must be set when object is created, original CDRH3 dataset had length of 10)
            QC (expected length matches actual sequence length, true/false) = self.is_good
    """
    def __init__(self, id, sequence, is_binder, expected_size=None):
        self.id = id
        self.sequence= sequence
        self.is_binder= is_binder

        self.size = len(sequence)
        self.expected_size = expected_size
        if len(sequence) == expected_size:
            self.is_good = True
        else:
            self.is_good = False


def expdata_QC(file_path, expected_seq_size = None):
    """
    QC of labeled amino acid dataset, as csv file containing sequence in column 1, binder status in column 2.
    Determines which sequences are not of the expected sequence length
    If expected seq size is not specified, will be set as the length of the first entry
    """
    #test file: "C:\Users\isaac\Desktop\MTLS Masters Program\Thesis\Brijj-Mason_data\brij_sequences_label.csv
    print('Analyzing...')
    file_path = file_path.replace('\\','/')
    ab_entries = {'Passed QC': [], 'Failed QC': []}
    good_entries = []
    bad_entries = []
    bad_ids = []

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next (csv_reader)

        id = 0

        for row in csv_reader:
            id += 1
            sequence, is_binder = row[0], row[1]
            if expected_seq_size is None and id == 1:
                expected_seq_size = len(sequence)

            entry = ABentry(id, sequence, is_binder, expected_seq_size)
            #print(entry.id, entry.size)

            if entry.is_good == True:
                good_entries.append(entry)
            elif entry.is_good == False:
                bad_entries.append(entry)
                bad_ids.append(id)

        ab_entries['Passed QC'] = good_entries
        ab_entries['Failed QC'] = bad_entries

        length_good = len(good_entries)
        length_bad = len(bad_entries)
        total_length = length_bad+length_good


        print(f'Total number of labeled seqs: {total_length}\n# of entries that pass: {length_good}\n# of entries that fail {length_bad}\n')
        if len(bad_ids) != 0:
            print(f'\nRow positions of failed entries:{bad_ids}')
    
            
        print(f'QC analysis of {file_path} complete\n\n')


    return ab_entries

def igdata_qc(file, length):
    """
    Integrated gradients QC of labeled sequences in .npy file format. Confirms that all IGs for all sequences are of appropriate length
    """
    bad_no = 0
    entry_id=0


    array_data = np.load(file)

    for ab_entry in array_data:
        entry_id += 1
        position_id = 0

        if len(ab_entry) != length:
            print(f'Fv Entry #{entry_id} is not the proper length')
            bad_no +=1

        for position in ab_entry:
            position_id += 1

            if len(position) != 20:
                print(f'Position #{position_id} of entry #{entry_id} is not the proper length')
                bad_no +=1

            ig_no = 0
            for aaig in position:
                if aaig > 0:
                    ig_no += 1
            
            if ig_no > 1:
                print(f'Position #{position_id} of entry #{entry_id} contains too many IG values')
                bad_no +=1

    if bad_no > 0:
        print(f'Analysis complete. Total number of bad entries found {bad_no}')
    elif bad_no == 0:
        print(f'Analysis complete. No bad entries found')
    print(f'QC analysis of {file} complete\n\n')



# Example dataset files
# brij_labeled_seqs = r"C:\Users\isaac\Desktop\MTLS Masters Program\Thesis\Brijj-Mason_data\brij_sequences_label.csv"
# mason_labeled_seqs = r"C:\Users\isaac\Desktop\MTLS Masters Program\Thesis\Brijj-Mason_data\mason_sequences_label.csv"
# mason_ig_npy = r"C:\Users\isaac\Desktop\MTLS Masters Program\Thesis\Brijj-Mason_data\masonIG.npy"
# brij_ig_npy = r"C:\Users\isaac\Desktop\MTLS Masters Program\Thesis\Brijj-Mason_data\brijIG.npy"

###Example usage, 
print(f'\nAnalyzing {mason_labeled_seqs} and {mason_ig_npy}\n\n')
expdata_QC(mason_labeled_seqs, 10)
igdata_qc(mason_ig_npy, 10)

print(f'\n\nAnalyzing {brij_labeled_seqs} and {brij_ig_npy}\n\n')
expdata_QC(brij_labeled_seqs, 10)
igdata_qc(brij_ig_npy, 10)





    