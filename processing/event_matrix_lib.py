import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.notebook import tqdm

column_list = ['SUBJECT_ID', 'HADM_ID', 'PRO_CODE','CODE_ICD', 'NDC']
feature_list = ['PRO_CODE','CODE_ICD', 'NDC']


def gen_event_location_dict(all_codes):
    """
    Function to generate a single dictionary that will hold the indices of
    all events when occurrence matrix and M matrix holding the PMI scores
    is generated
    ----
    Input:
      1. Diagnosis dictionary (all unique diagnoses as keys)
      2. Procedure dictionary (all unique procedures as keys)
      3. Medicine dictionary (all unique medicines as keys)
    Output:
      1. A single dictionary, with keys as unique medical events and values
         as their indices, which will be the basis of occurrence matrix and
         PMI matrix
      2. The number of total unique medical events
    """

    return {k: v for v, k in enumerate(all_codes)}


def build_occurence_matrix(event_location_dict, c, data):
    """
    Function to build the occurrence matrix for all of the events in the data
    Input:
      1. Diagnosis dictionary (all unique diagnoses as keys)
      2. Procedure dictionary (all unique procedures as keys)
      3. Medicine dictionary (all unique medicines as keys)
      4. Dataframe that contains unique admission level data
        holding lists of diagnoses/ procedures/ medications
    Output:
      1. Occurrence matrix containing information of how many times two
         events (regardless of event type —— e.g. medicine / procedure)
         occurred together.
    """

    matrix_occ = np.zeros((c, c))
    for idx, row in data.iterrows():
        ev_list = []
        for col in feature_list:
            ev_list.extend(row[col])
        for item in ev_list:
            loc_item1 = event_location_dict[item]
            for item2 in ev_list:
                loc_item2 = event_location_dict[item2]
                matrix_occ[loc_item1, loc_item2] += 1
    return matrix_occ


def get_pmi_score(matrix_occ, event1_idx, event2_idx, len_D = 20000, p=False):
    """
    Get the PMI score (point-wise mutual information score) to calculate
    the relationship between two edges based on how frequently they are
    co-occuring.
        PMI(i,j) = log ( (d(i,j) * |D|) / (d(i) * d(j)) )
    Where
      d(i,j) = How many times i and j occur together
      d(i) = How many times i occurs
      d(j)= How many times j occurs
      |D| = The total number of admission records
    ----
    Input:
      1. Occurrence matrix containing information about how frequently
          events are occuring together from the aggregated patient data.
      2. Index of event 1 (corresponding to where the event 1 information is
          stored in the occurence matrix)
      3. Index of event 2 (corresponding to where the event 2 information is
          stored in the occurence matrix)
      4. Length of all events (corresponding to the sum of unique procedures,
          unique diagnoses, unique medications in the data, as well as the
          shape of the occurrence matrix in 1.)
    Output:
      A float denoting the (non-negative) PMI score
    """
    pmi_score = None
    d_i_j = matrix_occ[event1_idx, event2_idx]
    d_i = matrix_occ[event1_idx, event1_idx]  # sum(matrix_occ[event1_idx, :])

    d_j = matrix_occ[event2_idx, event2_idx]
    pmi_score = np.log((d_i_j * len_D) / (d_i * d_j))

    if d_i == 0 or d_j == 0:
        print("One event is zero, i: {}, j: {}".format(event1_idx, event2_idx))
    if pmi_score < 0:
        pmi_score = 0

    if p:
        print("d(i,j): {} \n".format(d_i_j))
        print("d(i): {} \n".format(d_i))
        print("d(j): {} \n".format(d_j))
        print("PMI score, before negative handle: {} \n".format(pmi_score))

    return pmi_score


def build_M_matrix(matrix_occ, data, event_location_dict, c):
    """
    Function to build the M matrix, that takes the occurence matrix
    and populates it with PMI score values (that are non-negative)
    -----
    Input:
      1. Occurrency matrix showing how many times events occurred
         together in data
      2. Data containing aggregated record of patient admissions (to
          iterate over and build the matrix)
    Output:
      A matrix that contains the PMI scores of relevant info.
    """

    M_matrix = np.zeros((c, c))

    for idx, row in tqdm(data.iterrows(), total=len(data), desc='Row'):
        ev_list = []

        for col in feature_list:
            ev_list.extend(row[col])

        for item in ev_list:
            loc_item1 = event_location_dict[item]
            for item2 in ev_list:
                loc_item2 = event_location_dict[item2]
                M_matrix[loc_item1, loc_item2] = get_pmi_score(matrix_occ, loc_item1, loc_item2, len(data))
    return M_matrix


def build_event_matrix(combined):
    '''
    Takes in list of pairs of patient events and constructs PMI matrix for all events
    '''
    # all_codes = list(set([c if isinstance(l, list) else l for f in feature_list for l in combined[f] for c in l]))
    all_codes = list(set([c for f in feature_list for l in combined[f] for c in l]))
    event_location_dict = gen_event_location_dict(all_codes)
    c = len(event_location_dict)
    m_occ = build_occurence_matrix(event_location_dict, c, combined)
    return build_M_matrix(m_occ, combined, event_location_dict, c), event_location_dict


# Add additional filtering criterion for procedures and diagnosis codes
def includes_cardiac(entry):
    return [any([code[0] == 'B' or code[0] == 'C' for code in codes]) for codes in entry]

def check_unique_drugs(combined):
    drugs = combined['NDC']
    unique_drugs = []
    for drug_entry in drugs:
        for drug in drug_entry:
            unique_drugs.append(drug)
    unique_drugs = set(unique_drugs)
    # TODO meds list
    print('test')


def build_save_event_matrix(combined_path, matrix_path, event_location_dict_path):
    '''
    Create and Save Event Matrix to File
    '''
    combined = pd.read_pickle(combined_path)
    # combined = combined(columns=column_list)
    # combined = combined.head(1000)
    filtered_combined = combined[includes_cardiac(combined['NDC'])]
    check_unique_drugs(filtered_combined)
    M_matrix, event_location_dict = build_event_matrix(filtered_combined)

    with open(matrix_path, 'wb') as file:
        pickle.dump(M_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(event_location_dict_path, 'wb') as file:
        pickle.dump(event_location_dict, file, protocol=pickle.HIGHEST_PROTOCOL)