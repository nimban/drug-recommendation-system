'''
Methods for Building Patient Dictionary with event admission history
Result used as input for GATE model
'''

import pickle5
import numpy as np
from tqdm.auto import tqdm
import torch
from tqdm.notebook import tqdm


feature_list = ['PRO_CODE','CODE_ICD', 'NDC']


def get_label_idxs(meds, med_location_dict):
    return [med_location_dict[med] for med in meds]


def build_dynamic_co_occurence_graph(
        patient_id, data, m_matrix, event_location_dict):
    """
    Building dynamic co occurrence graph for an individual patient
    according to a single admission.
    """

    list_timesteps = []
    A_dict_list = []
    # label_idxs = []
    data = data[data.SUBJECT_ID == patient_id]
    n = data.shape[0]
    # whole_idx = [i for i in range(m_matrix.shape[0])]

    admission_num = 0
    for idx, row in data.iterrows():
        admission_num += 1

        # if admission_num==n:
        #    ev_list = diag + proc
        # else:
        #    ev_list = ndc + diag + proc

        ev_list = []

        for col in feature_list:
            ev_list.append(row[col])
        ev_list = list(np.concatenate(ev_list).flat)

        # ev_idxs = [event_location_dict[ev] for ev in ev_list]
        evs = []
        for ev in ev_list:
            if ev in event_location_dict:
                evs.append(ev)
        ev_list = evs
        dynamic_single_step = np.zeros((len(ev_list), len(ev_list)))
        A_dict = {}
        for idx1, item in enumerate(ev_list):
            loc_item1 = event_location_dict[item]
            for idx2, item2 in enumerate(ev_list):
                loc_item2 = event_location_dict[item2]

                dynamic_single_step[idx1, idx2] = m_matrix[loc_item1, loc_item2]

                A_dict[item] = idx1

        list_timesteps.append(dynamic_single_step)
        A_dict_list.append(A_dict)

    return list_timesteps, A_dict_list


def make_patient_H0s(patient_list, recs_add, M_matrix, event_location_dict, labels_dict):
    '''
    creates a dictionary, key: patient_id, value: list of lists of lists, each outer list is a patient,
    first nested list is from the 0th up until the nth admission,
    second nested list is
    [co_occurence_matrix, code indices, co_occurence matrix without medications, indices without medications, ground truth label]
    for the ith admission
    '''
    data_dict = {}
    for patient_id in tqdm(patient_list):
        A, A_dict = build_dynamic_co_occurence_graph(patient_id, recs_add, M_matrix, event_location_dict)
        data = recs_add[recs_add.SUBJECT_ID == patient_id].reset_index()
        Ht_list = []
        for i, A_i in enumerate(A):     # Admissions
            ndc = data.iloc[i, :]['NDC']
            ev_list = []
            word_list = []
            for col in feature_list:
                ev_list += list(data.iloc[i, :][col])
            evs = []
            for ev in ev_list:
                if ev in event_location_dict:
                    evs.append(ev)
            ev_list = evs
            ev_idxs = [event_location_dict[ev] for ev in ev_list]
            # H0 = embeddings(torch.tensor(ev_idxs))

            A_i_test, test_ev_idxs, y = None, None, None
            if i > 0:
                test_ev_list = []
                test_word_list = []
                for col in feature_list[0:-1]:  # because we're excluding ndc
                    test_ev_list += list(data.iloc[i, :][col])
                evs = []
                for ev in test_ev_list:
                    if ev in event_location_dict:
                        evs.append(ev)
                test_ev_list = evs
                test_ev_idxs = torch.tensor([event_location_dict[ev] for ev in test_ev_list])
                A_i_test = torch.tensor(A_i[0:len(ev_list) - len(ndc), 0:len(ev_list) - len(ndc)])
                class_labels = get_label_idxs(ndc, labels_dict)
                y = torch.zeros(len(labels_dict))
                y[class_labels] = 1

            Ht_list.append([torch.tensor(A_i), torch.tensor(ev_idxs), A_i_test, test_ev_idxs, y])
        data_dict[patient_id] = Ht_list
    return data_dict


def create_save_patient_dict(recs_add_path, M_matrix_path, patient_dict_path, event_location_dict_path, meds_path):
    '''
    Read Recurring Admission data and event Matrix
    Construct patient Dictionary and Medications List
    and save to file
    '''
    with open(recs_add_path, 'rb') as file:
        recs_add = pickle5.load(file)

    with open(event_location_dict_path, 'rb') as file:
        event_location_dict = pickle5.load(file)

    with open(M_matrix_path, 'rb') as f:
        M_matrix = pickle5.load(f)

    with open(meds_path, 'rb') as f:
        med_labels = pickle5.load(f)

    patient_list = recs_add.SUBJECT_ID.unique()
    patient_dict = make_patient_H0s(patient_list, recs_add, M_matrix, event_location_dict, med_labels)

    with open(patient_dict_path, 'wb') as f:
        pickle5.dump(patient_dict, f, pickle5.HIGHEST_PROTOCOL)


def save_medication_labels(patient_codes_cleaned_path, meds_path):
    '''
    Create medications Labels Dictionary and save to file
    '''
    with open(patient_codes_cleaned_path, 'rb') as file:
        combined_codes = pickle5.load(file)

    med_labels = sorted(list(set([med for p in combined_codes['NDC'] for med in p])))
    labels_dict = {k: v for v, k in enumerate(med_labels)}

    with open(meds_path, 'wb') as f:
        pickle5.dump(labels_dict, f, pickle5.HIGHEST_PROTOCOL)