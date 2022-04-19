'''
    Create Code Dictionary from Mimic 4 files
    Create list of Admission records grouped by patient id
    and admission id including all codes
    {
        "<patient_id>": [
            "diagnosis": [code_indexes],
            "procedures": [code_indexes],
            "medications": [code_indexes]
        ]
    }

'''

import pandas as pd
import os
import json

mimic_base_path = 'data/mimic-iv-1.0/'
inputs_base_path = 'data/input'

PROC_FILE = os.path.join(mimic_base_path, 'hosp/procedures_icd.csv')        # ICD codes per
MED_FILE = os.path.join(mimic_base_path, 'hosp/prescriptions.csv')          # NDC codes per admission/patient
DIAG_FILE = os.path.join(mimic_base_path, 'hosp/diagnoses_icd.csv')         # ICD codes per admission/patient
ICU_FILE = os.path.join(mimic_base_path, 'icu/icustays.csv')

ndc2atc_file = os.path.join(inputs_base_path, 'ndc2atc_level4.csv' )
cid_atc = os.path.join(inputs_base_path, 'drug-atc.csv')
ndc2rxnorm_file = os.path.join(inputs_base_path, 'ndc2rxnorm_mapping.txt')
DDI_FILE = os.path.join(inputs_base_path, 'drug-DDI.csv')

icd9_2_10 = json.loads(open(os.path.join(inputs_base_path, 'icd9_2_10.json'), 'r').read())
cardiac_codes_10 = json.loads(open(os.path.join(inputs_base_path, 'cardiac_codes_10.json'), 'r').read())


def process_procedure():
    """
    Preprocess procedure file to drop extra columns, remove duplicates and sort
    by Subject, Admission and sequence.
    """
    pro_pd = pd.read_csv(PROC_FILE, dtype={'icd9_code':'category'})
    pro_pd.columns = pro_pd.columns.str.upper()
    DROP_COLS = ['SEQ_NUM']
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=DROP_COLS, inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)
    return pro_pd


def process_med():
    """
    Function to clean and preprocess medications
    1. Drops extra columns and removes duplicates
    2. Sorts by subject ID, admission ID and time
    3. Finds medications prescribed in first 60 days of admission
    """
    med_pd = pd.read_csv(MED_FILE, dtype={'ndc': 'category'})
    med_pd.columns = med_pd.columns.str.upper()
    DROP_COLS = ['DRUG_TYPE', 'GSN', 'PROD_STRENGTH', 'DOSE_VAL_RX', \
                 'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', \
                 'FORM_UNIT_DISP', 'ROUTE', 'DRUG', 'FORM_RX', 'DOSES_PER_24_HRS', 'STOPTIME']

    med_pd.drop(columns=DROP_COLS, axis=1, inplace=True)
    med_pd.drop(index=med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['STARTTIME'] = pd.to_datetime(med_pd['STARTTIME'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'STARTTIME'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    # Finding medications in 60 days
    def filter_first60_days(medications, FILE_ICU):
        """
        Filtering for the first 60 days of admission
        """
        icustays = pd.read_csv(ICU_FILE)
        icustays.columns = icustays.columns.str.upper()
        ICU_COLS = ['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME']
        MERGE_COLS = ['SUBJECT_ID', 'HADM_ID']
        TIME_COLS = ['INTIME', 'OUTTIME']
        icustays = icustays[ICU_COLS]
        merged = pd.merge(medications, icustays, on=MERGE_COLS, how='left')
        merged[TIME_COLS] = merged[TIME_COLS].apply(pd.to_datetime)
        merged['DIFF'] = (merged['OUTTIME'] - merged['INTIME']).dt.days
        merged = merged[merged['DIFF'] <= 60]
        DROP_COLS = ['INTIME', 'OUTTIME', 'DIFF', 'STARTTIME']
        merged.drop(columns=DROP_COLS, inplace=True)
        return merged

    med_pd = filter_first60_days(med_pd, ICU_FILE)

    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    # Finding where patients visits were more than 2
    def vists_greater_than_2(med_pd):
        """
        Find where the number of patient visits is more than 2
        """
        v = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
        v['HADM_ID_Len'] = v['HADM_ID'].map(lambda x: len(x))
        v = v[v['HADM_ID_Len'] > 1]
        return v

    med_pd_greater_2 = vists_greater_than_2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(med_pd_greater_2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')

    # Drop extra columns and duplicates
    med_pd.drop(columns=['PHARMACY_ID'], inplace=True)
    med_pd.drop_duplicates(inplace=True)
    return med_pd.reset_index(drop=True)


def process_diag():
    """
    Preprocess diagnoses
    1. Droping nulls and duplicates
    2. Sorting
    3. Get cardiac diagnosses
    """
    diag_pd = pd.read_csv(DIAG_FILE)
    diag_pd.columns = diag_pd.columns.str.upper()
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM'], inplace=True)  # remove row id
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)

    def process_codes(code_string):
        return code_string.split(' ')[0]

    def get_cardiac_diag(df_diag):
        """
        Getting diagnoses that are cardiac  related
        """
        df_diag['ICD_CODE'] = df_diag['ICD_CODE'].apply(process_codes)
        cardiac_9 = list(icd9_2_10.keys())
        cardiac_10 = list(icd9_2_10.values())
        codes_in_9 = (df_diag.ICD_CODE.isin(cardiac_9))
        codes_in_10 = (df_diag.ICD_CODE.isin(cardiac_9))
        df_diag = df_diag[(codes_in_10) | (codes_in_9)]
        return df_diag

    diag_pd['ICD_CODE'] = diag_pd['ICD_CODE'].apply(process_codes)

    return diag_pd.reset_index(drop=True)


def ndc2atc(med_pd):
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    """
    Mapping from one type of medication to another
    1. Mapping to ATC from NDC
    2. Drop extra columns
    3. Finding ATC level 3 code
    """
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)

    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4': 'NDC'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])  # atc level 4 code
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def combine_codes():
    proc = process_procedure()
    med_pd = process_med()
    med_pd = ndc2atc(med_pd)
    diag_pd = process_diag()

    med_diag = med_pd.merge(diag_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    card_codes = list(set(list(cardiac_codes_10.keys()))) + list(set(list(icd9_2_10.keys())))
    cond_1 = med_diag.ICD_CODE.isin(card_codes)
    cond_2 = med_diag.ICD_CODE.str.startswith(tuple(card_codes))
    mdc = med_diag[(cond_1) | (cond_2)]
    combined_key = mdc[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    # Merging combined key on all medications/ diagnoses/ procedures
    meds_new = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    diag_new = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    proc_new = proc.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    def combine_icd_codes(version, code):
        return str(version) + '.' + str(code)

    # Applying code changes to diagnosis column to create new diagnosis column and dropping old columns
    diag_new['CODE_ICD'] = diag_new.apply(lambda x: combine_icd_codes(x['ICD_VERSION'], x['ICD_CODE']), axis=1)
    diag_new.drop(columns=['ICD_CODE', 'ICD_VERSION'], inplace=True)

    # Grouping by hospital admissions and patient so that each row is an admission
    diag_new = diag_new.groupby(by=['SUBJECT_ID', 'HADM_ID'])['CODE_ICD'].unique().reset_index()

    # Grouping by hospital admissions and patient so that each row is an admission
    meds_new = meds_new.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()

    # Grouping by hospital admissions and patient so that each row is an admission
    proc_new = proc_new.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD_CODE'].unique().reset_index()\
                    .rename(columns={'ICD_CODE':'PRO_CODE'})

    # Merging and combining them into a single dataset
    combined = meds_new.merge(diag_new, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined = combined.merge(proc_new, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    return combined
