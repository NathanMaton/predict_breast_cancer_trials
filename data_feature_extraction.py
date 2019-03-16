import pandas as pd
from collections import Counter
from helper import load_data,pipeline, unique_drugs, generate_drugs_df
import re
import numpy as np
from fuzzywuzzy import fuzz, process

def remove_dup_trial():
    '''
    Split out trials with multiple drugs into individual rows
    '''
    df_temp = load_data()
    #c = unique_drugs(trials)
    #df_data, exceptions = generate_drugs_df(c, trials)
    df_temp = df_temp.reset_index()


    ## expanded unique drugs to get a dataframe we can group by
    dupes = []
    for idx, item_list in enumerate(df_temp["Primary Drugs"]):
        split_list = item_list.split(",")
        if len(split_list)>1:
            for i in split_list:
                df_temp.append(df_temp.iloc[idx,:])
                df_temp.iloc[-1,-3]=i
            dupes.append(idx)

    df_trials = df_temp.drop(dupes)

    # Removes white space from drug names
    df_trials["Primary Drugs"].apply(lambda x: x.strip())

    return df_trials

def drug_phase_df_template(df_trials):
    '''
    Creates a template dataframe that has drugs vs phases
    populates all cells with zeros
    shape [n drugs x 8 phases]

    Phases: ['Phase III', 'Phase II', 'Phase I', 'Phase I/II', 'Phase 0',
       'Phase IV', 'Phase II/III', 'Clinical Phase Unknown']

    '''

    drug_list = df_trials["Primary Drugs"].unique()
    phase_of_trial_list = df_trials["Phase of Trial"].unique()
    df_drug_phase_temp = pd.DataFrame(np.zeros([len(drug_list),len(phase_of_trial_list)]),columns=phase_of_trial_list)
    df_drug_phase_temp['Primary Drugs'] = drug_list
    df_drug_phase_temp = df_drug_phase_temp.set_index('Primary Drugs')

    return df_drug_phase_temp



def map_GB_drug_phase_to_df(df_trials,groupby_object):
    '''
    Maps a grouby object that has the drug, phase, and feature into a dataframe
    that accounts for each drug and each phase

    Input:

    groupby_object: expects [n drug x 3] shape where the columns are exactly
    [Primary Drugs, Phase of Trial, %FEATURE%]

    Output:
    df_drug_phase: A populated df with the groupby data
    [n drugs x phases ]

    '''
    # Generates a df template with drug x phase shape
    df_drug_phase = drug_phase_df_template(df_trials=df_trials)

    # Pulls index from df_drug_phase which has all the unique drug names
    drug_list = df_drug_phase.index.tolist()

    name_of_feature = [column for column in groupby_object.columns if ("Primary Drugs" not in column ) and ("Phase of Trial" not in column)][0]

    for drug in drug_list:
        for item in groupby_object[groupby_object["Primary Drugs"]==drug].iterrows():
            phase = list(item)[1]['Phase of Trial']
            count = list(item)[1][name_of_feature]
            df_drug_phase.loc[drug,phase] = count

    return df_drug_phase



def groupby_object_list(df_trials):
    '''
    User defined groupby objects for feature extraction

    Objects will be used as inputs to generate df_data features

    '''
    # Phase names
    phase_of_trial_list = df_trials["Phase of Trial"].unique()

    # Takes the number of subjects per trial per phase and calculates the mean
    groupby_patient_count = df_trials.groupby(["Primary Drugs","Phase of Trial"])["Planned Subject Number"].agg('mean').reset_index()
    name_subject_mean=[i+' subject mean' for i in phase_of_trial_list]


    # Takes the average difference of the start and stop time per drug per phase
    df_trials["time_diff"] = df_trials["trial_end"] - df_trials["trial_start"]
    df_trials["nano_time_diff"] = df_trials["time_diff"].astype(np.int64)
    #removes trials that have zero time delta
    zero_time_delta_mask = df_trials["time_diff"] > np.timedelta64(0)
    df_trials = df_trials[zero_time_delta_mask]
    groupby_time_deltas = pd.to_timedelta(df_trials.groupby(["Primary Drugs","Phase of Trial"])["nano_time_diff"].agg('mean')).reset_index()

    groupby_object_list = [groupby_patient_count,groupby_time_deltas]

    name_trial_length=[i+' trial length' for i in phase_of_trial_list]

    name_column_adjust_list= [name_subject_mean, name_trial_length]

    return groupby_object_list, name_column_adjust_list

def df_feature_extraction_by_phase(df_trials):

    df_data = pd.DataFrame()

    groupby_objects, name_column_adjust_list = groupby_object_list(df_trials=df_trials)

    for idx, groupby_object in enumerate(groupby_objects):

        # Adjust column names to phase - feature
        column_name_adj = name_column_adjust_list[idx]

        # Generate a dataframe with features
        df_temp = map_GB_drug_phase_to_df(df_trials=df_trials,groupby_object=groupby_object)
        df_temp.columns = column_name_adj
        # Combines the feature dataframes together
        df_data = pd.concat([df_data, df_temp],axis=1)

    return df_data

def feature_phase_pass_nopass(df_data):
    '''
    Add columns of feature that determine whether trial was successful
    Based on whether there were subjects in the following phase

    Example Phase I got a pass if Phase II has patients in them

    '''

    df_data['Phase I Pass'] = df_data['Phase II subject mean']!=0
    df_data['Phase II Pass'] = df_data['Phase III subject mean']!=0
    df_data['Phase I/II Pass'] = df_data['Phase III subject mean']!=0
    df_data['Phase III Pass'] = df_data['Phase IV subject mean']!=0
    df_data['Phase II/III Pass'] = df_data['Phase IV subject mean']!=0
    return df_data

def remove_org_dupes(x):
    drop_list = []
    #print(type(x))
    if type(x) != type('astring'):
        return []
    if x == np.nan:
        return []
    split_orgs = x.split(",")
    if len(split_orgs)>1:
        for idx1, i in enumerate(split_orgs[:-1]):
            for idx2, j in enumerate(split_orgs[idx1+1:]):
                #print(i,idx1, j ,idx2, fuzz.ratio(i,j))
                if fuzz.ratio(i,j) > 50:
                    drop_list.append(idx1+idx2+1)
        output = np.delete(split_orgs, list(set(drop_list)))
        return [item.strip() for item in output]
    return [item.strip() for item in [x]]

def feature_orangization_count(df_trials,df_data):
    '''
    Assumes df_data already exist with a table on the unique drug names in the index
    '''

    df_trials['Unique Organizations'] = df_trials['Organisations'].apply(remove_org_dupes)
    df_trials['Number of Organizations'] = df_trials['Unique Organizations'].apply(lambda x:len(x))

    # Grab a subset of the dataframe
    df_org_count = df_trials[['Primary Drugs','Number of Organizations']]
    groupby_orgs = df_org_count.groupby("Primary Drugs")["Number of Organizations"].agg('max').reset_index()



    df_data = df_data.merge(groupby_orgs, left_index=True, right_on="Primary Drugs", how='left')
    df_data.set_index('Primary Drugs', inplace=True)

    return df_data


if __name__ == '__main__':

    df_trials = remove_dup_trial()
    df_data = df_feature_extraction_by_phase(df_trials=df_trials)
    df_data = feature_phase_pass_nopass(df_data)
    df_data = feature_orangization_count(df_trials,df_data)


df_data.filter(regex="Phase II ")
