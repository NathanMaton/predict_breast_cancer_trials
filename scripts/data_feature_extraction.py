"""
This script takes our raw data and turns it into the features used in our models
and saves them as csvs.
"""

import pandas as pd
from collections import Counter
import re
import numpy as np
from fuzzywuzzy import fuzz, process  # we use this to find number of orgs.

## HELPER FUNCTIONS ##
def strip_planned_start(x):
    """Function designed to be used by Pandas.apply.
    Function takes a row and finds the start date and returns that.
    If the trial is planned vs. complete it returns 0.
    """
    date_str = re.findall("[\d\s\w]*", x["Trial Initiation date"])[0].strip()
    plan_str = re.findall("planned", x["Trial Initiation date"])
    # If you want to included planned trials
    # if (plan_str and not date_str):
    if plan_str:

        return 0
    else:
        return pd.to_datetime(date_str)


def strip_planned_end(x):
    """Function designed to be used by Pandas.apply.
    Function takes a row and finds the end date and returns that.
    If the trial is planned vs. complete it returns 0.
    """
    date_str = re.findall("[\d\s\w]*", x["Trial End date"])[0].strip()
    plan_str = re.findall("planned", x["Trial End date"])
    # If you want to included planned trials
    # if (plan_str and not date_str):

    if plan_str:
        return 0
    else:
        return pd.to_datetime(date_str)


def load_data():
    """This function takes the raw excel sheet provided by ADIS database
    removes any trials that have null values for primary drugs,
    creates new columns with cleaned up start and end dates for the trials,
    removes trials that are planned or have a completion date after the
    beginning of 2019 when this analysis was completed.
    """
    trials = pd.read_excel("data/bcdrugsct.xlsx")  # loads in data
    trials = trials[trials["Primary Drugs"].isna() == False]
    trials["trial_start"] = trials.apply(strip_planned_start, axis=1)
    trials["trial_end"] = trials.apply(strip_planned_end, axis=1)
    # this line uses !=0 because the cleaning functions we wrote returned 0
    # for unclean data.
    start_mask = trials["trial_start"] != 0
    end_mask = trials["trial_end"] != 0
    trials = trials[(start_mask) & (end_mask)]

    # Removes trials completing after the beginning of 2019.
    start_before_2019_mask = trials["trial_start"] < np.datetime64("2019-01-01")
    end_before_2019_mask = trials["trial_end"] < np.datetime64("2019-01-01")
    trials = trials[(start_before_2019_mask) & (end_before_2019_mask)]

    # sanity check that we got all na drugs out of DataFrame.
    if trials["Primary Drugs"].isna().sum() > 0:
        print("did not remove all na drug rows")

    return trials


def remove_dup_trial():
    """
    Function splits trial rows with multiple drugs into individual rows
    with duplicated information for all other columns.
    """
    df_temp = load_data()
    df_temp = df_temp.reset_index()

    ## expanded unique drugs to get a dataframe we can group by
    # logic is to append a duplicate row, change its values, do
    # this for all matching rows, then remove all the original rows.
    dupes = []
    for idx, item_list in enumerate(df_temp["Primary Drugs"]):
        split_list = item_list.split(",")
        if len(split_list) > 1:
            for i in split_list:
                # Appends a new row to data for a unique design value
                df_temp = df_temp.append(df_temp.iloc[idx, :])
                # 3 is the column number of where Trial Design lives
                # Changes the value of that specific cell to a single design value
                df_temp.iloc[-1, -3] = i
            dupes.append(idx)
    # we reset index so we can drop proper rows in next line
    df_temp2 = df_temp.reset_index(drop=True)
    df_trials = df_temp2.drop(dupes, axis=0)

    # Removes white space from drug names
    df_trials["Primary Drugs"] = df_trials["Primary Drugs"].apply(lambda x: x.strip())

    return df_trials


def feature_trial_design(df_trials, df_data):
    """
    This function is to extract the various types of trial designs in our
    dataset. It extracts out the unique values for the Trial Design column of
    df_trials and counts them per drug per phase.

    It turns a few of the trial design types into an other category due to
    low value counts.

    Inputs: df_trials and df_data
    Outputs: df_data with columns for most trial designs.
    """

    df_temp = df_trials
    dupes = []
    bad_cases = []
    df_temp = df_temp.reset_index()  # not sure if this is needed but it works.
    for idx, item_list in enumerate(df_temp["Trial Design"]):
        try:
            # Splits the string into tokenized words
            split_list = item_list.split(",")
        except:
            # Case exist where value of the cell is nan
            bad_cases.append(idx)
            continue
        if len(split_list) > 1:
            for i in split_list:
                # Appends a new row to data for a unique design value
                df_temp = df_temp.append(df_temp.iloc[idx, :])
                # 10 is the column number of where Trial Design lives
                # Changes the value of that specific cell to a single design value
                df_temp.iloc[-1, 10] = i.strip()
            dupes.append(idx)
    # we reset index so we can drop proper rows in next line
    df_temp = df_temp.reset_index(drop=True)
    df_temp = df_temp.drop(dupes, axis=0)

    # collapse all potential unique trial design values down to values with more than 20 counts
    df_temp["Trial Design"] = df_temp["Trial Design"].str.replace(
        "case control", "Other Design Status"
    )
    df_temp["Trial Design"] = df_temp["Trial Design"].str.replace(
        "survey", "Other Design Status"
    )
    df_temp["Trial Design"] = df_temp["Trial Design"].str.replace(
        "epidemiological", "Other Design Status"
    )
    df_temp["Trial Design"] = df_temp["Trial Design"].str.replace(
        "cross-sectional", "Other Design Status"
    )
    df_temp["Trial Design"] = df_temp["Trial Design"].str.replace(
        "postmarketing surveillance", "Other Design Status"
    )
    df_temp["Trial Design"] = df_temp["Trial Design"].str.replace(
        "sequential", "Other Design Status"
    )
    df_temp["Trial Design"] = df_temp["Trial Design"].str.replace(
        "single-blind", "Other Design Status"
    )

    # using a pivot table to turn the unique values of trial design into columns by drugs
    status_groups = df_temp.groupby(
        ["Primary Drugs", "Phase of Trial", "Trial Design"]
    )["Trial Design"].agg("count")
    status_groups = status_groups.rename(
        "Trial_Design_Mean"
    ).reset_index()  # couldn't reset index without renaming
    status_groups["Phase Design Type"] = (
        status_groups["Phase of Trial"] + " " + status_groups["Trial Design"]
    )
    pivoted_status_groups = status_groups.pivot(
        "Primary Drugs", "Phase Design Type", "Trial_Design_Mean"
    )
    pivoted_status_groups = pivoted_status_groups.fillna(0)
    df_data = df_data.merge(
        pivoted_status_groups, left_index=True, right_index=True, how="left"
    )

    # prints # of na values found in enumerating the trial design column.
    # comment these in for debugging
    # print(f'skipped {len(bad_cases)} bad cases')
    # print(f'{bad_cases}')
    return df_data


def drug_phase_df_template(df_trials):
    """
    Creates a template dataframe that has drugs vs phases
    populates all cells with zeros
    shape [n drugs x 8 phase types]

    Phases: ['Phase III', 'Phase II', 'Phase I', 'Phase I/II', 'Phase 0',
       'Phase IV', 'Phase II/III', 'Clinical Phase Unknown']

    Input: df_trials
    Output: df_drug_phase_temp
    """

    drug_list = df_trials["Primary Drugs"].unique()
    phase_of_trial_list = df_trials["Phase of Trial"].unique()
    # generate empty DataFrame with right dimensions
    df_drug_phase_temp = pd.DataFrame(
        np.zeros([len(drug_list), len(phase_of_trial_list)]),
        columns=phase_of_trial_list,
    )
    df_drug_phase_temp["Primary Drugs"] = drug_list
    df_drug_phase_temp = df_drug_phase_temp.set_index("Primary Drugs")

    return df_drug_phase_temp


def groupby_object_list(df_trials):
    """
    Function created to generalize feature extraction for features that
    need both per trial and per phase dimensions (e.g. number of trial
    centers). Designed to be used by function df_feature_extraction_by_phase.

    User needs to define any additional groupby objects here for future features.
    Objects used as inputs to generate df_data features

    Input: df_trials dataframes
    Output: groupby objects and name adjustment lists for each user defined
    feature.
    """
    # Phase names list for generating new columns for extracted features
    phase_of_trial_list = df_trials["Phase of Trial"].unique()

    ## FIRST USER GENERATED FEATURE: NUMBER OF TRIAL CENTERS ##
    # Takes the number of centers per trial per phase and calculates the mean
    df_trials["Number of Centers"] = df_trials["Trial Centre Details"].apply(
        trial_center_detail_to_table
    )
    df_trials["Number of Centers"].fillna(
        df_trials["Number of Centers"].mean(), inplace=True
    )

    # Takes the number of centers per phase and calculates the mean
    groupby_center_count = (
        df_trials.groupby(["Primary Drugs", "Phase of Trial"])["Number of Centers"]
        .agg("mean")
        .reset_index()
    )
    name_center_mean = [i + " center count" for i in phase_of_trial_list]

    ## SECOND USER GENERATED FEATURE: NUMBER OF SUBJECTS ##
    # Takes the number of subjects per trial per phase and calculates the mean
    groupby_patient_count = (
        df_trials.groupby(["Primary Drugs", "Phase of Trial"])["Planned Subject Number"]
        .agg("mean")
        .reset_index()
    )
    name_subject_mean = [i + " subject mean" for i in phase_of_trial_list]

    ## 3RD USER GENERATED FEATURE: TRIAL LENGTH ##
    # Takes the average difference of the start and stop time per drug per phase
    df_trials["time_diff"] = df_trials["trial_end"] - df_trials["trial_start"]
    df_trials["nano_time_diff"] = df_trials["time_diff"].astype(np.int64)

    # removes trials that have zero time delta
    zero_time_delta_mask = df_trials["time_diff"] > np.timedelta64(0)
    df_trials = df_trials[zero_time_delta_mask]
    groupby_time_deltas = pd.to_timedelta(
        df_trials.groupby(["Primary Drugs", "Phase of Trial"])["nano_time_diff"].agg(
            "mean"
        )
    ).reset_index()

    # put user defined group by objects into a list
    groupby_object_list = [
        groupby_patient_count,
        groupby_time_deltas,
        groupby_center_count,
    ]
    name_trial_length = [i + " trial length" for i in phase_of_trial_list]

    # we need to adjust column names, here is a list of the columns to adjust.
    name_column_adjust_list = [name_subject_mean, name_trial_length, name_center_mean]

    return groupby_object_list, name_column_adjust_list


def map_GB_drug_phase_to_df(df_trials, groupby_object):
    """
    Maps a grouby object that has the drug, phase, and feature into a dataframe
    that accounts for each drug and each phase

    Input:

    groupby_object: expects [n drug x 3] shape where the columns are exactly
    [Primary Drugs, Phase of Trial, %FEATURE%]

    Output:
    df_drug_phase: A populated df with the groupby data
    [n drugs x phases ]

    """
    # Generates a df template with drug x phase shape
    df_drug_phase = drug_phase_df_template(df_trials=df_trials)

    # Pulls index from df_drug_phase which has all the unique drug names
    drug_list = df_drug_phase.index.tolist()

    name_of_feature = [
        column
        for column in groupby_object.columns
        if ("Primary Drugs" not in column) and ("Phase of Trial" not in column)
    ][0]

    for drug in drug_list:
        for item in groupby_object[groupby_object["Primary Drugs"] == drug].iterrows():
            phase = list(item)[1]["Phase of Trial"]
            count = list(item)[1][name_of_feature]
            df_drug_phase.loc[drug, phase] = count

    return df_drug_phase


def df_feature_extraction_by_phase(df_trials):
    """
    Function extracts features that have data per phase and per trial.
    It calls groupby_object_list, then map_GB_drug_phase_to_df to turn semi-raw
    df_trials data into the df_data format.

    Input: df_trials
    Output: df_data
    """

    df_data = pd.DataFrame()

    # get all user defined group by objects.
    groupby_objects, name_column_adjust_list = groupby_object_list(df_trials=df_trials)

    # loop through groupby_objects, adjust their column names and add features
    # to new dataframe.
    for idx, groupby_object in enumerate(groupby_objects):

        # Adjust column names to phase - feature
        column_name_adj = name_column_adjust_list[idx]

        # Generate a dataframe with features
        df_temp = map_GB_drug_phase_to_df(
            df_trials=df_trials, groupby_object=groupby_object
        )
        df_temp.columns = column_name_adj
        # Combines the feature dataframes together
        df_data = pd.concat([df_data, df_temp], axis=1)

    return df_data


def feature_phase_pass_nopass(df_data):
    """
    Add columns of feature that determine whether trial was successful
    Based on whether there were subjects in the following phase

    Example Phase I got a pass if Phase II has patients in them
    INPUT: df_data
    OUTPUT: df_data with whether the phase is successful.
    """

    # using masks to create features using this logic:
    # if there are subjects in a future phase, then the drug must
    # have passed previous phase.
    df_data["Phase I Pass"] = df_data["Phase II subject mean"] != 0
    df_data["Phase II Pass"] = df_data["Phase III subject mean"] != 0
    df_data["Phase I/II Pass"] = df_data["Phase III subject mean"] != 0
    df_data["Phase III Pass"] = df_data["Phase IV subject mean"] != 0
    df_data["Phase II/III Pass"] = df_data["Phase IV subject mean"] != 0
    return df_data


def remove_org_dupes(x):
    """
    Helper function to feature_orangization_count
    Function designed for pd.apply. Takes the organisations column and
    removes duplicates using fuzzywuzzy (which uses levenshtein distance.)
    INPUT: cell of DataFrame
    OUTPUT: cell with duplicates removed.
    """

    drop_list = []
    # make sure the cell contains a string type.
    if type(x) != type("astring"):
        return []
    # makes sure null values are returned as null.
    if x == np.nan:
        return []
    # create a list of the array of organizations (x)
    split_orgs = x.split(",")

    # if there's more than 1 organization in list see if the levenshtein
    # distance between the items is greater than %50 (threshold picked by
    # eyeballing several examples with duplicates) then say they are
    # the same organization. If so, append their id to a list.
    # then delete lists.
    if len(split_orgs) > 1:
        for idx1, i in enumerate(split_orgs[:-1]):
            for idx2, j in enumerate(split_orgs[idx1 + 1 :]):
                # print(i,idx1, j ,idx2, fuzz.ratio(i,j))
                if fuzz.ratio(i, j) > 50:
                    drop_list.append(idx1 + idx2 + 1)
        output = np.delete(split_orgs, list(set(drop_list)))
        return [item.strip() for item in output]
    return [item.strip() for item in [x]]


def feature_orangization_count(df_trials, df_data):
    """
    Main function to extract number of organizations. Assumes df_data already
    exist with a table on the unique drug names in the index.

    Turns a string with multiple organizations into a number of Organization
    dedupping with the fuzzywuzzy library.
    INPUTS: df_trials, df_data
    OUTPUTS: df_data with new feature merged.
    """

    df_trials["Unique Organizations"] = df_trials["Organisations"].apply(
        remove_org_dupes
    )
    df_trials["Number of Organizations"] = df_trials["Unique Organizations"].apply(
        lambda x: len(x)
    )

    # Grab 2 columns from the dataframe.
    df_org_count = df_trials[["Primary Drugs", "Number of Organizations"]]
    # group by the max # of organizations in any row. Reset index to avoid having to
    # manage multi-level indices in Pandas.
    groupby_orgs = (
        df_org_count.groupby("Primary Drugs")["Number of Organizations"]
        .agg("max")
        .reset_index()
    )

    # add new feature to df_data
    df_data = df_data.merge(
        groupby_orgs, left_index=True, right_on="Primary Drugs", how="left"
    )
    df_data.set_index("Primary Drugs", inplace=True)

    return df_data


def extract_feature_trial_status(df_trials, df_data):
    """
    Assumes df_data already exist with a table on the unique drug names in the index

    This takes in df_trials and df_data, extracts out the counts of each trial status
    per drug and returns each of those status values as a column and merges to the df_data
    dataframe.
    INPUTS: df_trials, df_data
    OUTPUTS: df_data with new feature
    """

    # collapse all potential unique trial status down to just Completed, Discontinued
    # and other because the others are a small (32 of ~1450) set of the results currently
    df_trials["Trial Status"] = df_trials["Trial Status"].str.replace(
        "Active, no longer recruiting", "Other Trial Status"
    )
    df_trials["Trial Status"] = df_trials["Trial Status"].str.replace(
        "Withdrawn prior to enrolment", "Other Trial Status"
    )
    df_trials["Trial Status"] = df_trials["Trial Status"].str.replace(
        "Recruiting", "Other Trial Status"
    )
    df_trials["Trial Status"] = df_trials["Trial Status"].str.replace(
        "Suspended", "Other Trial Status"
    )

    # counts the number of trial statuses per unique category.
    status_groups = df_trials.groupby(
        ["Primary Drugs", "Phase of Trial", "Trial Status"]
    )["Trial Status"].agg("count")
    status_groups = status_groups.rename(
        "Trial_Status_Count"
    ).reset_index()  # couldn't reset index without renaming

    # created a new future column name as a cell in the status_groups df to be pivoted in the next line.
    status_groups["Phase Trial Status"] = (
        status_groups["Phase of Trial"] + " " + status_groups["Trial Status"]
    )

    # use pivot table to pivot data to create a separate column for each trial status count per drug and trial
    pivoted_status_groups = status_groups.pivot(
        "Primary Drugs", "Phase Trial Status", "Trial_Status_Count"
    )
    pivoted_status_groups = pivoted_status_groups.fillna(0)
    df_data = df_data.merge(
        pivoted_status_groups, left_index=True, right_index=True, how="left"
    )

    return df_data


def separate_phases_into_dfs(df_data):
    """
    Function separates out the dataframes into a data frame for each major phase
    (Phase I, Phase II, Phase III). Other less used phases ignored.

    Input = df_data
    Output = list_of_dfs where it is currently hard coded to 3.
    """
    # create list of phases to loop through, intentionally leave space
    # in the string for each phase to help the regex filter
    phase_list = ["Phase I ", "Phase II ", "Phase III "]

    list_of_dfs = []
    for i in phase_list:
        phase = df_data.filter(regex=i).reset_index()
        phase["Number of Organizations"] = pd.Series(
            data=df_data["Number of Organizations"].values
        )
        phase.set_index("Primary Drugs", inplace=True)
        list_of_dfs.append(phase)
        phase.to_pickle("data/df_phaseIfeatures_" + i + ".pk")
    return list_of_dfs


def trial_center_detail_to_table(x):
    """
    helper function to turn a trial center cell into a table
    """
    try:
        rows = str.split(x, ";")
        table = [str.split(row, "|") for row in rows]
        return len(rows)
    except:
        return np.nan


def save_data(df_data, list_of_dfs, date="SOMEDATE"):
    """
    Saves the extracted features and the list of dfs to csvs to be used
    in modeling scripts.

    INPUTS: df_data, list_of_dfs, a date when you run this function.
    OUTPUTS: None. Files will be saved on your local computer.
    """
    # Saves data to csv for viewing
    df_data.to_csv(f"{date}.csv")

    for idx, item in enumerate(list_of_dfs):
        item.to_pickle(f"data/df_{idx+1}.pk")


# if __name__ == '__main__':
def main():
    df_trials = remove_dup_trial()
    df_data = df_feature_extraction_by_phase(df_trials=df_trials)
    df_data = feature_phase_pass_nopass(df_data)
    df_data = extract_feature_trial_status(df_trials=df_trials, df_data=df_data)
    df_data = feature_orangization_count(df_trials, df_data)
    df_data = feature_trial_design(df_trials, df_data)
    list_of_dfs = separate_phases_into_dfs(df_data)
    save_data(df_data, list_of_dfs, date="Mar21")
    print("Feature Extraction Complete!")
