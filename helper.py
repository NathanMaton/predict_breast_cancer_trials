import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re


def load_data():

    trials = pd.read_excel("bcdrugsct.xlsx")
    trials = trials[trials["Primary Drugs"].isna() == False]
    cols = [
        "Trial ID",
        "Mechanism of action",
        "Drug class (therapeutic effect)",
        "Drug class (chemical)",
        "Indication",
        "Organisations",
        "Trial Design",
        "Location",
        "Phase of Trial",
        "Subject Age",
        "Planned Subject Number",
        "Trial Centre Details",
        "Lead Centre",
        "Trial Initiation date",
        "Trial End date",
        "Trial Status",
        "Trial History",
        "Diseases treated",
        "Primary Drugs",
    ]
    # smaller_trials = trials[cols]
    date_cols = [
        "Phase of Trial",
        "Trial Initiation date",
        "Trial End date",
        "trial_start",
        "trial_end",
    ]
    if trials["Primary Drugs"].isna().sum() > 0:
        print("did not remove all na drug rows")

    return trials


# not DRY as the answers to make multiargument apply functions involved more complicated things than I wanted.
def start_to_datetime(x):
    date_str = re.findall("[\d\s\w]*", x["Trial Initiation date"])[0].strip()
    if date_str == "":
        return pd.NaT
    else:
        return pd.to_datetime(date_str)


def end_to_datetime(x):
    date_str = re.findall("[\d\s\w]*", x["Trial End date"])[0].strip()
    if date_str == "":
        return pd.NaT
    else:
        return pd.to_datetime(date_str)


def prob_of_success(drug_trials):
    # see if the drug has a trial in each phase.
    unique_trial_values = drug_trials["Phase of Trial"].unique()
    (
        pass_phase_1,
        pass_phase_2,
        pass_phase_3,
        pass_phase_one_two,
        pass_phase_two_three,
    ) = (False, False, False, False, False)
    if ("Phase I" in unique_trial_values) & ("Phase II" in unique_trial_values):
        pass_phase_1 = True

    if ("Phase II" in unique_trial_values) & ("Phase III" in unique_trial_values):
        pass_phase_2 = True

    if ("Phase III" in unique_trial_values) & ("Phase IV" in unique_trial_values):
        pass_phase_3 = True

    if ("Phase I/II" in unique_trial_values) & ("Phase III" in unique_trial_values):
        pass_phase_one_two = True

    if ("Phase II/III" in unique_trial_values) & ("Phase IV" in unique_trial_values):
        pass_phase_two_three = True

    return [
        pass_phase_1,
        pass_phase_2,
        pass_phase_3,
        pass_phase_one_two,
        pass_phase_two_three,
    ]


def get_times(df):
    # finds only the dates that aren't false for each phase.
    phase_1_start = np.min(
        list(
            df[(df["trial_start"] != False) & (df["Phase of Trial"] == "Phase I")][
                "trial_start"
            ].values
        )
    )

    phase_1_first_end = np.min(
        list(
            df[
                (df["trial_start"] != False)
                & (df["Phase of Trial"] == "Phase II")
                & (df["trial_start"] > phase_1_start)
            ]["trial_start"].values
        )
    )

    phase_1_last_end = np.max(
        list(
            df[(df["trial_start"] != False) & (df["Phase of Trial"] == "Phase I")][
                "trial_start"
            ].values
        )
    )
    return phase_1_start, phase_1_first_end, phase_1_last_end


def return_null_row(drug_name, count):
    return [drug_name, count, False, False, False, False, False, pd.NaT, pd.NaT, pd.NaT]


def pipeline(drug_name, count, trials):
    """
    Use a drug name and get out the probabilitiy that it passes multiple phases
    and the times for it to do so.
    """
    drug_trials = trials[trials["Primary Drugs"].str.contains(drug_name, case=False)]
    (
        pass_phase_1,
        pass_phase_2,
        pass_phase_3,
        pass_phase_one_two,
        pass_phase_two_three,
    ) = prob_of_success(drug_trials)

    if pass_phase_1 == False:
        return return_null_row(drug_name, count)

    # get rid of rows outside Phase I or II
    trial_phases = ["Phase I", "Phase II"]
    cleaned_drug_trials = drug_trials[
        drug_trials["Phase of Trial"].isin(trial_phases)
    ]  # only rows w/ clean trials

    # turn categorial variables into ordinals
    phase_dummies = pd.get_dummies(cleaned_drug_trials["Phase of Trial"])
    df = cleaned_drug_trials.join(phase_dummies)

    # convert times to time objects
    df["trial_start"] = df.apply(start_to_datetime, axis=1)
    df["trial_end"] = df.apply(end_to_datetime, axis=1)

    # commented out to try and solve for cases with Phase I trials having NAT like 'Epirubicin'
    # if  numpy.datetime64('NaT') in df[df['Phase of Trial'] == 'Phase I']['trial_start'].values:
    #    return return_null_row(drug_name, count)

    phase_1_start, phase_1_first_end, phase_1_last_end = get_times(df)

    res = [
        drug_name,
        count,
        pass_phase_1,
        pass_phase_2,
        pass_phase_3,
        pass_phase_one_two,
        pass_phase_two_three,
        phase_1_start,
        phase_1_first_end,
        phase_1_last_end,
    ]
    return res


def unique_drugs(trials):
    unique_drug_list = list(trials["Primary Drugs"].unique())
    unique_drugs = {drug for row in unique_drug_list for drug in row.split(", ")}

    # are these 3 lines necessary?
    # drug_counts = trials['Primary Drugs'].value_counts()
    # drugs_10_trials = drug_counts[drug_counts > 1].index
    # all_drugs = drug_counts.index

    c = Counter(
        np.array([drug for row in unique_drug_list for drug in row.split(", ")])
    )
    return c


def generate_drugs_df(counts, trials):
    res, except_ids = [], []
    for k, v in counts.items():
        try:
            res.append(pipeline(k, v, trials))
        except:
            except_ids.append(k)

    print(f"got {len(res)} results and had {len(except_ids)} exceptions")

    drugs_df = pd.DataFrame(
        res,
        columns=[
            "drug",
            "num_of_trials",
            "phase_1_success",
            "phase_2_success",
            "phase_3_success",
            "phase_1_2_success",
            "phase_2_3_success",
            "phase_1_start",
            "phase_1_first_end",
            "phase_1_last_end",
        ],
    )

    ans = [
        drugs_df[drugs_df["phase_1_success"] == True].shape[0] / drugs_df.shape[0],
        drugs_df[drugs_df["phase_2_success"] == True].shape[0]
        / drugs_df[drugs_df["phase_1_success"] == True].shape[0],
        drugs_df[drugs_df["phase_3_success"] == True].shape[0]
        / drugs_df[drugs_df["phase_2_success"] == True].shape[0],
    ]
    print(f"probs are{[round(100 * i, 2) for i in ans]}")
    sorted_drugs = drugs_df.sort_values("num_of_trials", ascending=False)
    return sorted_drugs, except_ids


def inspect_counts_of_exceptions(counts, exceptions):
    for k, v in c.items():
        if k in exceptions:
            print(f"exception drug {k} has {v} trials in data")
