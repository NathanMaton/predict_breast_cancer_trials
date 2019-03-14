import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# # not DRY as the answers to make multiargument apply functions involved more complicated things than I wanted.
def strip_planned_start(x):
    date_str = re.findall("[\d\s\w]*", x["Trial Initiation date"])[0].strip()
    plan_str = re.findall("planned", x["Trial Initiation date"])
    if plan_str:
        return 0
    else:
        return pd.to_datetime(date_str)

def strip_planned_end(x):
    date_str = re.findall("[\d\s\w]*", x["Trial End date"])[0].strip()
    plan_str = re.findall("planned", x["Trial End date"])
    if plan_str:
        return 0
    else:
        return pd.to_datetime(date_str)

def load_data():
    trials = pd.read_excel("bcdrugsct.xlsx")
    trials = trials[trials["Primary Drugs"].isna() == False]
    trials["trial_start"] = trials.apply(strip_planned_start, axis=1)
    trials["trial_end"] = trials.apply(strip_planned_end, axis=1)
    start_mask = trials["trial_start"] != 0
    end_mask = trials["trial_end"] != 0
    trials = trials[(start_mask) & (end_mask)]

    start_before_2019_mask = trials["trial_start"] < np.datetime64('2019-01-01')
    end_before_2019_mask = trials["trial_end"] < np.datetime64('2019-01-01')
    trials = trials[(start_before_2019_mask) & (end_before_2019_mask)]

    if trials["Primary Drugs"].isna().sum() > 0:
        print("did not remove all na drug rows")

    return trials

def p1_subject_counts(df):
    # see if the drug has a trial in each phase.
    p1_df= df[df["Phase of Trial"]=='Phase I']
    return sum(p1_df['Planned Subject Number'].values)

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
    try:
        phases = ["Phase I", "Phase II", "Phase III"]
        res = []
        for idx, v in enumerate(phases[:-1]):
            p_start = np.min(
                list(
                    df[(df["trial_start"] != False) & (df["Phase of Trial"] ==  phases[idx])][
                        "trial_start"
                    ].values
                )
            )
            res.append(p_start)

            p_first_end = np.min(
                list(
                    df[
                        (df["trial_start"] != False)
                        & (df["Phase of Trial"] ==  phases[idx+1])
                        & (df["trial_start"] > p_start)
                    ]["trial_start"].values
                )
            )
            res.append(p_first_end)

            p_last_end = np.max(
                list(
                    df[(df["trial_start"] != False) & (df["Phase of Trial"] ==  phases[idx])][
                        "trial_start"
                    ].values
                )
            )
            res.append(p_last_end)
        return res
    except:
        return pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT


def return_null_row(drug_name, count):
    return [drug_name, count, False, False, False, False, False,
            pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT]


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

    #subset to majority of data
    trial_phases = ["Phase I", "Phase II", "Phase III", "Phase IV", "Phase I/II", "Phase II/III"]
    cleaned_drug_trials = drug_trials[
        drug_trials["Phase of Trial"].isin(trial_phases)
    ]  # only rows w/ clean trials

    # turn categorial variables into ordinals
    phase_dummies = pd.get_dummies(cleaned_drug_trials["Phase of Trial"])
    df = cleaned_drug_trials.join(phase_dummies)

    # convert times to time objects
    # df["trial_start"] = df.apply(start_to_datetime, axis=1)
    # df["trial_end"] = df.apply(end_to_datetime, axis=1)

    p1_start, p1_first_end, p1_last_end, p2_start, p2_first_end, p2_last_end = get_times(df)

    res = [
        drug_name,
        count,
        pass_phase_1,
        pass_phase_2,
        pass_phase_3,
        pass_phase_one_two,
        pass_phase_two_three,
        p1_start,
        p1_first_end,
        p1_last_end,
        p2_start,
        p2_first_end,
        p2_last_end
    ]
    return res


def unique_drugs(trials):
    messy_drug_list = list(trials["Primary Drugs"].values)
    single_word_drugs_list = [drug for row in messy_drug_list for drug in row.split(", ")]
    c = Counter(
        np.array([drug for row in single_word_drugs_list for drug in row.split(", ")])
    )
    filtered_c = {k: v for k, v in c.items() if v > 0}

    return filtered_c


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
            "phase_2_start",
            "phase_2_first_end",
            "phase_2_last_end"
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
