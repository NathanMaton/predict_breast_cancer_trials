"""
Script for plotting for project presentation.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

## HELPER FUNCTIONS ##

# these 3 functions allow you to change the type of the trial length from
# timedelta to a float
def timedelta_change(x):
    # Applies a change to the days to a float, there's a mix of datetime
    #Objects and floats in the dataframe column length
    try:
        y = x.days
    except:
        y = x
    return y


def apply_length_adjustment(df_data):
    """
    Special feature adjuster to change delta time object to a float
    applies the timedelta_change definition
    """
    label_col_name = df_data.filter(regex="trial length").columns[0]
    df_data[label_col_name] = df_data[label_col_name].apply(timedelta_change)
    df_data.fillna(0, inplace=True)
    return df_data


def change_time_data_type(df_data):
    """
    changes each phase trial length (e.g. df_data["Phase I trial length"] to float instead of timedelta
    """
    return apply_length_adjustment(df_data)


## PLOTTING CODE ##

# makes the plots look nicer
plt.style.use("ggplot")
%config InlineBackend.figure_format = 'svg'

# plots phase 1 success
df_phase1 = pd.read_pickle("data/df_1.pk")
plot1 = sns.countplot(df_phase1["Phase I Pass"])
plt.title("Phase I Pass rate")
plt.yticks([0, 50, 100])
figure = plot1.get_figure()
figure.savefig("images/phase1_success.svg", bbox_inches="tight")

# plots phase 2 success
df_phase2 = pd.read_pickle("data/df_2.pk")
plot2 = sns.countplot(df_phase2["Phase II Pass"])
plt.title("Phase II Pass rate")
figure = plot2.get_figure()
figure.savefig("images/phase2_success.svg", bbox_inches="tight")

# plots phase 3 success
df_phase3 = pd.read_pickle("data/df_3.pk")
plot3 = sns.countplot(df_phase3["Phase III Pass"])
plt.title("Phase III Pass rate")
figure = plot3.get_figure()
figure.savefig("images/phase3_success.svg", bbox_inches="tight")


# phase 1 distribution of completion times graphing code
plot4 = change_time_data_type(df_phase1)["Phase I trial length"].hist(bins=50)
plt.title("Distribution of days to finish Phase I")
plt.xlabel("Days to complete phase")
plt.ylabel("Count")
figure = plot4.get_figure()
figure.savefig("images/phase1_time_to_completion_histogram.svg", bbox_inches="tight")

# phase 2 distribution of completion times graphing code
plot5 = change_time_data_type(df_phase2)["Phase II trial length"].hist(bins=50)
plt.title("Distribution of days to finish Phase II")
plt.xlabel("Days to complete phase")
plt.ylabel("Count")
figure = plot5.get_figure()
figure.savefig("images/phase2_time_to_completion_histogram.svg", bbox_inches="tight")

# phase 3 distribution of completion times graphing code
plot6 = change_time_data_type(df_phase3)["Phase III trial length"].hist(bins=50)
plt.title("Distribution of days to finish Phase III")
plt.xlabel("Days to complete phase")
plt.ylabel("Count")
figure = plot6.get_figure()
figure.savefig("images/phase3_time_to_completion_histogram.svg", bbox_inches="tight")

# phase 1 distribution of target
plot7 = change_time_data_type(df_phase1)["Phase I trial length"].hist(bins=50)
plt.title("Distribution of days to finish Phase I")
plt.xlabel("Days to complete phase")
plt.ylabel("Count")
figure = plot7.get_figure()
figure.savefig("images/phase1_target_class_balance.svg", bbox_inches="tight")

# phase 2 distribution of target
plot8 = change_time_data_type(df_phase2)["Phase II trial length"].hist(bins=50)
plt.title("Distribution of days to finish Phase II")
plt.xlabel("Days to complete phase")
plt.ylabel("Count")
figure = plot8.get_figure()
figure.savefig("images/phase2_target_class_balance.svg", bbox_inches="tight")

# phase 3 distribution of target
plot9 = change_time_data_type(df_phase3)["Phase III trial length"].hist(bins=50)
plt.title("Distribution of days to finish Phase III")
plt.xlabel("Days to complete phase")
plt.ylabel("Count")
figure = plot9.get_figure()
figure.savefig("images/phase3_target_class_balance.svg", bbox_inches="tight")

# phase 1 distribution of number of organizations
plot9 = df_phase1["Number of Organizations"].hist(bins=10)
plt.title("Distribution of number of organizations")
plt.xlabel("Number of organizations working on drug")
plt.ylabel("Count")
figure = plot9.get_figure()
figure.savefig("images/num_orgs.svg", bbox_inches="tight")

# EDA on trial length plot, data input from looking at
# class imbalance on models.
phase_time_eda_df = pd.DataFrame(
    [
        ["Phase 1", 0.59, "0-1.5"],
        ["Phase 1", 0.25, "1.5-3"],
        ["Phase 1", 0.16, "3+"],
        ["Phase 2", 0.53, "0-1.5"],
        ["Phase 2", 0.31, "1.5-3"],
        ["Phase 2", 0.15, "3+"],
        ["Phase 3", 0.87, "0-1.5"],
        ["Phase 3", 0.11, "1.5-3"],
        ["Phase 3", 0.02, "3+"],
    ],
    columns=["phase", "percentage", "completion time (in years)"],
)
plot10 = sns.barplot(
    x="phase", y="percentage", data=phase_time_eda_df, hue="completion time (in years)"
)
plt.title("Class balance for time to phase completion")
plt.ylabel("Percentage")
plt.xlabel("Phases")
figure = plot10.get_figure()
figure.savefig("images/time_results.svg", bbox_inches="tight")



## MODEL SUCCESS RESULTS CHARTS

# This data was input from our logging files after running our models.
success_naive_phase_1 = 0.5
success_model_phase_1 = 0.57  # random forrest
success_naive_phase_2 = 0.66
success_model_phase_2 = 0.7  # random forrest
success_naive_phase_3 = 0.78
success_model_phase_3 = 0.75  # random forrest
# approval_percentage = .85

success_total_naive = (
    success_naive_phase_1 * success_naive_phase_2 * success_naive_phase_3
)
success_total_naive
success_total_model = (
    success_model_phase_1 * success_model_phase_2 * success_model_phase_3
)
success_pct_diff = (success_total_model - success_total_naive) / success_total_model
expected_value_using_model = 2.6 * (success_pct_diff)
expected_value_using_model
success_pct_diff

time_naive_phase_1 = 0.34
time_model_phase_1 = 0.56  # random forrest
time_naive_phase_2 = 0.39
time_model_phase_2 = 0.61  # random forrest
time_naive_phase_3 = 0.63
time_model_phase_3 = 0.93  # random forrest
# approval_percentage = .85
time_total_naive = time_naive_phase_1 * time_naive_phase_2 * time_naive_phase_3
time_total_model = time_model_phase_1 * time_model_phase_2 * time_model_phase_3
time_total_naive
time_total_model
time_pct_diff = (time_total_model - time_total_naive) / time_total_naive
time_pct_diff
expected_value_using_model = 2.6 * (time_pct_diff)

# potential success results
phase_success_df.head()

phase_success_df = pd.DataFrame(
    [
        ["Phase 1", success_naive_phase_1, "Industry Standard"],
        ["Phase 1", success_model_phase_1, "Prototype"],
        ["Phase 2", success_naive_phase_2, "Industry Standard"],
        ["Phase 2", success_model_phase_2, "Prototype"],
        ["Phase 3", success_naive_phase_3, "Industry Standard"],
        ["Phase 3", success_model_phase_3, "Prototype"],
        ["Phase I to approval", success_total_naive, "Industry Standard"],
        ["Phase I to approval", success_total_model, "Prototype"],
    ],
    columns=["phase", "percentage", "model"],
)
plot11 = sns.barplot(x="phase", y="percentage", data=phase_success_df, hue="model")
plt.title("Industry standard vs. model predicted success rates")
plt.ylabel("Model confidence (exp. log loss)")
plt.xlabel("Phases")
figure = plot11.get_figure()
figure.savefig("images/success_results.svg", bbox_inches="tight")


## TRIAL LENGTH MODEL RESULTS CHARTS CODE ##
phase_time_df.head()
phase_time_df = pd.DataFrame(
    [
        ["Phase 1", time_naive_phase_1, "Industry Standard"],
        ["Phase 1", time_model_phase_1, "Prototype"],
        ["Phase 2", time_naive_phase_2, "Industry Standard"],
        ["Phase 2", time_model_phase_2, "Prototype"],
        ["Phase 3", time_naive_phase_3, "Industry Standard"],
        ["Phase 3", time_model_phase_3, "Prototype"],
        ["Phase I to approval", time_total_naive, "Industry Standard"],
        ["Phase I to approval", time_total_model, "Prototype"],
    ],
    columns=["phase", "percentage", "model"],
)
plot12 = sns.barplot(x="phase", y="percentage", data=phase_time_df, hue="model")
plt.title("Industry standard vs. model predicted time to completion")
plt.ylabel("Model confidence (exp. log loss)")
plt.xlabel("Phases")
figure = plot12.get_figure()
figure.savefig("images/time_results.svg", bbox_inches="tight")
