'''
Script for plotting for presentation
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

%config InlineBackend.figure_format = 'svg'

#makes the plots look nicer
plt.style.use('ggplot')


#potential results chart
phase_success_df = pd.DataFrame([['Phase 1',.63,'Industry Standard'],['Phase 1',.87,'Prototype'],
                        ['Phase 2',.3,'Industry Standard'],['Phase 2',.6,'Prototype'],
                        ['Phase 3',.58,'Industry Standard'],['Phase 3',.8,'Prototype'],
                        ],columns=['phase','percentage','model'])
sns.barplot(x='phase', y='percentage',data=phase_success_df, hue='model')
plt.title('Industry standard vs. project predicted success rates');

#plots phase 1 success
df_phase1 = pd.read_pickle('data/df_1.pk')
plot1 = sns.countplot(df_phase1['Phase I Pass'])
plt.title('Phase I Pass rate')
plt.yticks([0,50,100]);
figure = plot1.get_figure()
figure.savefig('images/phase1_success.svg',  bbox_inches='tight')

#plots phase 2 success
df_phase2 = pd.read_pickle('data/df_2.pk')
plot2 = sns.countplot(df_phase2['Phase II Pass'])
plt.title('Phase II Pass rate');
figure = plot2.get_figure()
figure.savefig('images/phase2_success.svg',  bbox_inches='tight')

#plots phase 3 success
df_phase3 = pd.read_pickle('data/df_3.pk')
plot3 = sns.countplot(df_phase3['Phase III Pass'])
plt.title('Phase III Pass rate');
figure = plot3.get_figure()
figure.savefig('images/phase3_success.svg',  bbox_inches='tight')

#these 3 functions allow you to change the type of the trial length from
#timedelta to a float
def timedelta_change(x):
    # Applies a change to the days to a float, there's a mix of datetime Objects and floats in the dataframe column length
    try:
        y = x.days
    except:
        y = x
    return y

def apply_length_adjustment(df_data):
    '''
    Special feature adjuster to change delta time object to a float
    applies the timedelta_change definition
    '''
    label_col_name = df_data.filter(regex='trial length').columns[0]
    df_data[label_col_name] = df_data[label_col_name].apply(timedelta_change)
    df_data.fillna(0,inplace=True)
    return df_data

def change_time_data_type(df_data):
    '''
    changes each phase trial length (e.g. df_data["Phase I trial length"] to float instead of timedelta
    '''
    return apply_length_adjustment(df_data)

#phase 1 distribution of completion times graphing code
plot4 = change_time_data_type(df_phase1)["Phase I trial length"].hist()
plt.title("Distribution of days to finish Phase I")
plt.xlabel("Days to complete phase")
plt.ylabel("Count")
figure = plot4.get_figure()
figure.savefig('images/phase1_time_to_completion_histogram.svg',  bbox_inches='tight');

#phase 2 distribution of completion times graphing code
plot5 = change_time_data_type(df_phase2)["Phase II trial length"].hist()
plt.title("Distribution of days to finish Phase II")
plt.xlabel("Days to complete phase")
plt.ylabel("Count")
figure = plot5.get_figure()
figure.savefig('images/phase2_time_to_completion_histogram.svg',  bbox_inches='tight');

#phase 3 distribution of completion times graphing code
plot6 = change_time_data_type(df_phase3)["Phase III trial length"].hist()
plt.title("Distribution of days to finish Phase III")
plt.xlabel("Days to complete phase")
plt.ylabel("Count")
figure = plot6.get_figure()
figure.savefig('images/phase3_time_to_completion_histogram.svg',  bbox_inches='tight');
