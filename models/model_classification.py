
'''
Scipt builds a logistic regression model for a binary classification

'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB




class ClassificationModel():

    # Initalizer / Instance Attributes
    def __init__(self,df_data,model_type):
        '''
        df_data = dataframe of data with X and y in the data
        model_tyle = explicitly states which model to return
            ex = model_tyle = 'logistic_regression'
        '''
        self.df_data = df_data
        self.model_type = model_type

        # Add dict here
        model_dict = dict{
            'logistic_regression':self.logistic_regression_model,
            'gaussian_naive_bayes':self.gaussian_naive_bayes_model,
            'multinomial_naive_bayes':self.multinomial_naive_bayes_model,
            'random_forest':self.random_forest_model,
            'xgboost':self.xgboost_model,
            }


        # Load this model
        try:
            model_dict[model_type]()
        except:
            print('Please enter in a valid model type')
            return


    def timedelta_change(x):
        # Applies a change to the days to a float, there's a mix of datetime Objects and floats in the dataframe column length
        try:
            y = x.days
        except:
            y = x
        return y

    def apply_length_adjustment(self):
        '''
        Special feature adjuster to change delta time object to a float
        applies the timedelta_change definition
        '''
        label_col_name = df_data.filter(regex='trial length').columns[0]
        self.df_data[label_col_name] = self.df_data[label_col_name].apply(timedelta_change)
        self.df_data.fillna(0,inplace=True)


    def test_train_split(self,random_state=42,test_size=0.2):
        '''
        Performs a train test split
        '''

        # Make adjustment of time_delta_objects to floats - needed to run model
        if self.df_data.filter(regex='trial length').shape[1] > 1:
            self.apply_length_adjustment()


        # Pulls the label column from dataframe and assigns
        # other columns as feature
        # First check if the column there's only one column with pass in it
        if self.df_data.filter(regex='Pass').shape[1] =! 1:
            print('More than one pass label column - adjust column names')
            return

        # Pulls the column name
        self.label_col_name = self.df_data.filter(regex='Pass').columns[0]

        # Define train and test data
        X = self.df_data.drop(label_col_name,axis=1)
        y = self.df_data[label_col_name]

        # Split the dataset in two equal parts
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)


    def logistic_regression_model(self):
        '''
        Build a logistic regression model with L2 regularization and defines
        the grid search parameters space

        Standard scaler is applied to the features
        '''
        # LogisticRegression, balanced is used because classes are highly
        # imbalanced
        self.model = LogisticRegression(
            solver='liblinear',
            random_state=42,
            class_weight='balanced')

        # Builds pipe. Apply a standard scaler to features
        self.pipe = Pipeline(
                    steps=[
                        ('scale', StandardScaler()),
                        ('logistic', self.model)
                    ])

        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        self.param_grid = {
            'logistic__penalty': ['l2'],
            'logistic__C': np.logspace(0, 4, 2),
            }

    def gaussian_naive_bayes_model(self):
        '''
        Builds a gaussian naive bayes model
        '''
        self.model = GaussianNB()
        self.pipe = Pipeline(
                    steps=[
                        ('scale', StandardScaler()),
                        ('gaussianNB', self.model)
                    ])
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        self.param_grid = {}

    def multinomial_naive_bayes_model(self):
        '''
        Builds a multinominal naive bayes model
        '''
        self.model = MultinomialNB()
        self.pipe = Pipeline(
                    steps=[
                        ('scale', StandardScaler()),
                        ('multinominalNB', self.model)
                    ])
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        self.param_grid = {}



    def grid_search(self,cv=5):


        # Performs a grid search for the model
        self.model_gscv = GridSearchCV(self.pipe, param_grid=self.param_grid, iid=False, cv=cv,
                              return_train_score=False)
        self.model_gscv.fit(self.X_train, self.y_train)


        print('Best model')
        print(f"CV score {self.model_gscv.best_score_}")
        print(f"Best parameters: {self.model_gscv.best_params_}")

        if self.model_type=='logistic_regression':
            LR_coef = np.exp(self.model_gscv.best_estimator_.steps[0][1].coef_)
            print(f'Odds coefficients: {LR_coef[0]}')

#     def predict_test(self):
#
#         # Make a prediction on entire training set
#         y_pred = model_gscv.best_estimator_.predict(X_test)
#
# report = classification_report(y_true=y_test, y_pred=y_pred)
# print(report)
#
# sum(y_test==0)/len(y_test)
# sum(y_pred==0)/len(y_pred)



if __name__ == '__main__':
    df_data = pd.read_pickle('data/df_phaseIfeatures_Phase III .pk')
