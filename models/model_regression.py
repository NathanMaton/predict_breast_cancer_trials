
'''
Script builds a regression model for a binary classification

'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score

# ML Models
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from models.model_classification import ClassificationModel

# import logger
from loguru import logger


# Start a log filter
#logger.add(f'logs/model_performance.log')


class RegressionModel(ClassificationModel):

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
        model_dict = {
            'ols':self.ols,
            'lasso':self.lasso,
            'ridge':self.ridge,
        }

        #Load this model
        try:
            model_dict[model_type]()
        except:
            print('Please enter in a valid model type')
            return

        # Runs the script
        self.wrapper()

    def test_train_split(self,random_state=42,test_size=0.2):
        '''
        Performs a train test split
        '''

        # Make adjustment of time_delta_objects to floats - needed to run model
        if self.df_data.filter(regex='trial length').shape[1] == 1:
            self.apply_length_adjustment()

        # Pulls the label column from dataframe and assigns
        # other columns as feature
        # First check if the column there's only one column with pass in it
        if self.df_data.filter(regex='trial length').shape[1] != 1:
            print('More than one pass label column - adjust column names')
            return

        # Pulls the column name
        pass_col_name = self.df_data.filter(regex='Pass').columns[0]
        label_col_name = self.df_data.filter(regex='trial length').columns[0]
        # Change data type to all floats
        self.df_data = self.df_data.astype(dtype='float')

        # Define train and test data
        X = self.df_data.drop([pass_col_name,label_col_name],axis=1)
        y = self.df_data[label_col_name]

        # Split the dataset in two equal parts
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

    def ols(self):
        '''
        Build an OLS model and defines
        the grid search parameters space

        '''
        # LogisticRegression, balanced is used because classes are highly
        # imbalanced
        self.model = linear_model.LinearRegression(
            fit_intercept=True,
            normalize=False,
            )

        # Builds pipe. Apply a standard scaler to features
        self.pipe = Pipeline(
                    steps=[
                        ('scale', StandardScaler()),
                        ('ols', self.model)
                    ])

        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        self.param_grid = {
            'ols__fit_intercept': [True, False],
            'ols__normalize': [True, False],
            }

    def lasso(self):
        '''
        Build a lasso model and defines
        the grid search parameters space

        '''
        # LogisticRegression, balanced is used because classes are highly
        # imbalanced
        self.model = linear_model.Lasso(
            alpha=.1,
            fit_intercept=True,
            normalize=False,
            random_state=42,
            )

        # Builds pipe. Apply a standard scaler to features
        self.pipe = Pipeline(
                    steps=[
                        ('scale', StandardScaler()),
                        ('lasso', self.model)
                    ])

        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        self.param_grid = {
            'lasso__alpha': [.001,.01,.1,1,3,5,10],
            'lasso__fit_intercept': [True, False],
            'lasso__normalize': [True, False],
            }

    def ridge(self):
        '''
        Build a lasso model and defines
        the grid search parameters space

        '''
        # LogisticRegression, balanced is used because classes are highly
        # imbalanced
        self.model = linear_model.Ridge(
            alpha=.1,
            fit_intercept=True,
            normalize=False,
            random_state=42,
            )

        # Builds pipe. Apply a standard scaler to features
        self.pipe = Pipeline(
                    steps=[
                        ('scale', StandardScaler()),
                        ('ridge', self.model)
                    ])

        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        self.param_grid = {
            'ridge__alpha': [.001,.01,.1,1,3,5,10],
            'ridge__fit_intercept': [True, False],
            'ridge__normalize': [True, False],
            }

    # def grid_search(self,cv=5):
    #
    #
    #     # Performs a grid search for the model
    #     self.model_gscv = GridSearchCV(self.pipe, param_grid=self.param_grid, iid=False, cv=cv,
    #                           return_train_score=False)
    #     self.model_gscv.fit(self.X_train, self.y_train)
    #
    #
    #     print('Best model')
    #     print(f"CV score {self.model_gscv.best_score_}")
    #     print(f"Best parameters: {self.model_gscv.best_params_}")
    #
    #     if self.model_type=='logistic_regression':
    #         LR_coef = np.exp(self.model_gscv.best_estimator_.steps[1][1].coef_)[0]
    #
    #         #print()
    #
    #         features = self.X_train.columns.tolist()
    #         print(f'Odds coefficients')
    #         print(list(zip(features,np.round(LR_coef,3))))
    #
    #         #print(f'Odds coefficients: {LR_coef[0]}')

    def predict_test(self):

        # Make a prediction on entire training set
        self.y_pred = self.model_gscv.best_estimator_.predict(self.X_test)
        self.score = np.sqrt(mean_squared_error(y_true=self.y_test, y_pred=self.y_pred))
        logger.info(f'RMSE Score: {np.round(self.score,3)}')
        logger.info(f'-------------------------------')
if __name__ == '__main__':
    print('Running regression model')
    # df_data1 = pd.read_pickle('data/df_1.pk')
    # model1 = RegressionModel(df_data=df_data1,model_type='ridge')
    #
    # df_data2 = pd.read_pickle('data/df_2.pk')
    # model2 = RegressionModel(df_data=df_data2,model_type='ridge')
    #
    # df_data3 = pd.read_pickle('data/df_3.pk')
    # model3 = RegressionModel(df_data=df_data3,model_type='ridge')
