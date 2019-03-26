
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score
from scripts.model_classification import ClassificationModel

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
            'rfregression':self.rfregression,
            'gbreg':self.gbreg,
            'xgbreg':self.xgbreg,
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
        # Linear Regression - Lasso
        self.model = linear_model.Lasso(
            alpha=.5,
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
            'lasso__alpha': np.logspace(-4,4,20),
            'lasso__fit_intercept': [True, False],
            'lasso__normalize': [True, False],
            }

    def ridge(self):
        '''
        Build a lasso model and defines
        the grid search parameters space

        '''
        # Linear Regression - Ridge
        self.model = linear_model.Ridge(
            alpha=.5,
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
            'ridge__alpha': np.logspace(-4,4,20),
            'ridge__fit_intercept': [True, False],
            'ridge__normalize': [True, False],
            }

    def rfregression(self):
        '''
        Build a rf regressor and defines
        the grid search parameters space

        '''

        self.model = RandomForestRegressor(
            max_depth=2,
            n_estimators=100,
            random_state=42,
            bootstrap=True
            )

        # Builds pipe. Apply a standard scaler to features
        self.pipe = Pipeline(
                    steps=[
                        ('scale', StandardScaler()),
                        ('rfregression', self.model)
                    ])

        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        self.param_grid = {
            'rfregression__max_depth': [2,4,6,8,10,20,40],
            #'rfregression__n_estimators': [100,500,1000],
            'rfregression__n_estimators': [20,40,50,60,70,80,100,300],
            }

    def gbreg(self):
        '''
        Build a xg regressor and defines
        the grid search parameters space
        '''
        # LogisticRegression, balanced is used because classes are highly
        # imbalanced
        self.model = GradientBoostingRegressor(
            loss='huber',
            learning_rate=.1,
            n_estimators=100,
            random_state=42,
            verbose=4
            )

        # Builds pipe. Apply a standard scaler to features
        self.pipe = Pipeline(
                    steps=[
                        ('scale', StandardScaler()),
                        ('xgreg', self.model)
                    ])

        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        # self.param_grid = {
        #
        #     'gbreg__loss': ['huber','ls','lad'],
        #     'gbreg__learning_rate':[.001,.005,.01,.05,.1],
        #     'gbreg__n_estimators':[40,60,80,100,200,400,600,800,1000],
        #     }
        self.param_grid = {

            'gbreg__loss': ['huber'],
            'gbreg__learning_rate':[.1],
            'gbreg__n_estimators':[100],
            }

    def xgbreg(self):
        '''
        Build a xg regressor and defines
        the grid search parameters space
        '''
        # LogisticRegression, balanced is used because classes are highly
        # imbalanced
        self.model = XGBRegressor(

            max_depth=5,
            learning_rate=.1,
            n_estimators=100,
            random_state=42,
            verbose=2,
            nthreads=-1
            )

        # Builds pipe. Apply a standard scaler to features
        self.pipe = Pipeline(
                    steps=[
                        #('scale', StandardScaler()),
                        ('xgbreg', self.model)
                    ])

        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        self.param_grid = {
               'xgbreg__colsample_bytree':np.arange(start=.4,stop=.9,step=.1),
               'xgbreg__gamma': np.arange(start=0,stop=10,step=1),
               'xgbreg__min_child_weight':[1.5],
               'xgbreg__learning_rate':np.arange(start=.01,stop=.08,step=.02),
               'xgbreg__max_depth':np.arange(start=2,stop=5,step=1),
               'xgbreg__n_estimators':[40,50,70,80,90,100,200],
               'xgbreg__reg_alpha':[1e-2],
               'xgbreg__reg_lambda':[1e-2],
               'xgbreg__subsample':np.arange(start=.6,stop=.8,step=.1),
               'xgbreg__scale_pos_weight':[1,5,9,11]
                }

    def predict_test(self):

        # Make a prediction on entire training set
        self.y_pred = self.model_gscv.best_estimator_.predict(self.X_test)
        self.score = np.sqrt(mean_squared_error(y_true=self.y_test, y_pred=self.y_pred))
        logger.info(f'RMSE Score: {np.round(self.score,3)}')
        logger.info(f'----------------------------------')

if __name__ == '__main__':
    print('Running regression model')
    df_data1 = pd.read_pickle('data/df_1.pk')
    model1 = RegressionModel(df_data=df_data1,model_type='ols')
    df_data1.columns

    #['Phase I Completed', 'Phase I Discontinued',' Phase I Other Trial Status']

    #
    # df_data2 = pd.read_pickle('data/df_2.pk')
    # model2 = RegressionModel(df_data=df_data2,model_type='ridge')
    #
    # df_data3 = pd.read_pickle('data/df_3.pk')
    # model3 = RegressionModel(df_data=df_data3,model_type='ridge')
