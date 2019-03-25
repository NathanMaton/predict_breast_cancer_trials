
'''
Scipt builds a logistic regression model for a multi classification

'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, log_loss, recall_score, f1_score

# ML Models
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler

# import logger
from loguru import logger


# Start a log filter
#logger.add(f'logs/model_performance.log')

class TrialTimeModel():

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
            'logistic_regression':self.logistic_regression_model,
            'gaussian_naive_bayes':self.gaussian_naive_bayes_model,
            'multinomial_naive_bayes':self.multinomial_naive_bayes_model,
            'random_forest':self.random_forest_model,
            'xgboost':self.xgboost_model,
            }


        #Load this model
        try:
            model_dict[model_type]()
            #logger.info(f'Running: {model_type}')
        except:
            print('Please enter in a valid model type')
            raise
            return

        # Runs the script
        self.wrapper()


    def wrapper(self):
        self.test_train_split()
        self.grid_search(cv=6)
        self.predict_test()

    def timedelta_change(self,x):
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
        self.label_col_name = self.df_data.filter(regex='trial length').columns[0]
        self.df_data[self.label_col_name ] = self.df_data[self.label_col_name ].apply(self.timedelta_change)
        # Bins the data into 3 separate bins of time
        self.df_data[self.label_col_name] = self.df_data[self.label_col_name].apply(self.turn_length_to_bin)
        self.df_data.fillna(0,inplace=True)



    def turn_length_to_bin(self, x):
        if x < 365*1.5:
            return 0
        if x < 365*3:
            return 1
        else:
            return 2

    def timedelta_change(self, x):
        # Applies a change to the days to a float, there's a mix of datetime Objects and floats in the dataframe column length
        try:
            y = x.days
        except:
            y = x
        return y


    def drop_bad_columns(self):
        bad_columns = 'Completed|Discontinued|Other Trial Status'
        bad_column_names = self.df_data.filter(regex=bad_columns).columns
        self.df_data = self.df_data.drop([bad_column_names],axis=1)



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
        if self.df_data.filter(regex='Pass').shape[1] != 1:
            logger.info('More than one pass label column - adjust column names')
            return


        # Change data type to all floats
        self.df_data = self.df_data.astype(dtype='float')

        # Define train and test data
        X = self.df_data.drop(self.label_col_name,axis=1)
        y = self.df_data[self.label_col_name]

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
            class_weight='balanced'
            )

        # Builds pipe. Apply a standard scaler to features
        self.pipe = Pipeline(
                    steps=[
                        ('scale', StandardScaler()),
                        ('logistic', self.model)
                    ])

        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        self.param_grid = {
            'logistic__penalty': ['l2'],
            'logistic__C': np.logspace(-4, 1, 10),
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
                        #('scale', StandardScaler()),
                        ('multinominalNB', self.model)
                    ])
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        self.param_grid = {}

    def random_forest_model(self):
        '''
        Builds a random forest model
        '''
        self.model = RandomForestClassifier(
                        random_state=42,
                        class_weight="balanced",
                        bootstrap = True,
                        max_features = 'auto')

        self.pipe = Pipeline(
                    steps=[
                        #('scale', StandardScaler()),
                        ('RF', self.model)
                        ])

        # Hyperparameter space for RF
        # Number of trees in random forest
        n_estimators = [20,40,60,80,100,500]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 20, num=20)]
        max_depth = np.array(max_depth)
        # Minimum number of samples required to split a node
        min_samples_split = np.array([2, 5, 8, 10])
        # Minimum number of samples required at each leaf node
        min_samples_leaf = np.array([2,4,6,8,10])

        # # Test parameters for prototyping
        # n_estimators = [20]
        # # Maximum number of levels in tree
        # max_depth = [int(x) for x in np.linspace(10, 20, num=2)]
        # max_depth = np.array(max_depth)
        # # Minimum number of samples required to split a node
        # min_samples_split = np.array([2])
        # # Minimum number of samples required at each leaf node
        # min_samples_leaf = np.array([2])



        # Create the random grid
        self.param_grid = {'RF__n_estimators': n_estimators,
                           'RF__max_depth': max_depth,
                           'RF__min_samples_split': min_samples_split,
                           'RF__min_samples_leaf': min_samples_leaf}
    # MAKE THE PARAMETERS
    def xgboost_model(self):
            '''
            Builds a xgboost model
            '''
            self.model = XGBClassifier(
                            seed=42,
                            nthreads=-1,
                            objective = 'binary:logistic',
                            )

            self.pipe = Pipeline(
                        steps=[
                            #('scale', StandardScaler()),
                            ('XGB', self.model)
                            ])

            self.param_grid = {
                           'XGB__colsample_bytree':np.arange(start=.4,stop=.9,step=.2),
                           'XGB__gamma': np.arange(start=0,stop=10,step=2),
                           'XGB__min_child_weight':np.logspace(-2,2,10),
                           'XGB__learning_rate':np.logspace(-3,-1,10),
                           'XGB__max_depth':np.arange(start=3,stop=7,step=1),
                           'XGB__n_estimators':[40,50,70,80,90,100,200,500],
                           'XGB__reg_alpha':np.logspace(-2,3,10),
                           'XGB__reg_lambda':np.logspace(-2,3,10),
                           'XGB__subsample':np.arange(start=.6,stop=.8,step=.1),
                           'XGB__scale_pos_weight':[1,5,9,11]
                           }
            # #Test parameters small set for prototyping
            # self.param_grid = {
            #                'XGB__colsample_bytree':[.1],
            #                'XGB__gamma': [4],
            #                'XGB__min_child_weight':[1.5],
            #                'XGB__learning_rate':[.2],
            #                'XGB__max_depth':[3],
            #                'XGB__n_estimators':[100],
            #                'XGB__reg_alpha':[1e-2],
            #                'XGB__reg_lambda':[1e-2],
            #                'XGB__subsample':[.5],
            #                'XGB__scale_pos_weight':[9]
            #                }

    def grid_search(self,cv=10):

        # Performs a grid search for the model
        self.model_gscv = GridSearchCV(self.pipe, param_grid=self.param_grid, iid=False, cv=cv,
                              return_train_score=False,verbose=10,n_jobs=-1)
        self.model_gscv.fit(self.X_train, self.y_train)

        logger.info(f"CV score {self.model_gscv.best_score_}")
        logger.info(f"Best parameters: {self.model_gscv.best_params_}")

        if self.model_type=='logistic_regression':
            LR_coef = np.exp(self.model_gscv.best_estimator_.steps[1][1].coef_)[0]
            features = self.X_train.columns.tolist()
            logger.info(f'Odds coefficients')
            logger.info(list(zip(features,np.round(LR_coef,3))))

    def naive_probs(self):
        naive_prob=np.histogram(self.y_train, density=True, bins=[-.5,.5,1.5,2.5])[0]
        self.naive_predict_prob=np.ones([self.y_test.shape[0],3])*naive_prob
    def predict_test(self):

        # Make a prediction on entire training set
        self.y_pred = self.model_gscv.best_estimator_.predict(self.X_test)
        self.predict_prob = self.model_gscv.best_estimator_.predict_proba(self.X_test)
        self.naive_probs()

        # Different score meaures
        self.accuracy_score = accuracy_score(y_true=self.y_test, y_pred=self.y_pred)
        self.precision_score = precision_score(y_true=self.y_test, y_pred=self.y_pred, average='weighted', labels=[0,1,2])
        self.recall_score = recall_score(y_true=self.y_test, y_pred=self.y_pred, average='weighted', labels=[0,1,2])
        self.f1_score = f1_score(y_true=self.y_test, y_pred=self.y_pred, average='weighted', labels=[0,1,2])

        # The average probability estimate
        # we put in the negative value since we multiplied by -1/N
        self.log_loss_score = np.exp(-1*log_loss(y_true=self.y_test, y_pred=self.predict_prob, labels=[0,1,2]))
        self.naive_log_loss_score = np.exp(-1*log_loss(y_true=self.y_test, y_pred=self.naive_predict_prob, labels=[0,1,2]))




        #self.naive_log_loss_score = np.exp(-1*log_loss(y_true=self.y_test, y_pred=self.naive_preds, labels=[0,1,2]))


        #self.precision_score = precision_score(y_true=self.y_test, y_pred=self.y_pred)
        logger.info(f'Accuracy Test Score: {np.round(self.accuracy_score,3)}')
        logger.info(f'Precision Test Score: {np.round(self.precision_score,3)}')
        logger.info(f'Recall Test Score: {np.round(self.recall_score,3)}')
        logger.info(f'F1 Test Score: {np.round(self.f1_score,3)}')
        logger.info(f'Log Loss Test Loss Score: {np.round(self.log_loss_score,3)}')
        logger.info(f'Naive Log Loss Test Loss Score: {np.round(self.naive_log_loss_score,3)}')

        logger.info(f'------------------------------------')
        #


if __name__ == '__main__':
    #historic probabilty for the phase
    df_data1 = pd.read_pickle('data/df_1.pk')
    df_data1.head()
    model1 = TrialTimeModel(df_data=df_data1,model_type='logistic_regression')

    model1.X_train.head()
    hist = model1.y_train.hist()
    p=np.histogram(model1.y_train, density=True, bins=[-.5,.5,1.5,2.5])[0]
    preds=np.ones([model1.y_test.shape[0],3])*p

    p
    preds=np.random.choice([0,1,2],size=model1.y_test.shape[0],p=p)
    preds.pop().pop().pop()
    log_loss(model1.y_test,preds)
    model1.y_test
    preds
    model1.y_train[model1.y_train==0].shape
    model1.y_train.shape
    model1.predict_prob.sum(axis=1)
    240/409
    #draw randomly length of y_test times with the historic prob Distribution, this is y_pred
