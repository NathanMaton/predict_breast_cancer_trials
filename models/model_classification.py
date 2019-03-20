
'''
Scipt builds a logistic regression model for a binary classification

'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# ML Models
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler



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
        model_dict = {
            'logistic_regression':self.logistic_regression_model,
            'gaussian_naive_bayes':self.gaussian_naive_bayes_model,
            'multinomial_naive_bayes':self.multinomial_naive_bayes_model,
            'random_forest':self.random_forest_model,
            'xgboost':self.xgboost_model,
            }
        model_dict[model_type]()
        #Load this model
        # try:
        #     model_dict[model_type]()
        # except:
        #     print('Please enter in a valid model type')
        #     return



    def wrapper(self):
        self.test_train_split()
        self.grid_search(cv=5)

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
        label_col_name = df_data.filter(regex='trial length').columns[0]
        self.df_data[label_col_name] = self.df_data[label_col_name].apply(self.timedelta_change)
        self.df_data.fillna(0,inplace=True)


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
            print('More than one pass label column - adjust column names')
            return

        # Pulls the column name
        label_col_name = self.df_data.filter(regex='Pass').columns[0]

        # Change data type to all floats
        self.df_data = df_data.astype(dtype='float')

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
        n_estimators = n_estimators = np.arange(start=1, stop=4, step=1)
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 20, num=11)]
        max_depth = np.array(max_depth)
        # Minimum number of samples required to split a node
        min_samples_split = np.array([2, 5, 10])
        # Minimum number of samples required at each leaf node
        min_samples_leaf = np.array([1, 2, 4])


        # Create the random grid
        self.param_grid = {'RF__n_estimators': n_estimators,
                           'RF__max_depth': max_depth,
                           'RF__min_samples_split': min_samples_split,
                           'RF__min_samples_leaf': min_samples_leaf}
    # MAKE THE PARAMETERS
    def xgboost_model(self):
            '''
            Builds a xgboost model

            Parameters for Tree Booster
            eta [default=0.3, alias: learning_rate]

            Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
            range: [0,1]
            gamma [default=0, alias: min_split_loss]

            Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
            range: [0,∞]
            max_depth [default=6]

            Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
            range: [0,∞] (0 is only accepted in lossguided growing policy when tree_method is set as hist)
            min_child_weight [default=1]

            Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.
            range: [0,∞]
            max_delta_step [default=0]

            Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
            range: [0,∞]
            subsample [default=1]

            Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.
            range: (0,1]
            colsample_bytree, colsample_bylevel, colsample_bynode [default=1] - This is a family of parameters for subsampling of columns. - All colsample_by* parameters have a range of (0, 1], the default value of 1, and

            specify the fraction of columns to be subsampled.

            colsample_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
            colsample_bylevel is the subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree.
            colsample_bynode is the subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. Columns are subsampled from the set of columns chosen for the current level.
            colsample_by* parameters work cumulatively. For instance, the combination {'colsample_bytree':0.5, 'colsample_bylevel':0.5, 'colsample_bynode':0.5} with 64 features will leave 4 features to choose from at each split.
            lambda [default=1, alias: reg_lambda]

            L2 regularization term on weights. Increasing this value will make model more conservative.
            alpha [default=0, alias: reg_alpha]

            L1 regularization term on weights. Increasing this value will make model more conservative.
            tree_method string [default= auto]

            The tree construction algorithm used in XGBoost. See description in the reference paper.
            XGBoost supports hist and approx for distributed training and only support approx for external memory version.
            Choices: auto, exact, approx, hist, gpu_exact, gpu_hist
            auto: Use heuristic to choose the fastest method.
            For small to medium dataset, exact greedy (exact) will be used.
            For very large dataset, approximate algorithm (approx) will be chosen.
            Because old behavior is always use exact greedy in single machine, user will get a message when approximate algorithm is chosen to notify this choice.
            exact: Exact greedy algorithm.
            approx: Approximate greedy algorithm using quantile sketch and gradient histogram.
            hist: Fast histogram optimized approximate greedy algorithm. It uses some performance improvements such as bins caching.
            gpu_exact: GPU implementation of exact algorithm.
            gpu_hist: GPU implementation of hist algorithm.
            sketch_eps [default=0.03]

            Only used for tree_method=approx.
            This roughly translates into O(1 / sketch_eps) number of bins. Compared to directly select number of bins, this comes with theoretical guarantee with sketch accuracy.
            Usually user does not have to tune this. But consider setting to a lower number for more accurate enumeration of split candidates.
            range: (0, 1)
            scale_pos_weight [default=1]

            Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative instances) / sum(positive instances). See Parameters Tuning for more discussion. Also, see Higgs Kaggle competition demo for examples: R, py1, py2, py3.
            updater [default= grow_colmaker,prune]

            A comma separated string defining the sequence of tree updaters to run, providing a modular way to construct and to modify the trees. This is an advanced parameter that is usually set automatically, depending on some other parameters. However, it could be also set explicitly by a user. The following updater plugins exist:
            grow_colmaker: non-distributed column-based construction of trees.
            distcol: distributed tree construction with column-based data splitting mode.
            grow_histmaker: distributed tree construction with row-based data splitting based on global proposal of histogram counting.
            grow_local_histmaker: based on local histogram counting.
            grow_skmaker: uses the approximate sketching algorithm.
            sync: synchronizes trees in all distributed nodes.
            refresh: refreshes tree’s statistics and/or leaf values based on the current data. Note that no random subsampling of data rows is performed.
            prune: prunes the splits where loss < min_split_loss (or gamma).
            In a distributed setting, the implicit updater sequence value would be adjusted to grow_histmaker,prune by default, and you can set tree_method as hist to use grow_histmaker.
            refresh_leaf [default=1]

            This is a parameter of the refresh updater plugin. When this flag is 1, tree leafs as well as tree nodes’ stats are updated. When it is 0, only node stats are updated.
            process_type [default= default]

            A type of boosting process to run.
            Choices: default, update
            default: The normal boosting process which creates new trees.
            update: Starts from an existing model and only updates its trees. In each boosting iteration, a tree from the initial model is taken, a specified sequence of updater plugins is run for that tree, and a modified tree is added to the new model. The new model would have either the same or smaller number of trees, depending on the number of boosting iteratons performed. Currently, the following built-in updater plugins could be meaningfully used with this process type: refresh, prune. With process_type=update, one cannot use updater plugins that create new trees.
            grow_policy [default= depthwise]

            Controls a way new nodes are added to the tree.
            Currently supported only if tree_method is set to hist.
            Choices: depthwise, lossguide
            depthwise: split at nodes closest to the root.
            lossguide: split at nodes with highest loss change.
            max_leaves [default=0]

            Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set.
            max_bin, [default=256]

            Only used if tree_method is set to hist.
            Maximum number of discrete bins to bucket continuous features.
            Increasing this number improves the optimality of splits at the cost of higher computation time.
            predictor, [default=``cpu_predictor``]

            The type of predictor algorithm to use. Provides the same results but allows the use of GPU or CPU.
            cpu_predictor: Multicore CPU prediction algorithm.
            gpu_predictor: Prediction using GPU. Default when tree_method is gpu_exact or gpu_hist.
            num_parallel_tree, [default=1] - Number of parallel trees constructed during each iteration. This option is used to support boosted random forest.
            '''
            self.model = RandomForestClassifier(
                            random_state=42,
                            class_weight="balanced",
                            bootstrap = True,
                            max_features = 'auto')

            self.pipe = Pipeline(
                        steps=[
                            #('scale', StandardScaler()),
                            ('XGB', self.model)
                            ])

            # Hyperparameter space for RF
            # Number of trees in random forest
            n_estimators = n_estimators = np.arange(start=1, stop=4, step=1)
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 20, num=11)]
            max_depth = np.array(max_depth)
            # Minimum number of samples required to split a node
            min_samples_split = np.array([2, 5, 10])
            # Minimum number of samples required at each leaf node
            min_samples_leaf = np.array([1, 2, 4])


            # Create the random grid
            self.param_grid = {'RF__n_estimators': n_estimators,
                               'RF__max_depth': max_depth,
                               'RF__min_samples_split': min_samples_split,
                               'RF__min_samples_leaf': min_samples_leaf}


    def grid_search(self,cv=5):


        # Performs a grid search for the model
        self.model_gscv = GridSearchCV(self.pipe, param_grid=self.param_grid, iid=False, cv=cv,
                              return_train_score=False)
        self.model_gscv.fit(self.X_train, self.y_train)


        print('Best model')
        print(f"CV score {self.model_gscv.best_score_}")
        print(f"Best parameters: {self.model_gscv.best_params_}")

        if self.model_type=='logistic_regression':
            LR_coef = np.exp(self.model_gscv.best_estimator_.steps[1][1].coef_)
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
    model = ClassificationModel(df_data=df_data,model_type='logistic_regression')
    model.wrapper()

    #model.model_gscv.best_estimator_.steps[1][1].coef_
