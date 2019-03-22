import models.model_classification as class_obj
import models.model_regression as reg_obj
import pandas as pd
import time
from loguru import logger

# Start a log filter
logger.add(f'logs/model_performance.log')

def run_all_model_class_models(models):
    '''
    Looks through each phase and runs all the models user specifies
    models should be a list of model names availble

    Example = models = ['logistic_regression', 'gaussian_naive_bayes',\
    'multinomial_naive_bayes','random_forest','xgboost']
    '''
    phases = ['I','II','III']

    logger.info('***Classification Models***')
    for idx,phase in enumerate(phases):
        for model_type in models:
            logger.info(f'Working Phase {phase} - {model_type}')
            df_data = pd.read_pickle('data/df_'+str(idx+1)+'.pk')

            model = class_obj.ClassificationModel(
                        df_data=df_data,
                        model_type=model_type,
                        )

def run_all_model_regress_models(models):
    '''
    Looks through each phase and runs all the models user specifies
    models should be a list of model names availble

    Example = models = ['logistic_regression', 'gaussian_naive_bayes',\
    'multinomial_naive_bayes','random_forest','xgboost']
    '''
    phases = ['I','II','III']
    logger.info('***Regression Models***')
    for idx,phase in enumerate(phases):
        for model_type in models:
            logger.info(f'Working Phase {phase} - {model_type}')
            df_data = pd.read_pickle('data/df_'+str(idx+1)+'.pk')

            model = reg_obj.RegressionModel(
                        df_data=df_data,
                        model_type=model_type,
                        )

if __name__ == '__main__':
    class_models = ['logistic_regression', 'gaussian_naive_bayes',\
    'multinomial_naive_bayes','random_forest','xgboost']
    run_all_model_class_models(class_models)

    reg_models = ['rfregression', 'gbreg','xgbreg']
    #reg_models = ['ols', 'lasso','ridge']
    run_all_model_regress_models(reg_models)
