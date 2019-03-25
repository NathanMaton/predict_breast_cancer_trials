import codes.model_classification as class_obj
import codes.model_regression as reg_obj
import codes.model_trial_time as trial_time_obj
import pandas as pd
import time
from loguru import logger



def run_class_models(models):
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

def run_regress_models(models):
    '''
    Looks through each phase and runs all the models user specifies
    models should be a list of model names availble

    Example: models = ['ols', 'lasso','ridge','rfregression','gbreg','xgbreg']

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

def run_trial_time_models(models):
    '''
    Looks through each phase and runs all the models user specifies
    models should be a list of model names availble

    Example = models = ['logistic_regression', 'gaussian_naive_bayes',\
    'multinomial_naive_bayes','random_forest','xgboost']
    '''
    phases = ['I','II','III']

    logger.info('***Trial Time Models***')
    for idx,phase in enumerate(phases):
        for model_type in models:
            logger.info(f'Working Phase {phase} - {model_type}')
            df_data = pd.read_pickle('data/df_'+str(idx+1)+'.pk')

            model = trial_time_obj.TrialTimeModel(
                        df_data=df_data,
                        model_type=model_type,
                        )

# if __name__ == '__main__':
def main():
    #
    # # Start a log filter
    logger.add(f'logs/model_performance.log')

<<<<<<< HEAD
    class_models = ['logistic_regression', 'gaussian_naive_bayes',\
    'multinomial_naive_bayes','random_forest','xgboost']
    class_models = ['logistic_regression']
    run_all_model_class_models(class_models)
    logger.info('Classication models complete!')

    # reg_models = ['ols', 'lasso','ridge','rfregression','xgbreg']
    # run_all_model_regress_models(reg_models)
    # logger.info('Regression models complete!')
=======
    class_models = ['logistic_regression']
    run_class_models(class_models)
    logger.info('Classication models complete!')

    # reg_models = ['ols', 'lasso','ridge','rfregression','xgbreg']
    # run_regress_models(reg_models)
    # logger.info('Regression models complete!')

    # trial_time_models = ['logistic_regression', 'gaussian_naive_bayes',\
    # 'multinomial_naive_bayes']
    # run_trial_time_models(trial_time_models)
    # logger.info('Trial time models complete!')
>>>>>>> 36dddd8c6d3a37af26653871a4e71ad6ffa4c270
