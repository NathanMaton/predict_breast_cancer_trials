import models.model_classification as class_obj
import pandas as pd
import time
from loguru import logger

# Start a log filter
logger.add(f'logs/classificationmodel.log')

def run_all_model_phases(models,log_name='test'):
    '''
    Looks through each phase and runs all the models user specifies
    models should be a list of model names availble

    Example = models = ['logistic_regression', 'gaussian_naive_bayes',\
    'multinomial_naive_bayes','random_forest']
    '''
    phases = ['1','2','3']

    for phase in phases:
        for model_type in models:
            logger.info(f'Working Phase:{phase} Model:{model_type}')
            df_data = pd.read_pickle('data/df_'+phase+'.pk')
            model = class_obj.ClassificationModel(
                        df_data=df_data,
                        model_type=model_type,
                        )
            #time.sleep(1)
if __name__ == '__main__':
    models = ['logistic_regression']
    # , 'gaussian_naive_bayes','multinomial_naive_bayes', \
    # 'random_forest']
    run_all_model_phases(models)
