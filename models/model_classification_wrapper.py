import models.model_classification as class_obj
import pandas as pd
import time
from loguru import logger

# Start a log filter
logger.add(f'logs/classificationmodel.log')

def run_all_model_class_models(models):
    '''
    Looks through each phase and runs all the models user specifies
    models should be a list of model names availble

    Example = models = ['logistic_regression', 'gaussian_naive_bayes',\
    'multinomial_naive_bayes','random_forest','xgboost']
    '''
    phases = ['I','II','III']

    for idx,phase in enumerate(phases):
        for model_type in models:
            logger.info(f'Working Phase {phase} - {model_type}')
            df_data = pd.read_pickle('data/df_'+str(idx+1)+'.pk')
            model = class_obj.ClassificationModel(
                        df_data=df_data,
                        model_type=model_type,
                        )
            #time.sleep(1)
if __name__ == '__main__':
    models = ['logistic_regression', 'gaussian_naive_bayes',\
    'multinomial_naive_bayes','random_forest','xgboost']
    # , 'gaussian_naive_bayes','multinomial_naive_bayes', \
    # 'random_forest']
    run_all_model_class_models(models)
