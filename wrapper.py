from codes import data_feature_extraction
from codes import model_wrapper
from loguru import logger

if __name__ == '__main__':
    # Start a log filter
    logger.add(f'logs/model_performance.log')
    # Performs all the feature extraction
    logger.info('Started Feature Exaction')
    data_feature_extraction.main()
    logger.info('Completed Feature Exaction')
    # Runs the model wrapper that will run each of the classification model
    logger.info('Started Model Wrapper')
    model_wrapper.main()
    logger.info('Completed Model Wrapper')
