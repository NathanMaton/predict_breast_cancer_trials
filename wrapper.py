from codes import data_feature_extraction
from codes import model_wrapper
from loguru import logger

if __name__ == '__main__':
    # Start a log filter
    logger.add(f'logs/model_performance.log')
    # Performs all the feature extraction
    data_feature_extraction.main()
    # Runs the model wrapper that will run each of the classification model
    model_wrapper.main()
