
import mlflow
import numpy as np
from src.trainer import Trainer
from src.feast_utils import FeatureRetriever
from hyperopt import hp
from hyperopt.pyll.base import scope
from src.constants import *
import pandas as pd


def run(configs, features):

    fs = FeatureRetriever()
    
    training_df = fs.retreive_features(features)
    
    trainer = Trainer(df=training_df,
                      experiment_config=configs,
                      )
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(configs['experiment_name'])

    with mlflow.start_run():

        result = trainer.train()

        mlflow.set_tags(result['mlflow_tags'])

        mlflow.log_params(result['mlflow_tags']['best_params'])

        mlflow.log_metrics(result['metrics'])
        

if __name__ == "__main__":
   
    configs = {
        'experiment_name': 'xgboost_experiment',
        'train_with_tuning': True,
        'model_name': 'xgboost',
        'preprocessor_method': 'standard_scaler',
        'tuning_method': 'hyperopt',
        'model_params': {
                        'max_depth': hp.choice("max_depth", np.arange(1,20,1,dtype=int)),
                        'eta': hp.uniform("eta", 0, 1),
                        'gamma': hp.uniform("gamma", 0, 10e1),
                        'colsample_bytree': hp.uniform("colsample_bytree", 0.5,1),
                        'colsample_bynode': hp.uniform("colsample_bynode", 0.5,1), 
                        'colsample_bylevel': hp.uniform("colsample_bylevel", 0.5,1),
                        'n_estimators': hp.choice("n_estimators", np.arange(100,1000,10,dtype='int')),
                        'seed' : 44
                         }}
    
    
    features = ["meta_features_extracted_from_text:num_words",
                "meta_features_extracted_from_text:num_unique_words",
                "meta_features_extracted_from_text:num_stop_words",
                "meta_features_extracted_from_text:num_title_case",
                "meta_features_extracted_from_text:ave_length_words",
                "meta_features_extracted_from_text:num_characters",
                ]
        
    run(configs, features)
