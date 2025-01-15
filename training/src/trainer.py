import pandas as pd
from typing import Dict, Any, Tuple
from functools import partial
import numpy as np
from sklearn import metrics
from hyperopt import Trials, tpe
from .logger import setup_logger
from .metrics import MetricsCollector
from .config_validation import validate_experiment_config
from .registry import RegisterModel, RegisterPreprocessor, RegisterHpTuner
from .utils import create_folds, set_kfold_strategy, refit_strategy, create_train_test_split
from .dataset import DistilBertDataset
from torch.utils.data import DataLoader
from .models import DistilBertModel
from .engine import train_one_epoch, evaluate_one_epoch
from torch.optim import Adam
from .constants import (
    KFOLD_COLUMN,
    COLUMNS_TO_DROP_KFOLD,
    TARGET_COLUMN,
    NUMBER_FOLDS,
    HyperparameterTuningMethods,
    COLUMNS_TO_DROP_TUNING,
    HYPEROPT_MAX_EVALS,
    HP_TUNING_SCORING,
    NUMBER_OF_JOBS,
    SKLEARN_MODELS_RANDOM_STATE,
    HP_TUNING_N_ITER, 
    TRAIN_BATCH_SIZE,
    VALID_BATCH_SIZE, 
    LEARNING_RATE,
    DEVICE,
    NUMBER_EPOCHS
)

logger = setup_logger(__name__)

class Trainer:
    def __init__(self, df: pd.DataFrame, experiment_config: Dict[str, Any]):
        self.df = df
        self.experiment_config = experiment_config
        self.metrics = MetricsCollector()
        self.model = None
        self.preprocessor = None
        self.tuner = None
        self.response = {
            'mlflow_tags': {
                'model': self.experiment_config['model_name'],
                'preprocessor': self.experiment_config['preprocessor_method'],
                'best_params': self.experiment_config['model_params'],
                'tuning_method': self.experiment_config['tuning_method']},
            'metrics': None
        }

    def _prepare_fold_data(self, training_df: pd.DataFrame, fold: int):
        df_train = training_df[training_df[KFOLD_COLUMN] != fold]
        df_validation = training_df[training_df[KFOLD_COLUMN] == fold]

        x_train = df_train.drop(COLUMNS_TO_DROP_KFOLD, axis=1).values
        y_train = df_train[TARGET_COLUMN].values

        x_validation = df_validation.drop(COLUMNS_TO_DROP_KFOLD, axis=1).values
        y_validation = df_validation[TARGET_COLUMN].values
                    
        return x_train, y_train, x_validation, y_validation
    
    def _prepare_dataloaders(self):
        df_train, df_valid = create_train_test_split(self.df)
        train_dataset = DistilBertDataset(post=df_train["post"].values,
                                          label=df_train["label"].values
                                          )
        valid_dataset = DistilBertDataset(post=df_valid["post"].values,
                                          label=df_valid["label"].values
                                          )
        train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True)
        return train_dataloader, valid_dataloader
        
    
    def _train_fold(self, x_train, y_train, x_validation, y_validation):
        x_train = self.preprocessor.fit_transform(x_train)
        x_validation = self.preprocessor.transform(x_validation)

        self.model.fit(x_train, y_train)
        y_train_pred = self.model.predict(x_train)
        y_validation_pred = self.model.predict(x_validation)
        y_validation_prob = self.model.predict_proba(x_validation)[:, 1]

        self.metrics.cal_train_metrics(y_train, y_train_pred)
        self.metrics.cal_validation_metrics(y_validation, y_validation_pred)
        self.metrics.cal_roc_auc(y_validation, y_validation_prob)

    def train_without_tuning(self) -> Dict:
        training_df = create_folds(self.df)
        for fold in range(1, NUMBER_FOLDS + 1):
            x_train, y_train, x_validation, y_validation = self._prepare_fold_data(
                training_df, fold)
            self._train_fold(x_train, y_train, x_validation, y_validation)
            logger.info(f'Training completed for fold {fold}')
        self.response['metrics'] = self.metrics.aggregate()

        return self.response
        
    def _optimize_func_for_hyperopt(self, params, x_train, y_train) -> float:
        self.model.set_params(**params)
        skf = set_kfold_strategy()
        roc_auc_score = []

        for train_idx, test_idx in skf.split(X=x_train, y=y_train):
            x_train_fold, x_test_fold = x_train[train_idx], x_train[test_idx]
            y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]
            
            x_train_fold = self.preprocessor.fit_transform(x_train_fold)
            x_test_fold = self.preprocessor.transform(x_test_fold)

            self.model.fit(x_train_fold, y_train_fold)
            preds = self.model.predict_proba(x_test_fold)[:, 1]
            roc_auc_score.append(metrics.roc_auc_score(y_test_fold, preds))

        return -1 * np.mean(roc_auc_score)
    
    def train_with_tuning(self) -> Dict:
        if self.experiment_config['tuning_method'] == HyperparameterTuningMethods.HYPEROPT:
            best = self.tuner
            self.response['mlflow_tags']['best_params'] = best
            self.response['metrics'] = {'roc_auc': -1 * max(self.trials.losses())}
            return self.response
        else:
            x_train = self.df.drop(COLUMNS_TO_DROP_TUNING, axis=1).values
            y_train = self.df[TARGET_COLUMN].values

            self.tuner.fit(x_train, y_train)
            cv_results = self.tuner.cv_results_
            best_index = refit_strategy(cv_results)
            best_params = cv_results['params'][best_index]

            best_metrics = {
                "accuracy_mean": cv_results["mean_test_accuracy"][best_index],
                "accuracy_std": cv_results["std_test_accuracy"][best_index],
                "precision_mean": cv_results["mean_test_precision"][best_index],
                "precision_std": cv_results["std_test_precision"][best_index],
                "recall_mean": cv_results["mean_test_recall"][best_index],
                "recall_std": cv_results["std_test_recall"][best_index],
                "f1_score_mean": cv_results["mean_test_f1"][best_index],
                "f1_score_std": cv_results["std_test_f1"][best_index],
                "roc_auc_mean": cv_results["mean_test_roc_auc"][best_index],
                "roc_auc_std": cv_results["std_test_roc_auc"][best_index],
            }
            self.response['mlflow_tags']['best_params'] = best_params
            self.response['metrics'] = best_metrics
            return self.response

    def train_distilbert(self):
        self.model = DistilBertModel()
        self.model.to(DEVICE)
        train_dataloader, valid_dataloader = self._prepare_dataloaders()
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE) # ????
        for epoch in range(1 , NUMBER_EPOCHS + 1):
            train_loss_epoch, train_metrics = train_one_epoch(train_dataloader, self.model, self.optimizer, DEVICE, epoch)
            valid_loss_epoch, valid_metrics = evaluate_one_epoch(valid_dataloader, self.model, DEVICE, epoch)
            logger.info(f"Epoch: {epoch} | Ave_Training_Loss: {train_loss_epoch} | Ave_Validation_Loss: {valid_loss_epoch}")
            self.metrics.train_acccuracy.extend(train_metrics['accuracy'])
            self.metrics.validation_accuracy.extend(valid_metrics['accuracy'])
            self.metrics.precision.extend(valid_metrics['precision'])
            self.metrics.recall.extend(valid_metrics['recall'])
            self.metrics.f1_score.extend(valid_metrics['f1'])
            print(self.metrics.train_acccuracy, self.metrics.validation_accuracy, self.metrics.precision, self.metrics.recall, self.metrics.f1_score)
        self.response['metrics'] = self.metrics.aggregate()
        return self.response
            

    def _initialize_model(self):
        self.model = RegisterModel().instantiate_model(
            self.experiment_config['model_name'],
            self.experiment_config.get('model_params', {}),
        )

    def _initialize_preprocessor(self):
        self.preprocessor = RegisterPreprocessor().instantiate_preprocessor(
            self.experiment_config['preprocessor_method']
        )

    def _initialize_tuner(self):
        if self.experiment_config['tuning_method'] == HyperparameterTuningMethods.HYPEROPT:
            x_train = self.df.drop(COLUMNS_TO_DROP_TUNING, axis=1).values
            y_train = self.df[TARGET_COLUMN].values
            self.trials = Trials()
            self.tuner = RegisterHpTuner().instantiate_tuner(
                self.experiment_config['tuning_method'],
                space=self.experiment_config['model_params'],
                fn=partial(self._optimize_func_for_hyperopt, x_train=x_train, y_train=y_train),
                trials=self.trials,
                algo=tpe.suggest,
                max_evals=HYPEROPT_MAX_EVALS
            )

        elif self.experiment_config['tuning_method'] == HyperparameterTuningMethods.RANDOM_SEARCH:
            self.tuner = RegisterHpTuner().instantiate_tuner(
                self.experiment_config['tuning_method'],
                estimator=self.model,
                scoring=HP_TUNING_SCORING,
                param_distributions=self.experiment_config['model_params'],
                refit=False,
                cv=set_kfold_strategy(),
                n_jobs=NUMBER_OF_JOBS,
                n_iter=HP_TUNING_N_ITER,
                random_state=SKLEARN_MODELS_RANDOM_STATE
            )

    def train(self):
        validate_experiment_config(self.experiment_config)
        if self.experiment_config['model_name'] == 'distilbert':
            return self.train_distilbert()
        elif self.experiment_config['train_with_tuning']:

            self._initialize_model()
            self._initialize_preprocessor()
            self._initialize_tuner()
            return self.train_with_tuning()
        else:
            self._initialize_model()
            self._initialize_preprocessor()
            return self.train_without_tuning()
