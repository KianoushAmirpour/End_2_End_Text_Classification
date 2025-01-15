from .logger import setup_logger
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
from typing import Tuple, List
import matplotlib.pyplot as plt
from .constants import (
        NUMBER_FOLDS,
        SKLEARN_MODELS_RANDOM_STATE,
        KFOLD_COLUMN,
        TARGET_COLUMN,
        TRAIN_TEST_SPLIT_SIZE
)


logger = setup_logger(__name__)

def set_kfold_strategy():
    kf = StratifiedKFold(n_splits=NUMBER_FOLDS, shuffle=True, random_state=SKLEARN_MODELS_RANDOM_STATE)
    logger.info(f'KFold strategy created successfully. Number of folds: {NUMBER_FOLDS}')
    return kf
    
def create_folds(df: pd.DataFrame) -> pd.DataFrame:
    df[KFOLD_COLUMN] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kf = set_kfold_strategy()
    for fold_num, (_, valid_idx) in enumerate(kf.split(X=df, y=df[TARGET_COLUMN].values), start=1):
        df.loc[valid_idx, KFOLD_COLUMN] = fold_num
    logger.info(f'Folds created successfully. Number of folds: {df.kfold.nunique()}')
    return df

def create_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_valid = train_test_split(
        df,
        test_size=TRAIN_TEST_SPLIT_SIZE,
        random_state=SKLEARN_MODELS_RANDOM_STATE,
        stratify=df[TARGET_COLUMN].values
    )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    return df_train, df_valid

def refit_strategy(cv_results):
    
    results = pd.DataFrame(cv_results)
    
    results = results[
        [
            "mean_score_time",
            'mean_test_accuracy',
            'std_test_accuracy',
            'mean_test_f1',
            'mean_test_f1',
            'mean_test_roc_auc',
            "std_test_roc_auc",
            "mean_test_recall",
            "std_test_recall",
            "mean_test_precision",
            "std_test_precision",
            "rank_test_recall",
            "rank_test_precision",
            "rank_test_f1",
            "rank_test_accuracy",
            "rank_test_roc_auc",
            "params",
        ]
    ]
    
    results = results.sort_values(by=["rank_test_roc_auc", 'rank_test_recall', "rank_test_precision"], ascending=False)
    
    # From the best candidates, select the fastest model to predict
    fastest_top_recall_high_precision_index = results[
        "mean_score_time"
    ].idxmin()

    return fastest_top_recall_high_precision_index
    
def plot_losses(train_loss:List[float], valid_loss:List[float], n_epochs: int):
    
    if len(train_loss) and len(valid_loss) != n_epochs:
        raise ValueError(f'Length of losses must match the number of epochs.')
    plt.plot(range(1, n_epochs+1), train_loss, label='training_loss', color='b')
    plt.plot(range(1, n_epochs+1), valid_loss, label='validation_loss', color='r')
    plt.legend()
    plt.show()