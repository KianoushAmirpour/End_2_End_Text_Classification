from xgboost_model.constants import COLUMNS_FOR_MODEL
from xgboost_model.feature_generation import generate_features
from xgboost_model.predict import make_prediction


def test_features_generation(sample_input):
    features_dataframe = generate_features(sample_input)
    features_dataframe_cols = features_dataframe.columns.to_list()
    assert features_dataframe.shape[1] == 7
    assert features_dataframe_cols == COLUMNS_FOR_MODEL


def test_make_prediction(sample_input):
    results = make_prediction(sample_input)
    predictions = results['predictions']
    errors = results['errors']
    assert len(predictions) == 19
    assert isinstance(predictions, list)
    assert all(pred in ['insincere', 'sincere'] for pred in predictions)
    assert errors is None
