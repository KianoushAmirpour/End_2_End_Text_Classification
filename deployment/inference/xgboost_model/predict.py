from typing import Any, Dict

import pandas as pd  # type: ignore

from xgboost_model.constants import ARTIFACTS_ROOT, ROOT_DIR  # type: ignore
from xgboost_model.feature_generation import generate_features  # type: ignore
from xgboost_model.utils import load_artifact, map_output  # type: ignore
from xgboost_model.validation import validate_input  # type: ignore

with open(ROOT_DIR / "VERSION") as version_file:
    _version = version_file.read().strip()

model = load_artifact(ARTIFACTS_ROOT / f'model_{_version}.pkl')
scalar = load_artifact(ARTIFACTS_ROOT / f'preprocesssor_{_version}.b')


def make_prediction(input_data: pd.DataFrame) -> Dict[str, Any]:
    features_dataframe = generate_features(input_data)
    features_dataframe.drop(columns=['id'], inplace=True)
    features_dataframe.reset_index(drop=True, inplace=True)
    validated_features, errors = validate_input(features_dataframe)
    if not errors:
        scaled_data = scalar.transform(validated_features.to_numpy())
        prediction = model.predict(scaled_data)
        results = {"predictions": map_output(prediction),
                   'version': _version, "errors": errors}
        return results
    return {'predictions': [], 'version': _version, "errors": errors}
