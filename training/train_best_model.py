from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from src.feast_utils import FeatureRetriever
from src.constants import MLFLOW_TRACKING_URI, PREPROCESSOR_DIR, BEST_MODELS_DIR
import mlflow
import pickle

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('xgboost_standard_scalar')

fs = FeatureRetriever()

features = ["meta_features_extracted_from_text:num_words",
                "meta_features_extracted_from_text:num_unique_words",
                "meta_features_extracted_from_text:num_stop_words",
                "meta_features_extracted_from_text:num_title_case",
                "meta_features_extracted_from_text:ave_length_words",
                "meta_features_extracted_from_text:num_characters"]

training_df = fs.retreive_features(features)

X = training_df.drop(columns=['id', 'event_timestamp', 'label'],axis=1).values
y = training_df['label'].values

scalar = StandardScaler()
X = scalar.fit_transform(X)

best_params = {"colsample_bylevel": 0.8571794476577662,
               "colsample_bynode": 0.6931924515201482,
               "colsample_bytree": 0.8273764755171993,
               "eta": 0.10728514144455947,
               "gamma": 61.29565230224352,
               "max_depth": 5,
               "n_estimators": 22,
               'seed': 44}

clf = XGBClassifier(objective='binary:logistic',  **best_params)

clf.fit(X,y)

with open(PREPROCESSOR_DIR, 'wb') as f:
    pickle.dump(scalar, f)
    
with open(BEST_MODELS_DIR / 'model.pkl', "wb") as f:
    pickle.dump(clf, f)

mlflow.log_artifact(PREPROCESSOR_DIR, "preprocessor")
mlflow.xgboost.log_model(clf, artifact_path = "models")
