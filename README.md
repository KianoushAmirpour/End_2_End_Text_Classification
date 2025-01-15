## Table of contents
1. [Description](#description)
2. [Dataset](#dataset)
3. [Feature store](#feature-store)
4. [Feature Engineering Pipeline](#feature-engineering-pipeline)
5. [Training Module](#training-module)
6. [Deployment](#deployment)
7. [CI](#ci)

## Description
This project addresses an end-to-end text classification problem based on a Kaggle dataset. It features a pipeline built using Airflow and Feast (feature store) for preprocessing and feature generation. Training and hyperparameter tuning leverage tools like scikit-learn, PyTorch, Transformers, Hyperopt, and Llama.cpp. The best-performing model is packaged for reuse and deployed via FastAPI for predictions. Code quality and functionality are ensured using tox and GitHub Actions.  

## Dataset
The dataset is from  [Quora Insincere Questions Classification](https://www.kaggle.com/competitions/quora-insincere-questions-classification/overview).

## [Feature Store](https://github.com/KianoushAmirpour/End_2_End_Text_Classification/tree/main/feature_store)
**Feast** is used as the feature store, with SQLite serving as the online store.  
Features are defined as shown in the example below:
```
question_entity = Entity(name='question_id',
                         join_keys=['id'],
                         value_type=ValueType.STRING,
                         description='A key to fetch features for each question.')

meta_features_file_source = FileSource(name='meta_features_file_source',
                                       path='data/meta_features.parquet',
                                       timestamp_field='event_timestamp')

meta_features = FeatureView(
    name='meta_features_extracted_from_text',
    entities=[question_entity],
    ttl=timedelta(days=20),
    schema=[
        Field(name='num_words', dtype=Int32),
        Field(name='num_unique_words', dtype=Int32),
        Field(name='num_stop_words', dtype=Int32),
        Field(name='num_title_case', dtype=Int32),
        Field(name='ave_length_words', dtype=Float32),
        Field(name='num_characters', dtype=Int32),
        Field(name='num_punctuations', dtype=Int32)
    ],
    source=meta_features_file_source,
    online=True
)
``` 

## [Feature Engineering Pipeline](https://github.com/KianoushAmirpour/End_2_End_Text_Classification/tree/main/feature_pipeline)
The image below illustrates the feature engineering pipeline:

![feature_engineering_pipelines](https://github.com/user-attachments/assets/7c78a4c5-73f9-4b4e-a7b4-12b52a49e142)

The feature engineering pipeline is implemented with **Apache Airflow** to orchestrate four DAGs for preprocessing, cleaning, and feature generation,
along with a dedicated DAG for registering features in **Feast (feature store)**.  

Key components and features:  
* Data Processing: Utilizes **pandas, NLTK, and sentence-transformers** for text preprocessing and feature extraction.
* Data Quality Checks (Each DAG performs validations, including checks for):
   * Row count and column constraints
   * Minimum and null values
   * Presence of required columns
* Error Handling and Alerts: **Automatic email notifications** are triggered upon pipeline failure.
* Monitoring and Logging:
   * **Grafana, Prometheus, and StatsD** are integrated to monitor pipeline health and performance metrics.
   * A custom logging captures detailed process logs and stored locally.

## [Training Module](https://github.com/KianoushAmirpour/End_2_End_Text_Classification/tree/main/training)
To streamline training and experimentation, a custom module was developed with inspiration from the **Factory Method design pattern**.  
While not all aspects strictly adhere to the pattern, the module is designed to make it easier to scale and add new components,  
allowing experiments to be configured through a single configuration file that specifies parameters and feature store settings.

Currently Supported Methods and Features:
* Preprocessing: StandardScaler
* Models:
  * Logistic Regression
  * Random Forest
  * XGBoost
  * DistilBERT (via Transformers and PyTorch)
* Few-Shot Classification: Experimented with Meta-Llama-3-8B-Instruct locally using llama.cpp.
* Hyperparameter Optimization:
  * Random Search
  * Hyperopt

After experimentation, the best model is trained on the complete dataset for production use.

#### Experiment Tracking:  
**MLflow** is integrated throughout the experimentation process for seamless tracking of experiments, parameters, and results.

#### Usage: 
For experimentation, you simply need to modify the configuration file in [run_experiment.py](https://github.com/KianoushAmirpour/End_2_End_Text_Classification/blob/main/training/run_experiment.py).  
For example, the following setup will:  
Train an XGBoost model, tune its parameters using the Hyperopt library, and apply StandardScaler for preprocessing. Retrieve the specified features from Feast for model training.  
Note: The values used in this example are arbitrary and are intended solely for showcasing the process.

```
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

```

## [Deployment](https://github.com/KianoushAmirpour/End_2_End_Text_Classification/tree/main/deployment)
Once the best model is trained, it is converted into a reusable package for seamless deployment in production.  
The same pipeline used for training is leveraged for making predictions, ensuring consistency and reliability.

Testing and Code Quality
* Testing: The package is tested using **pytest** to ensure functionality.
* Code Quality: Tools like **flake8, isort, and mypy** are used for linting, sorting imports, and type checking, respectively.
* Automation: **tox** is utilized to automate testing and enforce code quality standards across environments.

Configuration files used to build the package can be found [here](https://github.com/KianoushAmirpour/End_2_End_Text_Classification/tree/main/deployment/inference).

To install the package locally, navigate to the directory containing setup.py and run the following command: `pip install -e .`

For more information regarding building and publishing the package, visit [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
The following commands are described there:  

```
py -m pip install --upgrade pip
py -m pip install --upgrade build
py -m build
```

After building the model package, prediction endpoints were developed using FastAPI to serve the model in production. These endpoints can be found [here](https://github.com/KianoushAmirpour/End_2_End_Text_Classification/tree/main/deployment/serving_api).

## [CI](https://github.com/KianoushAmirpour/End_2_End_Text_Classification/tree/main/.github/workflows) 
The CI pipeline is triggered on both push and pull request events to test multiple components, such as inference and serving API deployments.  
Currently it just runs tox for code quality for inference and serving API.

