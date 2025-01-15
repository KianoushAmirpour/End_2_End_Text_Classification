import os
import src
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'airflow',
    'email': [os.environ.get('AIRFLOW_EMAIL')],
    "depends_on_past": False,
    'email_on_failure': True,
    'retries': 2,
    'email_on_retry': False,
    'retry_delay': timedelta(seconds=30),
    'start_date': datetime(2025, 1, 6),
    'wait_for_downstream': True,
    'catchup': False,
}


with DAG(dag_id='Cleaned_stemmed_features',
         description='Generates cleaned and stemmed features',
         default_args=default_args,
         schedule_interval='0 0 * * 0',
         tags=['cleaned', 'stemmed'],
         ):

    start = DummyOperator(task_id='start')

    # wait_for_preprocessing = ExternalTaskSensor(
    #     task_id='wait_for_preprocessing',
    #     external_dag_id='preprocess_the_data',
    #     external_task_id='end',
    #     execution_delta=timedelta(minutes=5),
    #     mode='reschedule',
    #     timeout=60*5
    # )

    clean_stem_features = PythonOperator(task_id='clean_stem_features',
                               python_callable=src.clean_text)
    
    
    wait_for_cleaned_features = FileSensor(task_id='wait_for_cleaned_features',
                                           fs_conn_id='parquet_conn',
                                           poke_interval=60,
                                           timeout=60*5,
                                           mode='reschedule',
                                           soft_fail=True,
                                           filepath=src.DirectoriesConfig.FEATURE_STORE_DIR/'data'/'cleaned_stemmed.parquet',
                                           )

    end = DummyOperator(task_id='end')


# start >> wait_for_preprocessing >> clean_stem_features >> wait_for_cleaned_features >> end
start >> clean_stem_features >> wait_for_cleaned_features >> end

