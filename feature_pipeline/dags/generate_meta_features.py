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
    'start_date': datetime(2024, 12, 4),
    'wait_for_downstream': True,
    'catchup': False,
}


with DAG(dag_id='Meta_Features',
         description='Generates meta features',
         default_args=default_args,
         schedule_interval='0 0 * * 0',
         tags=['meta_features'],
         ):

    start = DummyOperator(task_id='start')

    # wait_for_preprocessing = ExternalTaskSensor(
    #     task_id='wait_for_preprocessing',
    #     external_dag_id='preprocess_the_data',
    #     external_task_id='end',
    #     execution_delta=timedelta(hours=1),
    #     mode='poke',
    #     timeout=60*5
    # )

    cal_meta_features = PythonOperator(task_id='calculate_meta_features',
                                       python_callable=src.calculate_meta_features,
                                       )

    

    wait_for_meta_features = FileSensor(task_id='wait_for_meta_features',
                                        fs_conn_id='parquet_conn',
                                        poke_interval=60,
                                        timeout=60*5,
                                        mode='reschedule',
                                        soft_fail=True,
                                        filepath=src.DirectoriesConfig.FEATURE_STORE_DIR/'data'/'meta_features.parquet',
                                        )

    end = DummyOperator(task_id='end')


# start >> wait_for_preprocessing >> cal_meta_features >> wait_for_meta_features >> end
start >> cal_meta_features >> wait_for_meta_features >> end
