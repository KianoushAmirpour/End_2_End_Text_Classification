import os
import src
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
 
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

with DAG(dag_id='Apply_Feast',
         description='Apply feast to register features',
         default_args=default_args,
         schedule_interval='0 0 * * 0',
         tags=['feast'],
         ):
    
    start = DummyOperator(task_id='start')
    
    # wait_for_meta_features = ExternalTaskSensor(
    #     task_id = 'wait_for_meta_features',
    #     external_dag_id='Meta_Features',
    #     external_task_id='end',
    #     mode='reschedule',
    #     timeout=60*5
    # )
    
    # wait_for_cleaned_features = ExternalTaskSensor(
    #     task_id = 'wait_for_cleaned_features',
    #     external_dag_id='Cleaned_stemmed_features',
    #     external_task_id='end',
    #     mode='reschedule',
    #     timeout=60*5
    # )
    
    # wait_for_embeddings = ExternalTaskSensor(
    #     task_id = 'wait_for_embeddings',
    #     external_dag_id='Embeddings_features',
    #     external_task_id='end',
    #     mode='reschedule',
    #     timeout=60*5
    # )
    
    remove_train_temp_file = PythonOperator(task_id='remove_train_temp_file',
                                        python_callable=src.remove_file,
                                        op_args=[src.DirectoriesConfig.TEMP_DIR/'preprocessed.csv'],
                                        provide_context=True)
    
    apply_feast = PythonOperator(task_id='feast_apply',
                                 python_callable=src.apply_feast)
    
    end_task = DummyOperator(task_id='end')
    
    # start >> [wait_for_meta_features , wait_for_cleaned_features] >> remove_train_temp_file >> apply_feast >> end_task
    start >> remove_train_temp_file >> apply_feast >> end_task

    
    