import os
import src
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator
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


with DAG(dag_id='preprocess_the_data',
         default_args=default_args,
         schedule_interval='0 0 * * 0',
         tags=['renaming_columns', 'add_id_column',
               'add_event_timestamp_column'],
         ):

    start = DummyOperator(task_id='start')
    
    preprocessing_task = PythonOperator(task_id='preprocess_the_data',
                                python_callable=src.preprocess
                                )
    

    end = DummyOperator(task_id='end')

start >> preprocessing_task >>  end
