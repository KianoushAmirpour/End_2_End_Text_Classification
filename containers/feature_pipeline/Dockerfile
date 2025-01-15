FROM apache/airflow:2.10.2 

ENV PYTHONPATH "${PYTHONPATH}:/opt/airflow"

COPY requirements.txt .

RUN pip install -r requirements.txt

