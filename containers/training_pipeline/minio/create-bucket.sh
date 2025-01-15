#!/bin/bash
# Configure MinIO Client
mc alias set minioserver http://minio:9000 ${MINIO_ACCESS_KEY} ${MINIO_SECRET_ACCESS_KEY}

# Create the MLFlow bucket
mc mb minioserver/mlflow