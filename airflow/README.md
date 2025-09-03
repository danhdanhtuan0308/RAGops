Airflow Local (Docker Compose)

Run Airflow UI locally to test DAGs that import from `MSDS_2024_2026/RAGops/ingestion`.

Prerequisites
- Docker and Docker Compose

Setup
1) Copy env template and set UID (optional):
   cp .env.example .env
   # macOS/Linux: echo "AIRFLOW_UID=$(id -u)" >> .env

2) Initialize DB and create admin user:
   docker compose up airflow-init

3) Start services:
   docker compose up -d airflow-webserver airflow-scheduler

4) Open UI: http://localhost:8080  (user/pass: airflow/airflow)

Volumes
- ./dags -> /opt/airflow/dags
- ./plugins -> /opt/airflow/plugins
- ./logs -> /opt/airflow/logs
- ../ingestion -> /opt/airflow/ingestion (on PYTHONPATH)

Notes
- Add your DAGs under `./dags`. They can `import ingestion.<module>`.
- Stop: docker compose down
- Logs: docker compose logs -f airflow-webserver

