# Local MinIO, MLflow & Opik via Docker Compose

- MinIO Console: http://localhost:9001 (user/pass in `.env` or `minioadmin/minioadmin`)
- S3 endpoint: http://localhost:9000
- MLflow UI: http://localhost:5000

Start all services:
```bash
docker compose -f ops/docker-compose.yml up -d --build
```

MinIO buckets will be created automatically: `mlflow`, `dvc`.

All data and configuration will be stored in folders specified by environment variables in the `.env` file.
