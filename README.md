# Data Science and Machine Learning Projects Portfolio

This repository contains a collection of data science and machine learning projects, each focusing on different aspects of the ML lifecycle and tooling.

## Projects Overview

### 1. Network Security ML Project

A machine learning pipeline for network security data classification. It's designed to detect phishing attempts and other network security threats by analyzing network traffic data.

- **Tech Stack**: FastAPI, Scikit-learn, MLflow, MongoDB, Docker
- **Key Features**: 
  - Automated ML pipeline (data ingestion, validation, transformation, training)
  - Web API for model training and prediction
  - Docker containerization
- **Directory**: [networksecurity/](networksecurity/)
- [Detailed README](networksecurity/src/README.md)

### 2. Basic Data Science Project

A foundational data science project demonstrating essential ML pipeline components.

- **Tech Stack**: Python, Scikit-learn
- **Key Features**: 
  - Data ingestion, validation, and transformation
  - Model training and evaluation
  - Artifact management
- **Directory**: [basics_ds_project/](basics_ds_project/)

### 3. DVC and MLflow Integration

A project showcasing Data Version Control (DVC) and MLflow for experiment tracking.

- **Tech Stack**: DVC, MLflow
- **Key Features**: 
  - Version control for datasets and models
  - Experiment tracking and model registry
  - Reproducible ML pipelines
- **Directory**: [dvc_mlflow_basics/](dvc_mlflow_basics/)

### 4. ETL with Airflow

An ETL (Extract, Transform, Load) pipeline implemented using Apache Airflow.

- **Tech Stack**: Apache Airflow, Docker Compose
- **Key Features**: 
  - Scheduled data processing workflows
  - Containerized workflow management
  - DAG-based pipeline orchestration
- **Directory**: [etl_with_airflow/](etl_with_airflow/)

## Getting Started

Each project has its own README with specific setup instructions. Navigate to the respective project directory to learn more about how to run each project.

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized projects)
- Git (for version control)
