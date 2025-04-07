# Network Security ML Project

## Overview

This project implements a machine learning pipeline for network security data classification that detects phishing attempts and other network security threats by analyzing network traffic data. The system features automated data processing, model training and evaluation, and a web API for making predictions.

## Key Features

- Data ingestion pipeline from MongoDB
- Automated data validation and transformation 
- Machine learning model training and evaluation
- FastAPI web interface for predictions
- Docker containerization
- Cloud deployment capabilities with AWS (ECR, EC2, S3)

## Tech Stack

- Python 3.9+
- FastAPI
- Scikit-learn
- MLflow for experiment tracking
- MongoDB for data storage
- Docker for containerization

## Setup and Running the Application

### Prerequisites

- Python 3.9+
- MongoDB connection
- Docker (optional)

### Quick Start

1. **Clone the repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file with:
   ```
   MONGODB_URI=your_mongodb_connection_string
   ```

4. **Run the API**
   ```bash
   # Run directly with Python
   python app.py
   
   # Alternatively, use uvicorn directly
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
   
   The API will be available at http://localhost:8000
   
   API documentation is accessible at http://localhost:8000/docs

### Docker Deployment

```bash
# Build image
docker build -t networksecurity:latest .

# Run container
docker run -p 8000:8000 networksecurity:latest
```

## API Endpoints

- **GET `/`**: Redirects to API documentation
- **GET `/train`**: Initiates the model training pipeline
- **POST `/predict`**: Upload a CSV file to get predictions