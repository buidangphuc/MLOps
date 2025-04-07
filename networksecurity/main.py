import sys

from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import (DataIngestionConfig, TrainingPipelineConfig)
from src.exception.exception import NetworkSecurityException
from src.log import log

if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        log.info("Initiate the data ingestion")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        log.info("Data Initiation Completed")
        print(dataingestionartifact)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
