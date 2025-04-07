import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    TrainingPipelineConfig,
)
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

        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, data_validation_config)
        log.info("Initiate the data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        log.info("data Validation Completed")
        print(data_validation_artifact)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
