import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
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

        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        log.info("data Transformation started")
        data_transformation = DataTransformation(
            data_validation_artifact, data_transformation_config
        )
        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )
        print(data_transformation_artifact)
        log.info("data Transformation completed")

    except Exception as e:
        raise NetworkSecurityException(e, sys)
