import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.datascience import log
from src.datascience.entity.config_entity import DataTransformationConfig


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    ## Note: You can add different data transformation techniques such as Scaler, PCA and all

    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        log.info("Splited data into training and test sets")
        log.info(train.shape)
        log.info(test.shape)

        print(train.shape)
        print(test.shape)
