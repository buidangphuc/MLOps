import json
import os
import sys

import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv

from src.exception.exception import NetworkSecurityException

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
ca = certifi.where()


class NetworkDataExtract:

    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            # Add proper connection options with certifi SSL certificates
            # and increase server selection timeout
            self.mongo_client = pymongo.MongoClient(
                MONGODB_URI, tlsCAFile=ca, serverSelectionTimeoutMS=5000
            )

            # Test connection before proceeding
            # This will raise an exception if connection fails
            self.mongo_client.admin.command("ping")

            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]

            # Insert data in batches to handle large datasets
            batch_size = 1000
            for i in range(0, len(self.records), batch_size):
                batch = self.records[i : i + batch_size]
                self.collection.insert_many(batch)

            return len(self.records)
        except pymongo.errors.ConnectionFailure as e:
            raise NetworkSecurityException(f"MongoDB Connection Failed: {e}", sys)
        except pymongo.errors.ServerSelectionTimeoutError as e:
            raise NetworkSecurityException(
                f"MongoDB Server Selection Timeout: Please check if MongoDB is running: {e}",
                sys,
            )
        except pymongo.errors.BulkWriteError as e:
            raise NetworkSecurityException(e.details, sys)
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    FILE_PATH = "/Users/phuc.buidang/Documents/Learn/Projects/networksecurity/Network_Data/phisingData.csv"
    DATABASE = "MLOps"
    Collection = "NetworkData"
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(f"Converted {len(records)} records from CSV to JSON")
    no_of_records = networkobj.insert_data_mongodb(records, DATABASE, Collection)
    print(f"Successfully inserted {no_of_records} records into MongoDB")
