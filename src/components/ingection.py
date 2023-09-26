import os
import sys
import pandas as pd
from src import utils
from src.log import logging
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass
class IngestionConfig:
    train_path:str = os.path.join('storage', 'train.csv')
    test_path:str = os.path.join('storage', 'test.csv')
    raw_path:str = os.path.join('storage', 'raw.csv')


class Ingestion:
    def __init__(self):
        self.ingestion_config = IngestionConfig()

    def start_ingestion(self):
        logging.info('Data Ingestion Begins')
        MONGO_DATABASE_NAME = "wafer_data"
        MONGO_COLLECTION_NAME = "wafer"
        try:
            df:pd.DataFrame = utils.export_collection_as_dataframe(db_name=MONGO_DATABASE_NAME, c_name=MONGO_COLLECTION_NAME)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_path, index=False, header=True)

            logging.info('Start of Training & Testing Split')
            train, test = train_test_split(df, test_size=0.15, random_state=52)
            train.to_csv(self.ingestion_config.train_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_path, index=False, header=True)
            logging.info("Ingestion of data is Completed")
            return (self.ingestion_config.train_path, self.ingestion_config.test_path)
        except Exception as e:
            logging.info('Error occured while Data Ingestion')
            raise CustomException(e,sys)