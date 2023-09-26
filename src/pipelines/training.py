import os
import sys
import pandas as pd
from src.log import logging
from src.exception import CustomException
from src.components.ingection import Ingestion
from src.components.transform import Transformation
from src.components.trainer import Trainer

class TrainingPipeline:
    def __init__(self)->None:
        self.ingection = Ingestion()
        self.transformation = Transformation()
        self.trainer = Trainer()

    def start_pipeline(self):
        try:
            train_path, test_path = self.ingection.start_ingestion()
            (train, test, preprocessor_filepath) = self.transformation.start_transformation(train_path=train_path, test_path=test_path)
            r2 = self.trainer.start_trainer(train_arr=train, test_arr=test, preprocessor_path=preprocessor_filepath,)
            print('Training Completed\nTrained model score: ',r2)
        except Exception as e:
            logging.info('Error in training pipeline')
            raise CustomException(e,sys)