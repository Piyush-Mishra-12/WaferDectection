from src.exception import CustomException
from dataclasses import dataclass
import os, sys, pandas
from src.log import logging
from flask import request
from src import utils

@dataclass
class Prediction:
    Predict_Odir: str = "Prediction"
    PredictFilename: str = "predicted.csv"

    def __init__(self, request:request): # type: ignore
        self.Predict_Odir: str = "Prediction"
        self.request = request
        self.PredictFilepath = os.path.join(self.Predict_Odir, self.PredictFilename)

    def input_file(self) -> str:
        try:
            inputFile = self.request.files['file']
            predFilepath = os.path.join(self.Predict_Odir, inputFile.filename)
            inputFile.save(predFilepath)
            return predFilepath
        except Exception as e:
            logging.info('Error in inputting file in Prediction Pipeline')
            raise CustomException(e, sys) # type: ignore

    def Predict(self, features):
        try:
            model_path = os.path.join('storage', 'model.dill')
            model = utils.load_object(filepath=model_path)
            pred = model.predict(features)
            return pred
        except Exception as e:
            logging.info('Error in predicting in Prediction Pipeline')
            raise CustomException(e, sys) # type: ignore

    def get_dataframe(self, df_path: pandas.DataFrame):
        try:
            df: pandas.DataFrame = pandas.read_csv(df_path) # type: ignore
            prediction = self.Predict(df)
            df['class'] = [p for p in prediction]
            target_map = {0: 'Negative', 1: 'Positive'}

            df['class'] = df['class'].map(target_map)
            os.makedirs(self.Predict_Odir, exist_ok=True)
            df.to_csv(self.PredictFilepath, index=False)
            logging.info('Prediction is completed')
            return self.PredictFilepath
        except Exception as e:
            logging.info('Error in getting dataframe in Prediction Pipeline')
            raise CustomException(e, sys) # type: ignore

    def run_pipe(self):
        try:
            df_path = self.input_file()
            self.get_dataframe(df_path=df_path) # type: ignore
            return self.PredictFilepath
        except Exception as e:
            logging.info('Error in running in Prediction Pipeline')
            raise CustomException(e, sys) # type: ignore