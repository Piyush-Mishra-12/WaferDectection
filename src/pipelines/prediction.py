from src.exception import CustomException
from dataclasses import dataclass
import os ,sys, shutil, pandas
from src.log import logging
from flask import request
from src import utils

@dataclass
class PredictFile:
    Predict_Odir: str = "Prediction"
    PredictFilename: str = "predicted.csv"
    PredictFilepath: str = os.path.join(Predict_Odir, PredictFilename)

class Prediction:
    def __init__(self, request:request):
        self.request = request
        self.predicted_file = PredictFile()

    def input_file(self)-> str:
        try:
            inputFile = self.request.files['file']
            predFilepath =  os.path.join(Predict_Odir, inputFile.filename)
            inputFile.save(predFilepath)
            return predFilepath
        except Exception as e:
            logging.info('Error in inputing file in Prediction Pipeline')
            raise CustomException(e,sys)
    
    def Predict(self, features):
        AWS_S3_BUCKET_NAME = "sensor-deployment"
        try:
            modelPath = utils.download_model(bucket_name=AWS_S3_BUCKET_NAME, bucket_filename='model.dill', dest_filename='model.dill',)
            model = utils.load_object(filepath=modelPath)
            pred = model.predict(features)
            return pred
        except Exception as e:
            logging.info('Error in predicting in Prediction Pipeline')
            raise CustomException(e,sys)

    def get_dataframe(self, df_path:pandas.DataFrame):
        try:
            df:pandas.DataFrame = pandas.read_csv(df_path)
            prediction = self.predict(df)
            df['class'] = [p for p in prediction]
            target_map = {0:'Negative', 1:'Positive'}

            df['class'] = df['class'].map(target_map)
            os.makedirs(self.PredictFile.Predict_Odir, exist_ok=True)
            df.to_csv(self.PredictedFile.PredictFilepath, index=False)
            logging.info('Predition is completed')
        except Exception as e:
            logging.info('Error in getting dataframe in Prediction Pipeline')
            raise CustomException(e,sys)

    def run_pipe(self):
        try:
            df_path = input_file()
            self.get_dataframe(df_path=df_path)
            return self.PredictFile
        except Exception as e:
            logging.info('Error in running in Prediction Pipeline')
            raise CustomException(e,sys)
