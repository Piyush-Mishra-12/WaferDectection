import os
import sys
import numpy as ny
import pandas as pd
from src.log import logging
from src.utils import save_obj
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, FunctionTransformer

@dataclass
class TransformConfig:
    preprocessor_filepath:str = os.path.join('storage', 'preprocessor.dill')

class Transformation:

    def __init__(self):
        self.transform_config = TransformConfig()

    def get_transformation(self):
        try:
            logging.info('Data pipeline Initiated')
            nan_replace = ('Replace', FunctionTransformer(lambda X: X.replace('na', ny.nan)))
            imputer = ('Imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler = ('Scaler', RobustScaler())
            preprocessor = Pipeline(steps=[nan_replace, imputer, scaler])
            logging.info('Data pipeline Completed')
            return preprocessor
        except Exception as e:
            logging.info('Error while getting transformation in Data Transformation')
            raise CustomException(e,sys)

    def start_transformation(self, train_path, test_path):
        try:
            # Getting X and Y from Train and Test
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)

            X_train = train.drop(columns=['Good/Bad'], axis=1)
            Y_train = train['Good/Bad'].astype(str).str.strip().replace({'-1': 0, '+1': 1}).astype(int)
            X_test = test.drop(columns=['Good/Bad'], axis=1)
            Y_test = test['Good/Bad'].astype(str).str.strip().replace({'-1': 0, '+1': 1}).astype(int)

            # Scaling through pipeline
            preprocessor = self.get_transformation() 
            X_Train = preprocessor.fit_transform(X_train)
            X_Test = preprocessor.transform(X_test)
            logging.info('scaling is completed')

            # Resampling Train and Test
            ada = ADASYN(sampling_strategy='auto')
            x_train, y_train = ada.fit_resample(X=X_Train, y=Y_train)
            logging.info('resampling is completed')

            # Making of array for Train and Test
            train_arr = ny.c_[x_train, ny.array(y_train)]
            test_arr = ny.c_[X_Test, ny.array(Y_test)]

            save_obj(filepath=self.transform_config.preprocessor_filepath, obj=preprocessor)
            logging.info('Data Transformation is completed')
            return (train_arr, test_arr, self.transform_config.preprocessor_filepath)

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)