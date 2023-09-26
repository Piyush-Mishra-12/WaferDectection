import os
import sys
import dill
import boto3
import numpy as ny
import pandas as pd
from src.log import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pymongo import MongoClient

def export_collection_as_dataframe(c_name, db_name):
    try:
        MONGO_DB_URL = "mongodb+srv://Piyush12:EtxAzA28XTrwLjzX@cluster0.83fdgam.mongodb.net/?retryWrites=true&w=majority"
        mongo_client = MongoClient(MONGO_DB_URL)
        collection = mongo_client[db_name][c_name]
        df = pd.DataFrame(list(collection.find()))
        if '_id' in df.columns.to_list():
            df = df.drop(columns=['_id'], axis=1)
        df.replace({'na':ny.nan}, inplace=True)
        return df
    except Exception as e:
        logging.info('Error in collecting file from mongoDB')
        raise CustomException(e,sys)

def save_obj(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        logging.info('Error in saving dill file')
        raise CustomException(e,sys)

def load_object(filepath):
    try:
        with open(filepath,'rb') as file_obj:
            logging.info('Loading object from utils is completed')
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Error occured while loading object from utils')
        raise CustomException(e,sys)

def upload_file(from_filename, to_filename, bucket_name):
    try:
        s3_resource = boto3.client('s3')
        s3_resource.upload_file(from_filename, to_filename, bucket_name)
    except Exception as e:
        logging.info('Error occured while uploading file from utils')
        raise CustomException(e,sys)

def download_model(bucket_name, bucket_filename, dest_filename):
    try:
        s3_client = boto3.client('s3')
        s3_client.meta.clint.download_model(bucket_name, bucket_filename, dest_filename)
        return dest_filename
    except Exception as e:
        logging.info('Error occured while downloading model from utils')
        raise CustomException(e,sys)

def evaluate(x, y, models):
            try:
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                report = {}
                for i in range(len(list(models))):
                    model = list(models.values())[i]
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    train_model_score = r2_score(y_train, y_train_pred)
                    test_model_score = r2_score(y_test, y_test_pred)
                    report[list(models.keys())[i]] = test_model_score
                return report
            except Exception as e:
                logging.info('Error occured while evaluting model from utils')
                raise CustomException(e, sys)
