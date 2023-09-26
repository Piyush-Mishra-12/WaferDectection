import os
import sys
from src import utils
from src.log import logging
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import r2_score

@dataclass
class TrainerConfig:
    trainer_filepath:str = os.path.join('storage', 'model.dill')


class Model:
    def __init__(self, preprocessor_obj, model_obj):
        self.preprocessor_obj = preprocessor_obj
        self.model_obj = model_obj
    def predict(self, x):
        transformed_feature = self.preprocessor_obj.transform(x)
        return self.model_obj.predict(transformed_feature)
    def __repr__(self):
        return f'{type(self.model_obj).__name__}()'
    def __str__(self):
        return f'{type(self.model_obj).__name__}()'


class Trainer:
    def __init__(self):
        self.trainer_config = TrainerConfig()
    
    def start_trainer(self, train_arr, test_arr, preprocessor_path):
        AWS_S3_BUCKET_NAME = "sensor-deployment"
        try:
            logging.info('Splitting training and test datasets')
            x_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            # Models and their names
            models = {
                'LR': LogisticRegression(),
                'Random Forest': RandomForestClassifier(random_state=47),
                'SVM_lin': SVC(kernel='linear'),
                'SVM_rbf': SVC(kernel='rbf'),
                'Gradient Boosting': GradientBoostingClassifier(),
                'xgb': XGBClassifier(objective='binary:logistic'),
                'lgb': LGBMClassifier(objective='binary', metric='binary_logloss'),
                'dt': DecisionTreeClassifier(),
                'knn': KNeighborsClassifier(n_neighbors=5)}
            
            logging.info('Extracting model config file path')
            model_report = utils.evaluate(x=x_train, y=y_train, models=models)
            
            # To get best model
            bmodel_score = max(sorted(model_report.values()))
            bmodel_name = list(model_report.keys())[list(model_report.values()).index(bmodel_score)]
            bmodel = models[bmodel_name]
            if bmodel_score < 0.6:
                raise Exception('No model is good enough')
            logging.info(f'Best model which is selected is {bmodel_name}')
            
            preprocessor_obj = utils.load_object(filepath=preprocessor_path)
            custom_model = Model(preprocessor_obj=preprocessor_obj, model_obj=bmodel)
            logging.info(f'Saving model at path: {self.trainer_config.trainer_filepath}')

            utils.save_obj(filepath=self.trainer_config.trainer_filepath, obj=custom_model)
            predicted = bmodel.predict(x_test)
            r2 = r2_score(y_test, y_pred= predicted)
            utils.upload_file(from_filename=self.trainer_config.trainer_filepath, to_filename='model.dill', bucket_name=AWS_S3_BUCKET_NAME,)
            return r2

        except Exception as e:
            logging.info('Error in Data Training')
            raise CustomException(e,sys)