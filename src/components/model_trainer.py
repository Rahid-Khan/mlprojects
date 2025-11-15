import os
import sys 
import pandas as pd 
import numpy as np

from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.utils import save_object, evaluate_models
from src.exception import CustomException
from src.logger import logging

@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
    
class ModelTrainer:
    def __init__ (self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_config(self, train_array, test_Array):
        try:
            logging.info("input of train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_Array[:,:-1],
                test_Array[:,-1]
            )
            models = {
                "Random Forest":RandomForestRegressor(),
                "DecsionTree": DecisionTreeRegressor(),
                "Linear Regressor": LinearRegression(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "k-nearest Neighbours" : KNeighborsRegressor(),
                "XbBoost": XGBRegressor(),
                "Adaboost" : AdaBoostRegressor()
            }
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test,y_test= y_test, models=models)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score <0.6:
                raise CustomException("no best model found")
            logging.info(f"best model is found on boath test and train data")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                
            )
            
            predicated = best_model.predict(X_test)
            
            r2_square = r2_score(y_test, predicated)
            
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)
        