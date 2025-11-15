import sys
import os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
 

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join("artifacts", "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()
        
    def get_data_transformation_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score" ,"reading_score"]
            categorical_columns = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course']
            
            num_pipline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
               
            )
            cat_pipline = Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ("onehotencoder", OneHotEncoder()),
                        ("scaler", StandardScaler(with_mean=False))
                    ]
                )
            logging.info("numerical and categorical features handling") 
            
            preprocessor = ColumnTransformer(
                [("num_pipline", num_pipline, numerical_columns),
                 ("cat_pipline", cat_pipline,categorical_columns)
                 ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("read train and test data")
            logging.info("obtainig preprocessor object")
            preprocessing_object = self.get_data_transformation_object()
            target_column_name = "math_score"
            numerical_columns= ["writing_score", "reading_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(
                f"appling preprocessing on train and test data set"
            )
            inpute_feature_train_arr= preprocessing_object.fit_transform(input_feature_train_df)
            inpute_feature_test_arr= preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[inpute_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[inpute_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("saving preprocess object")
            
            save_object (
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_object
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
        