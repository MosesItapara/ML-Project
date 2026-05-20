import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import dill

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")

            numerical_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
            logging.info(f"Numerical columns: {numerical_columns}")

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully")

            # Get preprocessor
            preprocessor_obj = self.get_data_transformer_object()
            logging.info("Preprocessor object obtained")

            target_column = "Calories"
            numerical_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

            # Split features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Applying preprocessor on train and test sets")

            # Fit on train, transform both
            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_test_transformed = preprocessor_obj.transform(X_test)

            # Combine features and target
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Save preprocessor object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as f:
                dill.dump(preprocessor_obj, f)
            logging.info("Preprocessor object saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)