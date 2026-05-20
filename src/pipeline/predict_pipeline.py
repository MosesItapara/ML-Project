import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            model = load_object(file_path=model_path)
            predicted = model.predict(features)
            return predicted

        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 age: int,
                 height: int,
                 weight: int,
                 duration: int,
                 heart_rate: int,
                 body_temp: int):
        self.age = age
        self.height = height
        self.weight = weight
        self.duration = duration
        self.heart_rate = heart_rate
        self.body_temp = body_temp

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Age": [self.age],
                "Height": [self.height],
                "Weight": [self.weight],
                "Duration": [self.duration],
                "Heart_Rate": [self.heart_rate],
                "Body_Temp": [self.body_temp]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)