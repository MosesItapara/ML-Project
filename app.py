from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline

application = Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            age=float(request.form['age']),
            height=float(request.form['height']),
            weight=float(request.form['weight']),
            duration=float(request.form['duration']),
            heart_rate=float(request.form['heart_rate']),
            body_temp=float(request.form['body_temp'])
        )
        pred_df=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(features=pred_df)
        return render_template('home.html', results=results[0])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)