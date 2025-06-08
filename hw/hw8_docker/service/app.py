
# -*- coding: windows-1251 -*-
import pandas as pd
from flask import Flask, request, jsonify
from service import app
import lib.utils as utils


app = Flask(__name__)
app.config['MODEL_PATH'] = './models/log_reg.joblib'


MODEL = utils.load_model(app.config['MODEL_PATH'])


@app.route('/predict', methods=["POST"])
def predict():

    try:
        data = request.json

        df = pd.DataFrame(data, index=[0])
        df = df.reset_index(drop=True)

        prediction = utils.predict(MODEL, df)
        classes = ["no disease", "disease"]
        prediction = classes[prediction[0]]
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    print(prediction)
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005) 