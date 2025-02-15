import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from ML import x

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)
     prediction = lr.predict(query)
     return jsonify({'prediction': list(prediction)})

model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 12345 # If you don't provide any port then the port will be set to 12345
    lr = joblib.load('model.pkl') # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load('model_columns.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(port=port, debug=True)
