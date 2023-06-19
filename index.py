import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model KNN
with open('../models/knn.pkl', 'rb') as file:
    knn_loaded = pickle.load(file)

def map_drought_condition(di):
    condition_map = {
        'D0': 'No-Drought',
        'D1': 'MILD',
        'D2': 'MODERATE',
        'D3': 'SEVERE',
        'D4': 'EXTREME'
    }
    return condition_map.get(di, 'Unknown')

@app.route('/predict', methods=['POST'])
def predict_drought_condition():
    # Mengambil data input dari permintaan POST
    input_data = request.json['input_data']
    
    # Mengubah data input menjadi DataFrame
    input_df = pd.DataFrame(input_data, columns=['RR', 'RH_avg', 'ff_avg', 'ss', 'Tx', 'Tn'])
    
    # Melakukan prediksi menggunakan model KNN
    prediction = knn_loaded.predict(input_df)
    
    # Mengirimkan hasil prediksi sebagai respons JSON
    result = {
        'predicted_condition': map_drought_condition(prediction[0])
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run()