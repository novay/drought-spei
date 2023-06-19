import pickle 
import pandas as pd
import numpy as np

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

# Contoh Prediksi
# ============================
# 'Curah Hujan': 0,
# 'Kelembaban (Rata-rata)': 0,
# 'Kecepatan Angin Maksimum (m/s)': 0,
# 'Lama Sinar Matahari': 0,
# 'Suhu (Max)': 50, 
# 'Suhu (Min)': 40,
input_data = np.array([[0, 0, 0, 0, 50, 40]])

input_df = pd.DataFrame(input_data, columns=['RR', 'RH_avg', 'ff_avg', 'ss', 'Tx', 'Tn'])
prediction = knn_loaded.predict(input_df)

print(map_drought_condition(prediction[0]))