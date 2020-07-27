from sklearn.externals import joblib
import numpy as np
import pandas as pd
from numpy import loadtxt
from flask import Flask, jsonify, request

model_file=r'E:\InfyU LABS Device Reading\App\GBC_Mang_UV_class.pkl'
model=joblib.load(model_file)

scaler_file=r'E:\InfyU LABS Device Reading\App\GBC_Mang_UV_class_scaler.pkl'
scaler=joblib.load(scaler_file)

pca_file=r'E:\InfyU LABS Device Reading\App\GBC_Mang_UV_class_scaler_pca.pkl'
pca=joblib.load(pca_file)

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])


def predict():
	data = request.get_json(force=True)
	X=data.split()
	x=np.array(X).astype(float)



	X_1=X[0:288].reshape(1,-1)
	X_scaled=scaler.transform(X_1)
	X_pca=pca.transform(X_scaled)
	result=model.predict(X_pca)



	output = {'results': int(result[0])}

	return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)