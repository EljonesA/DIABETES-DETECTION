from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = load('diabetes_model.joblib')

# Define feature names
FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or not all(key in data for key in FEATURE_NAMES):
            return jsonify({'error': 'Invalid input data'}), 400
        
        # Create DataFrame with feature names
        features = pd.DataFrame([data], columns=FEATURE_NAMES)
        prediction = model.predict(features)[0]
        
        # Get feature importances
        importances = dict(zip(
            FEATURE_NAMES,
            model.feature_importances_
        ))
        
        # Sort features by importance
        sorted_importances = dict(sorted(
            importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        # Get top 4 most important features
        top_features = {k: float(v) for k, v in list(sorted_importances.items())[:4]}
        
        result = 'High Risk of Diabetes' if prediction == 1 else 'No Diabetes'
        return jsonify({
            'prediction': int(prediction), 
            'result': result,
            'feature_importance': top_features
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
