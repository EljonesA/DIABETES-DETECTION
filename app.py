from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
import pandas as pd

import json
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download necessary NLTK resources
nltk.download('punkt') 

app = Flask(__name__)

# Load the pre-trained model
try:
    model = load('diabetes_model.joblib')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

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
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
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
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500


from chatbot import get_response

# Add new endpoint for chat
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        # Get response from chatbot
        response = get_response(user_message)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run()