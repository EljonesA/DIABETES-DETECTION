import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data with intents
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "How are you", "Greetings"],
            "responses": ["Hello! How can I help you with diabetes information today?", "Hi there! I'm here to help with diabetes-related questions."]
        },
        {
            "tag": "symptoms",
            "patterns": ["What are diabetes symptoms?", "How do I know if I have diabetes?", "Common symptoms", "Signs of diabetes"],
            "responses": [
                "Common diabetes symptoms include: \n- Increased thirst and urination\n- Extreme hunger\n- Fatigue\n- Blurred vision\n- Slow-healing sores\n- Frequent infections",
                "Key signs of diabetes include excessive thirst, frequent urination, unexplained weight loss, and fatigue. Please consult a doctor for proper diagnosis."
            ]
        },
        {
            "tag": "risk_factors",
            "patterns": ["What increases diabetes risk?", "Risk factors", "Am I at risk?", "Diabetes risk factors"],
            "responses": [
                "Common risk factors for type 2 diabetes include:\n- Being overweight\n- Physical inactivity\n- Age over 45\n- Family history\n- High blood pressure\n- Gestational diabetes history"
            ]
        },
        {
            "tag": "prevention",
            "patterns": ["How to prevent diabetes?", "Diabetes prevention", "Avoid getting diabetes", "Prevention tips"],
            "responses": [
                "To help prevent diabetes:\n- Maintain a healthy weight\n- Exercise regularly\n- Eat a balanced diet\n- Limit sugary foods\n- Monitor blood pressure\n- Get regular check-ups"
            ]
        },
        {
            "tag": "treatment",
            "patterns": ["How is diabetes treated?", "Diabetes treatment", "Managing diabetes", "Diabetes medication"],
            "responses": [
                "Diabetes treatment may include:\n- Blood sugar monitoring\n- Insulin therapy\n- Oral medications\n- Healthy diet\n- Regular exercise\n- Regular medical check-ups",
                "Treatment varies by type but typically involves blood sugar monitoring, medication, and lifestyle changes. Consult your doctor for personalized advice."
            ]
        },
        {
            "tag": "diet",
            "patterns": ["What should diabetics eat?", "Diabetes diet", "Food for diabetes", "Eating with diabetes"],
            "responses": [
                "A diabetes-friendly diet includes:\n- Whole grains\n- Lean proteins\n- Fresh vegetables\n- Limited processed foods\n- Controlled portions\n- Low glycemic foods"
            ]
        },
        {
            "tag": "glucose_levels",
            "patterns": ["Normal glucose levels", "Blood sugar range", "What should my glucose be?", "Target glucose"],
            "responses": [
                "Normal blood glucose levels are:\n- Fasting: 70-99 mg/dL\n- 2 hours after meals: <140 mg/dL\n- A1C: below 5.7%\nConsult your doctor for personal targets."
            ]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye", "See you", "Thanks", "Thank you"],
            "responses": ["Goodbye! Take care of your health!", "Thanks for chatting! Remember to consult healthcare professionals for medical advice."]
        }
    ]
}

# Prepare training data
def prepare_training_data():
    training_patterns = []
    training_tags = []
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            training_patterns.append(pattern)
            training_tags.append(intent["tag"])
    return training_patterns, training_tags

# Initialize and train the model
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
clf = MultinomialNB()

def train_chatbot():
    patterns, tags = prepare_training_data()
    X = vectorizer.fit_transform(patterns)
    clf.fit(X, tags)

def get_response(user_input):
    # Transform user input
    input_vec = vectorizer.transform([user_input])
    # Predict tag
    tag = clf.predict(input_vec)[0]
    # Get random response for the predicted tag
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to respond to that. Please ask about diabetes symptoms, treatments, or prevention."

# Train the model when module is loaded
train_chatbot()
