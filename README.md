# DiabetesAI - Diabetes Risk Assessment Tool

A machine learning-powered web application that predicts diabetes risk using clinical features. The application uses a Random Forest Classifier trained on the PIMA Indians Diabetes Dataset.

## Model Overview

The diabetes prediction model is built using a Random Forest Classifier, which:
- Uses an ensemble of decision trees to make predictions
- Analyzes 8 clinical features to assess diabetes risk
- Provides feature importance analysis for transparency
- Trained on the PIMA Indians Diabetes Dataset

### Key Features Used
1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration (2 hours after glucose tolerance test)
3. **Blood Pressure**: Diastolic blood pressure (mm Hg)
4. **Skin Thickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (μU/ml)
6. **BMI**: Body mass index
7. **Diabetes Pedigree Function**: A function scoring likelihood of diabetes based on family history
8. **Age**: Age in years

## How It Works

1. **Data Collection**: Users input their health metrics through a user-friendly interface
2. **Preprocessing**: The application processes and validates the input data
3. **Prediction**: The Random Forest model analyzes the features and provides a risk assessment
4. **Explanation**: Results are displayed with:
   - Clear risk indication (High Risk/Low Risk)
   - Feature importance visualization
   - Key metrics analysis

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML/CSS/JavaScript
- **Model**: Random Forest Classifier (scikit-learn)
- **Data Processing**: pandas, numpy

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-detection.git
cd diabetes-detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Access the application at `http://localhost:5000`

## Model Performance
The Random Forest model was trained on the PIMA Indians Diabetes Dataset and achieves:
- Accuracy: 77.3%
- Precision: 79% (weighted average)
- Recall: 77% (weighted average)
- F1-Score: 78% (weighted average)

Class-wise Performance:
- Non-diabetic (Class 0):
    - Precision: 87%
    - Recall: 76%
    - F1-Score: 81%
- Diabetic (Class 1):
    - Precision: 65%
    - Recall: 80%
    - F1-Score: 72%

## Usage Guidelines

- Enter accurate health metrics for reliable predictions
- Use recent medical test results when available
- Consult healthcare professionals for medical advice
- This tool is for preliminary risk assessment only

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Committing changes
4. Opening a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- PIMA Indians Diabetes Dataset
- scikit-learn documentation
- Flask documentation
