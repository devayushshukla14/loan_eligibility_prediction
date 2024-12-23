from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('loan_model.pkl')

@app.route('/')
def home():
    return "Welcome to the Loan Eligibility Prediction System!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract JSON data
        data = request.json
        income = data['income']
        credit_score = data['credit_score']
        loan_amount = data['loan_amount']

        # Model prediction
        features = np.array([[income, credit_score, loan_amount]])
        prediction = model.predict(features)

        # Return result
        result = 'Approved' if prediction[0] == 1 else 'Not Approved'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
