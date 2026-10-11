from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

# 1. Initialize the Flask App
app = Flask(__name__)
CORS(app)

# === NEW: Correct Health Check Route (Home Page) ===
@app.route('/', methods=['GET'])
def health_check():
    return "✅ AI Microservice is awake and listening for data!"

# 2. Load the AI Model into memory
print("Loading Multimodal Model... This might take a few seconds.")
try:
    model = tf.keras.models.load_model('multimodal_hf_prototype.keras')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 3. The Clinical Rules Engine
def generate_clinical_report(ef_value, ai_risk_score):
    report = {}
    if ef_value <= 40:
        report['hf_type'] = "HFrEF (Left-Sided: Reduced Ejection Fraction)"
        report['mechanism'] = "Systolic Failure: The heart muscle is weak and cannot contract effectively."
    elif 41 <= ef_value <= 49:
        report['hf_type'] = "HFmrEF (Left-Sided: Mid-range Ejection Fraction)"
        report['mechanism'] = "Mild Systolic Dysfunction: Pumping is slightly below normal."
    else:
        report['hf_type'] = "HFpEF (Left-Sided: Preserved Ejection Fraction)"
        report['mechanism'] = "Diastolic Failure: The heart muscle is stiff and cannot relax to fill properly."

    if ai_risk_score < 0.25:
        report['nyha_class'] = "Class I (No limitations)"
        report['stage'] = "Stage A / B"
        report['suggestion'] = "Focus on prevention. Manage blood pressure, lifestyle modifications, and monitor lipids."
    elif 0.25 <= ai_risk_score < 0.50:
        report['nyha_class'] = "Class II (Slight limitation)"
        report['stage'] = "Stage C"
        report['suggestion'] = "Initiate Guideline-Directed Medical Therapy (GDMT). Prescribe Beta-blockers and SGLT2 inhibitors."
    elif 0.50 <= ai_risk_score < 0.75:
        report['nyha_class'] = "Class III (Marked limitation)"
        report['stage'] = "Stage C (Advanced)"
        report['suggestion'] = "Optimize GDMT dosages. Prescribe loop diuretics for pulmonary congestion. Evaluate for ICD/CRT-D."
    else: 
        report['nyha_class'] = "Class IV (Symptoms at rest)"
        report['stage'] = "Stage D (End-stage)"
        report['suggestion'] = "Requires specialized interventions. Evaluate for mechanical circulatory support (LVAD) or transplant."
    return report

# 4. Create the API Endpoint (Correctly named /analyze)
@app.route('/analyze', methods=['POST'])
def analyze_patient():
    try:
        data = request.get_json()
        
        # We need the unscaled EF value for the clinical rules engine
        raw_ef_value = float(data.get('ejection_fraction', 50.0)) 
        
        ehr_features = np.array(data['ehr_data']).reshape(1, -1)
        
        if 'ecg_data' in data:
            ecg_features = np.array(data['ecg_data']).reshape(1, 1000, 1)
        else:
            ecg_features = np.zeros((1, 1000, 1))

        # Pass the data through the Y-shaped Neural Network
        risk_prediction = float(model.predict([ehr_features, ecg_features])[0][0])

        # Run the Rules Engine
        clinical_results = generate_clinical_report(raw_ef_value, risk_prediction)

        # Send the complete package back to Node.js
        return jsonify({
            'status': 'success',
            'ai_metrics': {
                'risk_score_raw': risk_prediction,
                'risk_percentage': round(risk_prediction * 100, 1)
            },
            'clinical_report': clinical_results
        }), 200

    except Exception as e:
        print(f"❌ PYTHON CRASH CAUSE: {str(e)}") # <--- We added this!
        return jsonify({'status': 'error', 'message': str(e)}), 400

# Run the server on port 5000
if __name__ == '__main__':
    app.run(debug=True, port=5000)