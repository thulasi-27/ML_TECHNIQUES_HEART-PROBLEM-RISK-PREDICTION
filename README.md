# Multimodal Heart Failure Risk Stratification & CDS Pipeline

An end-to-end Clinical Decision Support (CDS) system that fuses static Electronic Health Records (EHR) and temporal Electrocardiogram (ECG) waveforms to predict acute heart failure exacerbation. 

This project bridges the gap between deep learning mathematics and actionable medical guidelines by combining a **Dual-Branch Neural Network** with a **Deterministic Clinical Rules Engine** and a **Generative LLM Scribe**.

## System Architecture

The pipeline is built on a heavily decoupled microservice architecture:

1. **Dual-Branch AI Engine (Python/TensorFlow):** - **Branch 1:** A Multi-Layer Perceptron (MLP) processing 11 critical EHR clinical features.
   - **Branch 2:** A 1D-Convolutional Neural Network (1D-CNN) processing 1000-step Lead II ECG waveforms.
   - **Fusion:** A Gated Attention Mechanism dynamically weights the systemic metabolic data against the electrical arrhythmic data.
2. **Deterministic Rules Engine:** Maps the raw AI probability (0.0 - 1.0) and patient Ejection Fraction (EF%) to strict ACC/AHA guidelines and NYHA Functional Classes to ensure clinical safety.
3. **API Gateway (Node.js/Express):** Asynchronous traffic controller handling clinical payload routing.
4. **Medical LLM Scribe (Gemini 2.5 Flash):** Translates the deterministic rule outputs and SHAP explainability metrics into a human-readable, physician-facing narrative.

## Tech Stack

* **Machine Learning:** TensorFlow 2.x, Keras, Scikit-Learn, SHAP
* **AI Microservice:** Python, Flask, NumPy, Pandas
* **Backend Gateway:** Node.js, Express.js
* **LLM Integration:** Google Gemini 2.5 Flash API
* **Frontend (Planned):** React.js, Tailwind CSS
* **Dataset:** MIMIC-IV and MIMIC-IV-ECG (PhysioNet)

##  Key Features

* **Multimodal Inference:** Fuses tabular and time-series data for higher AUROC performance than unimodal baselines.
* **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) values to identify the localized mathematical drivers of a patient's risk score (e.g., elevated Serum Creatinine).
* **Failsafe Tensor Validation:** Strict API-level validation rejects mismatched payloads before inference to prevent critical UI crashes.
* **Zero-Hallucination Diagnosis:** The Generative LLM acts *strictly* as a formatting scribe, relying 100% on the deterministic Python rules engine for medical mapping.

## ⚙️ Local Setup & Installation

### 1. Python AI Microservice
```bash
# Navigate to the backend directory
cd hf-mern-backend

# Install dependencies (ensure you have a virtual environment active)
pip install flask tensorflow numpy pandas scikit-learn shap

# Run the Flask server (runs on Port 5000)
python app.py