# Symptom-Based Disease & Doctor Recommendation System

---

## Overview

This project is an advanced symptom-to-diagnosis intelligence system engineered to deliver:

- Real-time prediction of the most probable disease condition(s)
- Intelligent specialist routing to the appropriate medical practitioner

The system accepts natural language symptom descriptions, maps them into structured binary features, and uses pre-trained XGBoost classification models to generate clinical predictions.

This project is designed for educational research, prototyping, and demonstration of ML web integration.

---

## About the Project

This system is built on a structured symptom dataset with binary feature encoding and balanced disease classes. It uses two independent XGBoost models: one for disease classification and one for specialist recommendation. XGBoost is chosen for its strong performance on tabular data, ability to capture non-linear symptom patterns, and built-in regularization for generalization on synthetic datasets.

At runtime, the pipeline ingests free-text symptoms, maps phrases to known symptom features, generates a binary feature vector, scores disease probabilities, and outputs the top result(s) with a recommended specialist.

---

## Dataset Architecture

The system is trained on a structured dataset representing common health conditions.

### Key Characteristics

- Each row represents one patient
- Symptoms are encoded as binary values (0 = absent, 1 = present)
- Dataset is balanced across disease classes
- Symptom names use layman terminology

### Dataset File

```
indian_symptom_dataset_layman_50plus_with_doctor.csv
```

### Data Schema

| fever | cough | stomach_pain | tiredness | ... | disease   | doctor_type   |
| ----- | ----- | ------------ | --------- | --- | --------- | ------------- |
| 1     | 1     | 0            | 1         | ... | pneumonia | pulmonologist |

- Input Features: Binary symptom vector
- Output Labels:

  - `disease`
  - `doctor_type`

---

## Project Structure

```
project-root/
│
├── indian_symptom_dataset_layman_50plus_with_doctor.csv
│
├── disease_model.pkl
├── doctor_model.pkl
├── disease_encoder.pkl
├── doctor_encoder.pkl
├── symptom_columns.pkl
│
├── xgbtestonterminal.py
├── xgbtest.py
│
└── README.md
```

---

## File Responsibilities

### xgbtestonterminal.py

- CLI based testing interface
- Accepts symptom input via terminal
- Runs predictions locally
- Displays ranked disease results and recommended doctor
- Used for quick model validation

---

### xgbtest.py

- Flask API implementation
- Exposes `/predict` endpoint
- Accepts JSON symptom input
- Returns structured JSON response
- Designed for integration with PHP or web applications

---

### Model Artifacts (.pkl files)

- `disease_model.pkl` - Trained XGBoost disease classifier
- `doctor_model.pkl` - Trained XGBoost specialist classifier
- `disease_encoder.pkl` - Label encoder for disease names
- `doctor_encoder.pkl` - Label encoder for specialist names
- `symptom_columns.pkl` - Feature ordering metadata

---

## Machine Learning Design

### Model Type: XGBoost (Gradient Boosted Decision Trees)

Two independent models are used:

### 1. Disease Classification Model

- Multi-class classification
- Input: Binary symptom vector
- Output: Probability distribution across disease classes
- Objective: Multi-class log loss

### 2. Specialist Recommendation Model

- Multi-class classification
- Input: Same symptom vector
- Output: Recommended medical specialist
- Operates independently from disease prediction

---

## Prediction Pipeline

1. User enters symptom description
2. Text is mapped to structured binary features
3. Feature vector passed to disease model
4. Probabilities computed across all diseases
5. Top results extracted (or highest probability)
6. Specialist model predicts doctor type
7. Results returned to CLI or API response

---

## Running the System

### Requirements

Python 3.9+

Install dependencies using the requirements file:

```
pip install -r requirements.txt
```

---

## Running CLI Version (Terminal Testing)

```
python xgbtestonterminal.py
```

Used for:

- Model validation
- Local testing
- Demonstration without web integration

---

## Running Flask API Version

```
python xgbtest.py
```

API Endpoint:

```
POST http://127.0.0.1:5000/predict
```

Example Request:

```json
{
  "symptoms": "fever cough breathing problem"
}
```

Example Response:

```json
{
  "disease": "pneumonia",
  "confidence": 72.45,
  "recommended_doctor": "pulmonologist"
}
```

This API can be integrated with:

- PHP backend
- Web frontend
- Postman testing
- Other REST clients

---

## System Architecture (Web Integration)

```
User -> PHP Website -> Flask API -> ML Models -> Flask -> PHP -> User
```

All components can run locally on the same machine.

---

## Limitations

- Not a medical diagnostic system
- No laboratory or imaging data included
- Based solely on symptom pattern recognition
- Dataset is synthetic and intended for educational use

---

## Intended Use

- Academic projects
- ML web integration demonstrations
- Health AI prototyping
- Model deployment practice

---
