# **Symptom-Based Disease & Doctor Recommendation System**

## Overview

This project is an advanced symptom-to-diagnosis intelligence system engineered to deliver:
- Real-time prediction of the three most probable disease conditions
- Intelligent specialist routing to the appropriate medical practitioner

The system accepts natural language symptom descriptions from users, performs intelligent feature mapping to structured clinical attributes, and leverages state-of-the-art pre-trained machine learning models to generate clinical predictions. This platform is optimized for educational research, clinical prototyping, and exploratory medical AI applications.

---

## Dataset Architecture

The foundation of this project rests on a comprehensively structured medical dataset carefully curated to reflect prevalent health conditions across diverse demographic populations.

### Key Specifications
- Data Format: Single patient record per row with complete symptom profiling
- Feature Encoding: Binary-valued symptom representation (0/1) for optimal model performance
- Class Distribution: Carefully balanced across all disease categories to prevent bias
- Terminology: Accessible lay-language symptom descriptions without requiring medical expertise

### Data Source
```
indian_symptom_dataset_layman_50plus_with_doctor.csv
```

### Data Schema
- Input Features: Complete symptom vector with binary classification  
- Output Labels:
  - `disease` – clinical condition identifier
  - `doctor_type` – recommended specialist discipline

Sample Data Structure:

| fever | cough | stomach_pain | tiredness | ... | disease | doctor_type |
|------|------|--------------|-----------|-----|---------|-------------|
| 1 | 1 | 0 | 1 | ... | pneumonia | pulmonologist |

---

## System Architecture

```
project-root/
│
├── indian_symptom_dataset_layman_50plus_with_doctor.csv
│   └── [Training & inference dataset]
│
├── disease_model.pkl
│   └── [Pre-trained XGBoost disease classifier]
│
├── doctor_model.pkl
│   └── [Pre-trained XGBoost specialist recommender]
│
├── disease_encoder.pkl
├── doctor_encoder.pkl
├── symptom_columns.pkl
│   └── [Model artifacts: encoders & feature mappings]
│
├── xgbtest.py
│   └── [Inference engine & CLI interface]
│
└── README.md
    └── [Project documentation]
```

### Component Specifications

- **indian_symptom_dataset_layman_50plus_with_doctor.csv**  
  Core training dataset: symptom features linked to disease outcomes and specialist recommendations.

- **xgbtest.py**  
  Production inference module. Implements interactive command-line interface for real-time prediction. Accepts free-text symptom input and returns calibrated disease probability scores with specialist recommendations.

- **Model Artifacts (*.pkl)**  
  Serialized machine learning models, label encoders, and feature metadata. Required for deployment-stage inference.

---

## Installation & Execution

### Prerequisites

Requires Python 3.9 or later.

Install required dependencies:
```bash
pip install numpy pandas scikit-learn xgboost joblib
```

### Running the System

Execute the inference engine:
```bash
python xgbtest.py
```

Example Execution:
```
Enter symptoms: fever cough breathing problem

Predicted Conditions (Confidence Ranking):
1. pneumonia            : 62.10%
2. tuberculosis         : 21.45%
3. covid19              :  9.80%

Recommended Specialist:
pulmonologist
```

Exit the application with `Ctrl + C`.

---

## Important Disclaimers & Limitations

- **Non-Medical Device**: This system is not a medical diagnostic tool and must not be used for clinical decision-making
- **Evidence Scope**: Predictions derive exclusively from symptom pattern recognition without clinical validation
- **Incomplete Assessment**: No laboratory diagnostics, imaging, or clinical examination data are incorporated
- **Dataset Nature**: Training data is synthetically generated and designed for educational and prototyping scenarios only

---
