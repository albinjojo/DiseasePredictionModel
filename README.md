# Symptom-Based Disease & Doctor Recommendation System

## Overview

This project is a **terminal-based symptom checker** that predicts:
- The **top 3 most likely diseases**
- The **recommended medical specialist** to consult

Users enter symptoms in plain language. The system maps those inputs to structured symptom features and produces predictions using **pre-trained models**.

This project is intended for **educational, research, and prototyping purposes**.

---

## Dataset Description

The dataset used in this project is a **synthetic, structured medical dataset** inspired by common health conditions in the Indian population.

### Key Characteristics
- Each row represents **one patient**
- Symptoms are encoded as **binary values (0 / 1)**
- Dataset is **balanced across diseases**
- Symptom names are **layman-friendly**, not clinical jargon

### Dataset File
```
indian_symptom_dataset_layman_50plus_with_doctor.csv
```

### Dataset Structure
- **Input features**: Binary symptom columns  
- **Target labels**:
  - `disease` – disease name
  - `doctor_type` – recommended medical specialist

Example (simplified):

| fever | cough | stomach_pain | tiredness | ... | disease | doctor_type |
|------|------|--------------|-----------|-----|---------|-------------|
| 1 | 1 | 0 | 1 | ... | pneumonia | pulmonologist |

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
├── xgbtest.py
│
└── README.md
```

### File Descriptions

- **indian_symptom_dataset_layman_50plus_with_doctor.csv**  
  Dataset containing symptoms, diseases, and doctor types.

- **xgbtest.py**  
  Interactive terminal-based inference script.  
  Accepts free-text symptom input and outputs disease predictions and doctor recommendations.

- **\*.pkl files**  
  Pre-trained models, label encoders, and symptom column metadata required for inference.

---

## How to Run

### 1. Prerequisites

Ensure Python **3.9 or higher** is installed.

Install required dependencies:
```bash
pip install numpy pandas scikit-learn xgboost joblib
```

---

### 2. Run the Application

Start the terminal-based prediction system:
```bash
python xgbtest.py
```

Example interaction:
```
Enter symptoms: fever cough breathing problem

Possible Conditions (Top 3):
1. pneumonia            : 62.10%
2. tuberculosis         : 21.45%
3. covid19              :  9.80%

Recommended Specialist:
pulmonologist
```

Type `ctrl + c` to close the program.

---

## Notes and Limitations

- This system **does not provide a medical diagnosis**
- Predictions are based only on symptom patterns
- No laboratory tests or clinical confirmation are included
- The dataset is **synthetic** and intended for learning and prototyping

---
