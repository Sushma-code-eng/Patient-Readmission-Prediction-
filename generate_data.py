import pandas as pd
import numpy as np

np.random.seed(42)
n = 3000

age = np.random.randint(18, 90, n)
gender = np.random.choice(['Male', 'Female'], n)
race = np.random.choice(['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'], n, p=[0.45, 0.25, 0.15, 0.10, 0.05])
admission_type = np.random.choice(['Emergency', 'Urgent', 'Elective'], n, p=[0.50, 0.30, 0.20])
num_lab_procedures = np.random.randint(1, 120, n)
num_procedures = np.random.randint(0, 6, n)
num_medications = np.random.randint(1, 50, n)
num_outpatient = np.random.poisson(0.5, n)
num_inpatient = np.random.poisson(0.3, n)
num_emergency = np.random.poisson(0.2, n)
num_diagnoses = np.random.randint(1, 16, n)
time_in_hospital = np.random.randint(1, 14, n)
a1c_result = np.random.choice(['>7', '>8', 'Normal', 'None'], n, p=[0.15, 0.10, 0.25, 0.50])
glucose_serum = np.random.choice(['>200', '>300', 'Normal', 'None'], n, p=[0.05, 0.03, 0.12, 0.80])
diagnosis_category = np.random.choice(['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Musculoskeletal', 'Injury', 'Genitourinary', 'Neoplasms', 'Other'], n, p=[0.25, 0.12, 0.12, 0.15, 0.08, 0.08, 0.07, 0.06, 0.07])
insulin = np.random.choice(['Steady', 'Up', 'Down', 'No'], n, p=[0.30, 0.10, 0.05, 0.55])
diabetic_med = np.random.choice(['Yes', 'No'], n, p=[0.55, 0.45])
change_in_meds = np.random.choice(['Yes', 'No'], n, p=[0.45, 0.55])
discharge = np.random.choice(['Home', 'Short-term Hospital', 'SNF', 'Home Health Service', 'Other'], n, p=[0.55, 0.08, 0.12, 0.18, 0.07])

prob = np.full(n, 0.12)
prob += np.where(age > 65, 0.08, 0)
prob += np.where(age > 80, 0.05, 0)
prob += num_inpatient * 0.06
prob += num_emergency * 0.04
prob += np.where(time_in_hospital > 7, 0.06, 0)
prob += np.where(time_in_hospital > 10, 0.04, 0)
prob += np.where(np.array(a1c_result) == '>8', 0.08, 0)
prob += np.where(np.array(a1c_result) == '>7', 0.04, 0)
prob += np.where(np.array(diagnosis_category) == 'Circulatory', 0.05, 0)
prob += np.where(np.array(diagnosis_category) == 'Diabetes', 0.04, 0)
prob += np.where(num_diagnoses > 8, 0.05, 0)
prob += np.where(np.array(discharge) == 'SNF', 0.06, 0)
prob += np.where(np.array(discharge) == 'Short-term Hospital', 0.08, 0)
prob += np.where(num_medications > 30, 0.04, 0)
prob += np.where(np.array(change_in_meds) == 'Yes', 0.03, 0)
prob += np.where(np.array(insulin) == 'Up', 0.04, 0)
prob += np.random.normal(0, 0.03, n)
prob = np.clip(prob, 0.02, 0.95)
readmitted = np.random.binomial(1, prob)

df = pd.DataFrame({
    'age': age, 'gender': gender, 'race': race, 'admission_type': admission_type,
    'time_in_hospital': time_in_hospital, 'num_lab_procedures': num_lab_procedures,
    'num_procedures': num_procedures, 'num_medications': num_medications,
    'num_outpatient_visits': num_outpatient, 'num_inpatient_visits': num_inpatient,
    'num_emergency_visits': num_emergency, 'num_diagnoses': num_diagnoses,
    'a1c_result': a1c_result, 'glucose_serum_test': glucose_serum,
    'primary_diagnosis': diagnosis_category, 'insulin': insulin,
    'diabetic_medication': diabetic_med, 'change_in_medications': change_in_meds,
    'discharge_disposition': discharge, 'readmitted_30_days': readmitted
})

df.to_csv('/home/claude/patient-readmission-prediction/data/patient_readmission.csv', index=False)
print(f"Dataset: {df.shape}")
print(f"Readmission rate: {readmitted.mean():.1%} ({readmitted.sum()} / {n})")
