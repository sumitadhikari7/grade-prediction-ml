QUIZ_DEFAULT = 74.836214

import numpy as np
import joblib
import pandas as pd

model = joblib.load("student_grade_rf.pkl")

# For multiple inputs and calculate average

def get_average_input(prompt, allow_empty=False, default=None):
    user_input = input(prompt)

    if user_input.strip() == "":
        if allow_empty and default is not None:
            return default
        else:
            raise ValueError("Input cannot be empty")

    try:
        scores = [float(x.strip()) for x in user_input.split(",")]
        return sum(scores) / len(scores)
    except:
        print("Invalid input. Provide comma-separated numbers.")
        return get_average_input(prompt, allow_empty, default)

# Engineered Features

def calculate_participation(attendance, quiz_avg, study_hours):
    study_component = np.clip((study_hours/30)*100, 0, 100)

    participation = (
        0.5 * attendance +
        0.3 * quiz_avg +
        0.2 * study_component
    )

    return np.clip(participation, 0, 100)

def calculate_stress(midterm_avg, study_hours, sleep_hours):
    # Academic stress
    academic_stress = 100 - midterm_avg

    # Burnout stress (weekly hours beyond 35)
    burnout_stress = np.clip(((study_hours-35)/15)*100, 0, 100)

    # Sleep stress (ideal 7 hrs)
    sleep_stress = np.clip(((7-sleep_hours)/7)*100, 0, 100)

    combined = (
        0.5 * academic_stress +
        0.3 * sleep_stress +
        0.2 * burnout_stress
    )

    # Convert 0–100 scale → 1–10 scale (because model expects 1–10)
    stress_1_to_10 = round((combined / 100) * 9 + 1)

    return int(np.clip(stress_1_to_10, 1, 10))

# User Inputs

assignments = get_average_input("Enter Assignment scores: ")
projects = get_average_input("Enter Project scores: ")
midterms = get_average_input("Enter Midterm scores: ")
attendance = float(input("Attendance %: "))
study_hours = float(input("Study Hours per week: "))
sleep_hours = float(input("Sleep Hours per night: "))
age = int(input("Enter your age: "))

quiz = get_average_input(
    "Enter Quiz scores (comma-separated, or press Enter if none): ",
    allow_empty=True,
    default=QUIZ_DEFAULT
)

# Derived Values
participation_score = calculate_participation(attendance, quiz, study_hours)
stress_level = calculate_stress(midterms, study_hours, sleep_hours)

# Fixed Features
internet_access = 1
parent_edu = 3
family_income = 2
extracurricular = 1

# Build Input DataFrame
input_dict = {
    'Assignments_Avg': assignments,
    'Projects_Score': projects,
    'Midterm_Score': midterms,
    'Attendance (%)': attendance,
    'Study_Hours_per_Week': study_hours,
    'Quizzes_Avg': quiz,
    'Participation_Score': participation_score,
    'Internet_Access_at_Home': internet_access,
    'Parent_Education_Level': parent_edu,
    'Family_Income_Level': family_income,
    'Stress_Level (1-10)': stress_level,
    'Sleep_Hours_per_Night': sleep_hours,
    'Extracurricular_Activities': extracurricular,
    'Age': age
}

input_data = pd.DataFrame([input_dict])

# Force correct column order
input_data = input_data[model.feature_names_in_]

# Prediction
prediction = model.predict(input_data)[0]

grade_map_rev = {
    0: "F",
    1: "D",
    2: "C",
    3: "B",
    4: "A"
}

print("\nPredicted Grade:", grade_map_rev[prediction])

# print("\nFeature Importances:")
# for name, importance in zip(model.feature_names_in_, model.feature_importances_):
#     print(f"{name:30} {importance:.4f}")