import numpy as np
import joblib
import pandas as pd

model = joblib.load("student_grade_rf.pkl")



# Input Utility

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

# User Inputs

assignments = get_average_input("Enter Assignment scores: ")
projects = get_average_input("Enter Project scores: ")
midterms = get_average_input("Enter Midterm scores: ")
attendance = float(input("Attendance %: "))
study_hours = float(input("Study Hours per week: "))
sleep_hours = float(input("Sleep Hours per night: "))

calculated_default = (assignments+projects+midterms)/3
quiz = get_average_input(
    "Enter Quiz scores (comma-separated, or press Enter if none): ",
    allow_empty=True,
    default=calculated_default
)


# Derived Values

participation_score = calculate_participation(attendance, quiz, study_hours)

# Build Input DataFrame

input_dict = {
    'Attendance (%)': attendance,
    'Midterm_Score': midterms,
    'Assignments_Avg': assignments,
    'Quizzes_Avg': quiz,
    'Participation_Score': participation_score,
    'Projects_Score': projects,
    'Study_Hours_per_Week': study_hours,
    'Sleep_Hours_per_Night': sleep_hours
}

input_data = pd.DataFrame([input_dict])

# Force correct column order
input_data = input_data[model.feature_names_in_]


# Prediction (Hybrid Logic)

grade_map_rev = {
    0: "F",
    1: "D",
    2: "C",
    3: "B",
    4: "A"
}

academic_avg = np.mean([assignments, projects, midterms, quiz])

if academic_avg >90 and attendance>85 and study_hours>=10:
    final_grade = "A"
    logic_used = "Heuristic (Elite Criteria)"
else:
    p_num = model.predict(input_data)[0]
    final_grade = grade_map_rev[p_num]
    logic_used = "Random Forest AI"

print("\nPredicted Grade:", final_grade, "\nLogic: ", logic_used)
