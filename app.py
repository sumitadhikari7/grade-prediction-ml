import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("student_grade_rf.pkl")

st.title("Student Grade Predictor")
st.write("Predict final grade using AI + heuristic logic")

def get_average_input(input_string):
    try:
        scores = [float(x.strip()) for x in input_string.split(",")]
        return sum(scores) / len(scores)
    except:
        return None
    
def calculate_participation(attendance, quiz_avg, study_hours):
    study_component = np.clip((study_hours / 30) * 100, 0, 100)

    participation = (
        0.5 * attendance +
        0.3 * quiz_avg +
        0.2 * study_component
    )

    return np.clip(participation, 0, 100)

st.subheader("Enter Academic Scores")

assignments_input = st.text_input("Assignment Scores (comma separated)", "80,85,90")
projects_input = st.text_input("Project Scores (comma separated)", "85,90")
midterms_input = st.text_input("Midterm Scores (comma separated)", "78,82")
quiz_input = st.text_input("Quiz Scores (comma separated, optional)", "")

attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0)
study_hours = st.slider("Study Hours per Week", 0.0, 40.0, 10.0)
sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)


if st.button("Predict Grade"):

    assignments = get_average_input(assignments_input)
    projects = get_average_input(projects_input)
    midterms = get_average_input(midterms_input)

    if assignments is None or projects is None or midterms is None:
        st.error("Invalid input format. Use comma separated numbers.")
        st.stop()

    calculated_default = (assignments + projects + midterms) / 3

    if quiz_input.strip() == "":
        quiz = calculated_default
    else:
        quiz = get_average_input(quiz_input)
        if quiz is None:
            st.error("Invalid quiz input.")
            st.stop()

    
    participation_score = calculate_participation(attendance, quiz, study_hours)

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
    input_data = input_data[model.feature_names_in_]