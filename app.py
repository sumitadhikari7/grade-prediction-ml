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
