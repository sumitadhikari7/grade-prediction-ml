import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("student_grade_rf.pkl")

st.title("Student Grade Predictor")
st.write("Predict final grade using AI + heuristic logic")