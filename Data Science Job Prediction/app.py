import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("xgboost_model.joblib")

st.title("Data Science Job Success Prediction App")
st.write("Enter the candidate details to predict job success probability.")

# Input fields
city = st.number_input("City (as integer ID)", min_value=1, max_value=180)
city_development_index = st.number_input("City Development Index (e.g., 0.5 - 1.0)", min_value=0.0, max_value=1.0, step=0.01)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
relevent_experience = st.selectbox("Relevant Experience", ["Has relevant experience", "No relevant experience"])
enrolled_university = st.selectbox("Enrolled University", ["Full time course", "Part time course", "no_enrollment"])
education_level = st.selectbox("Education Level", ["Graduate", "Masters", "High School", "Phd", "Primary School"])
major_discipline = st.selectbox("Major Discipline", ["STEM", "Business Degree", "Arts", "No Major", "Other"])
experience = st.number_input("Years of Experience", min_value=0, max_value=30)
company_size = st.selectbox("Company Size", ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"])
company_type = st.selectbox("Company Type", ["Private", "Public", "Startup", "Other"])
training_hours = st.number_input("Training Hours", min_value=0, max_value=500)

# Manual Label Encoding (same order as used during model training)
gender_map = {"Male": 0, "Female": 1, "Other": 2}
relevent_experience_map = {"No relevant experience": 0, "Has relevant experience": 1}
enrolled_university_map = {"no_enrollment": 0, "Part time course": 1, "Full time course": 2}
education_level_map = {"Primary School": 0, "High School": 1, "Graduate": 2, "Masters": 3, "Phd": 4}
major_discipline_map = {"No Major": 0, "Other": 1, "Arts": 2, "Business Degree": 3, "STEM": 4}
company_size_map = {"<10": 0, "10-49": 1, "50-99": 2, "100-500": 3, "500-999": 4,
                    "1000-4999": 5, "5000-9999": 6, "10000+": 7}
company_type_map = {"Startup": 0, "Other": 1, "Public": 2, "Private": 3}

# Encoded inputs - FINAL list including all features
encoded_inputs = [
    city,
    city_development_index,
    gender_map[gender],
    relevent_experience_map[relevent_experience],
    enrolled_university_map[enrolled_university],
    education_level_map[education_level],
    major_discipline_map[major_discipline],
    experience,
    company_size_map[company_size],
    company_type_map[company_type],
    training_hours
]

features = np.array([encoded_inputs])

# Predict
if st.button("Predict Job Success"):
    prediction = model.predict(features)
    st.success(f"Prediction: {'Success' if prediction[0] == 1 else 'No Success'}")
