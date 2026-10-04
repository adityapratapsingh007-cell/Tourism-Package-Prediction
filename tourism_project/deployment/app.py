import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "tourism_project/model_building.joblib")
model = joblib.load(model_path)

st.title("tourism_Prediction App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Enter the sensor and configuration data below to get a prediction.
""")

TypeofContact       = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
Age     = st.number_input("Age", 18.0, 30.0, 40.0, 70.0)
CityTier = st.number_input("CityTier", 1.0, 2.0, 3.0 )
DurationOfPitch    = st.number_input("DurationOfPitch", 5, 50, 127)
NumberOfFollowups       = st.number_input("NumberOfFollowups", 1.0, 2.0, 5.0, 6.0)
PreferredPropertyStar       = st.number_input("PreferredPropertyStar", 1.0, 5.0, 4.0)
NumberOfPersonVisiting    = st.number_input("NumberOfPersonVisiting", 1.0, 5.0)
Occupation       = st.selectbox("Occupation", ["Salaried", "Free Lancer","Small Business","Large Business"])
Gender       = st.selectbox("Gender", ["Male", "Female"])
ProductPitched       = st.selectbox("ProductPitched", ["Basic", "Deluxe","King","Super Deluxe"])
MaritalStatus       = st.selectbox("MaritalStatus", ["Married", "Single","Divorced"])
Designation       = st.selectbox("Designation", ["Executive", "Manager","Senior Manager","AVP","VP"])
Passport       = st.selectbox("Passport", ["Yes", "No"])
pit_score = st.number_input("PitchSatisfactionScore", 1.0, 5.0, 4.0)
OwnCar       = st.selectbox("OwnCar", ["Yes", "No"])
NumberOfTrips       = st.number_input("NumberOfTrips", 1.0, 5.0, 4.0)
NumberOfChildrenVisiting       = st.number_input("NumberOfChildrenVisiting", 1.0, 5.0, 4.0)
MonthlyIncome       = st.number_input("MonthlyIncome", 1000.0, 16129.0, 98678.0)


input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": City_Tier,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "Occupation": Occupation,
    "Gender": Gender,
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
    "Passport": Passport,
    "PitchSatisfactionScore": pit_score,
    "OwnCar": OwnCar,
    "NumberOfTrips": NumberOfTrips,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "TypeofContact": TypeofContact,
    }])

if st.button("Predict ProdTaken"):
    prediction = model.predict(input_data)[0]
    result = "ProdTaken" if prediction == 1 else "Not taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
