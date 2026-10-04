import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model.joblib",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    page_icon="🌴",
)

st.title("🌴 Wellness Tourism Package Prediction")
st.write(
    "Enter customer and interaction details to predict whether "
    "the customer is likely to purchase the Wellness Tourism Package."
)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"],
)

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=36,
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3],
)

DurationOfPitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    max_value=150,
    value=15,
)

NumberOfFollowups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=10,
    value=4,
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5],
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=3,
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"],
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"],
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced"],
)

NumberOfTrips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=30,
    value=3,
)

Passport = st.selectbox(
    "Passport",
    ["Yes", "No"],
)

PitchSatisfactionScore = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5],
)

OwnCar = st.selectbox(
    "Own Car",
    ["Yes", "No"],
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=1,
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0,
    max_value=200000,
    value=23000,
)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

st.subheader("Input Data")
st.dataframe(input_data)

if st.button("Predict ProdTaken"):
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "Customer is likely to purchase the Wellness Tourism Package."
        )
    else:
        st.info(
            "Customer is unlikely to purchase the Wellness Tourism Package."
        )

    st.metric(
        "Purchase Probability",
        f"{probability:.2%}",
    )
