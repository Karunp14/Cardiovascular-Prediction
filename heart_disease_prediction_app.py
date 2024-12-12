import streamlit as st
import pandas as pd
import boto3
import joblib
from io import BytesIO
import json
import datetime

# AWS S3 Configuration
S3_BUCKET = "healthdata-storage"  # Replace with your bucket name
PREDICTION_FOLDER = "predictions/"  # Folder to store predictions in S3
MODEL_KEY = "models/cardiovascular_disease_model.pkl"  # Replace with your model path in S3
FEATURES_KEY = "models/feature_names.pkl"  # Replace with your feature names path in S3

@st.cache_resource
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["aws_credentials"]["AWS_DEFAULT_REGION"]
    )

# Initialize S3 client
s3 = boto3.client("s3")

# Function to fetch and load model and feature names from S3
@st.cache_resource
def load_model_and_features():
    # Download the model from S3
    model_obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
    model = joblib.load(BytesIO(model_obj["Body"].read()))
    
    # Download feature names from S3
    feature_obj = s3.get_object(Bucket=S3_BUCKET, Key=FEATURES_KEY)
    feature_names = joblib.load(BytesIO(feature_obj["Body"].read()))
    
    return model, feature_names

# Load model and feature names
model, feature_names = load_model_and_features()


# App title
st.title("Cardiovascular Disease Prediction")

# Input fields
sleep_hours = st.number_input("Sleep Hours (hours per day)", min_value=0.0, max_value=24.0, step=0.5)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, step=0.1)
# Categorical inputs
age_category = st.selectbox(
    "Age Category",
    [
        "Age 18 to 24",
        "Age 25 to 29",
        "Age 30 to 34",
        "Age 35 to 39",
        "Age 40 to 44",
        "Age 45 to 49",
        "Age 50 to 54",
        "Age 55 to 59",
        "Age 60 to 64",
        "Age 65 to 69",
        "Age 70 to 74",
        "Age 75 to 79",
        "Age 80 or older",
    ],
)
# Categorical inputs
sex = st.selectbox("Sex", ["Female", "Male"])
general_health = st.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"])
last_checkup_time = st.selectbox(
    "Last Checkup Time",
    [
        "5 or more years ago",
        "Within past 5 years (2 years but less than 5 years ago)",
        "Within past 2 years (1 year but less than 2 years ago)",
        "Within past year (anytime less than 12 months ago)",
    ],
)
physical_activities = st.selectbox("Physical Activities", ["Yes", "No"])
smoker_status = st.selectbox(
    "Smoker Status",
    [
        "Never smoked",
        "Former smoker",
        "Current smoker - now smokes every day",
        "Current smoker - now smokes some days",
    ],
)
had_angina = st.selectbox(
    "History of Angina", ["Yes", "No"]
)
had_heart_attack = st.selectbox(
    "History of Heart Attack", ["Yes", "No"]
)
had_stroke = st.selectbox(
    "History of Stroke", ["Yes", "No"]
)

# Map categorical inputs to one-hot encoded features
input_data = {
    "SleepHours": sleep_hours,
    "BMI": bmi,
    "AgeCategory_Age 18 to 24": 1 if age_category == "Age 18 to 24" else 0,
    "AgeCategory_Age 25 to 29": 1 if age_category == "Age 25 to 29" else 0,
    "AgeCategory_Age 30 to 34": 1 if age_category == "Age 30 to 34" else 0,
    "AgeCategory_Age 35 to 39": 1 if age_category == "Age 35 to 39" else 0,
    "AgeCategory_Age 40 to 44": 1 if age_category == "Age 40 to 44" else 0,
    "AgeCategory_Age 45 to 49": 1 if age_category == "Age 45 to 49" else 0,
    "AgeCategory_Age 50 to 54": 1 if age_category == "Age 50 to 54" else 0,
    "AgeCategory_Age 55 to 59": 1 if age_category == "Age 55 to 59" else 0,
    "AgeCategory_Age 60 to 64": 1 if age_category == "Age 60 to 64" else 0,
    "AgeCategory_Age 65 to 69": 1 if age_category == "Age 65 to 69" else 0,
    "AgeCategory_Age 70 to 74": 1 if age_category == "Age 70 to 74" else 0,
    "AgeCategory_Age 75 to 79": 1 if age_category == "Age 75 to 79" else 0,
    "AgeCategory_Age 80 or older": 1 if age_category == "Age 80 or older" else 0,
    "Sex_Female": 1 if sex == "Female" else 0,
    "Sex_Male": 1 if sex == "Male" else 0,
    "GeneralHealth_Excellent": 1 if general_health == "Excellent" else 0,
    "GeneralHealth_Very good": 1 if general_health == "Very good" else 0,
    "GeneralHealth_Good": 1 if general_health == "Good" else 0,
    "GeneralHealth_Fair": 1 if general_health == "Fair" else 0,
    "GeneralHealth_Poor": 1 if general_health == "Poor" else 0,
    "LastCheckupTime_5 or more years ago": 1 if last_checkup_time == "5 or more years ago" else 0,
    "LastCheckupTime_Within past 5 years (2 years but less than 5 years ago)": 1
    if last_checkup_time == "Within past 5 years (2 years but less than 5 years ago)"
    else 0,
    "LastCheckupTime_Within past 2 years (1 year but less than 2 years ago)": 1
    if last_checkup_time == "Within past 2 years (1 year but less than 2 years ago)"
    else 0,
    "LastCheckupTime_Within past year (anytime less than 12 months ago)": 1
    if last_checkup_time == "Within past year (anytime less than 12 months ago)"
    else 0,
    "PhysicalActivities_No": 1 if physical_activities == "No" else 0,
    "PhysicalActivities_Yes": 1 if physical_activities == "Yes" else 0,
    "SmokerStatus_Never smoked": 1 if smoker_status == "Never smoked" else 0,
    "SmokerStatus_Former smoker": 1 if smoker_status == "Former smoker" else 0,
    "SmokerStatus_Current smoker - now smokes every day": 1
    if smoker_status == "Current smoker - now smokes every day"
    else 0,
    "SmokerStatus_Current smoker - now smokes some days": 1
    if smoker_status == "Current smoker - now smokes some days"
    else 0,
    "HadAngina_Yes": 1 if had_angina == "Yes" else 0,
    "HadAngina_No": 1 if had_angina == "No" else 0,
    "HadHeartAttack_Yes": 1 if had_heart_attack == "Yes" else 0,
    "HadHeartAttack_No": 1 if had_heart_attack == "No" else 0,
    "HadStroke_Yes": 1 if had_stroke == "Yes" else 0,
    "HadStroke_No": 1 if had_stroke == "No" else 0,
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Align the input DataFrame with model features
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns with default value 0

# Ensure the input DataFrame columns are in the correct order
input_df = input_df[feature_names]

# Predict when the user clicks the "Predict" button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.error(
            f"High Risk of Cardiovascular Disease (Probability: {probability[1]*100:.2f}%)"
        )
    else:
        st.success(
            f"Low Risk of Cardiovascular Disease (Probability: {probability[0]*100:.2f}%)"
        )
    # Save the result to S3
    result_data = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "inputs": input_data,
        "prediction": int(prediction),
        "probabilities": {"low_risk": probability[0], "high_risk": probability[1]},
    }

    # Convert to JSON format
    result_json = json.dumps(result_data)

    # Create a unique filename for the prediction result
    prediction_file = f"{PREDICTION_FOLDER}prediction_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"

    # Upload the prediction result to S3
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=prediction_file, Body=result_json)
        st.success(f"Prediction result saved to S3: s3://{S3_BUCKET}/{prediction_file}")
    except Exception as e:
        st.error(f"Failed to save prediction result to S3: {str(e)}")
