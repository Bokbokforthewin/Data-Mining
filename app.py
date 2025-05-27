import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Configuration ---
MODEL_PATH = 'model_RF.pkl' # Ensure this path is correct

# --- Load the model (using Streamlit's caching for efficiency) ---
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model(MODEL_PATH)

# --- Mappings for Display (Make sure these match your dataset's codebook) ---
AGE_MAP = {
    1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44',
    6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69',
    11: '70-74', 12: '75-79', 13: '80+'
}

EDUCATION_MAP = {
    1: 'Never Attended School', 2: 'Elementary', 3: 'Some High School',
    4: 'High School Graduate', 5: 'Some College/Technical', 6: 'College Graduate'
}

INCOME_MAP = {
    1: '<$10,000', 2: '$10,000 - $14,999', 3: '$15,000 - $19,999',
    4: '$20,000 - $24,999', 5: '$25,000 - $34,999', 6: '$35,000 - $49,999',
    7: '$50,000 - $74,999', # FIX: Changed single quotes to double quotes for consistency and to avoid error
    8: '>= $75,000'
}

# --- New: Mappings for input to model (strings to numbers) ---
YES_NO_INPUT_MAP = {'Yes': 1, 'No': 0}
SEX_INPUT_MAP = {'Male': 1, 'Female': 0}

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Diabetes Health Predictor")

st.title('🩺 Diabetes Health Status Predictor')
st.markdown("""
    This application helps predict an individual's likelihood of being **prediabetic or diabetic**
    based on a comprehensive set of health and lifestyle indicators.
    Please fill in the details below.
""")

st.markdown('---')

# --- Section 1: Demographics ---
st.header('👤 Demographics')

# --- Name Input ---
user_name = st.text_input(
    'Your Name',
    value=" ", # Default value
    max_chars=50,
    help="Enter your name to personalize the prediction message.",
    key='user_name_input'
)

col_age, col_sex, col_bmi = st.columns(3)

with col_age:
    age = st.selectbox(
        'Age Category',
        options=list(AGE_MAP.keys()),
        format_func=lambda x: AGE_MAP[x],
        help="Select your age range based on the CDC categories.",
        key='age'
    )
with col_sex:
    # Changed to st.radio
    sex_display = st.radio(
        '**Biological Sex**', # Question in bold
        options=['Female', 'Male'], # Display options
        horizontal=True,
        key='sex_radio'
    )
    sex = SEX_INPUT_MAP[sex_display] # Map back to numerical for model

with col_bmi:
    bmi = st.number_input(
        'Body Mass Index (BMI)',
        min_value=10, max_value=100, value=25,
        help="Enter your Body Mass Index (BMI). Calculated as weight (kg) / [height (m)]^2.",
        key='bmi'
    )

col_edu, col_income = st.columns(2)
with col_edu:
    education = st.selectbox(
        'Education Level',
        options=list(EDUCATION_MAP.keys()),
        format_func=lambda x: EDUCATION_MAP[x],
        help="Your highest level of education.",
        key='education'
    )
with col_income:
    income = st.selectbox(
        'Income Level',
        options=list(INCOME_MAP.keys()),
        format_func=lambda x: INCOME_MAP[x],
        help="Your household income level.",
        key='income'
    )

st.markdown('---')

# --- Section 2: Current Health Conditions & History ---
st.header('⚕️ Health Conditions & History')
col_bp, col_chol, col_cholcheck = st.columns(3)
with col_bp:
    # Changed to st.radio
    high_bp_display = st.radio(
        '**Diagnosed with High Blood Pressure?**',
        options=['No', 'Yes'], # Display options
        horizontal=True,
        key='high_bp_radio'
    )
    high_bp = YES_NO_INPUT_MAP[high_bp_display] # Map back to numerical

with col_chol:
    # Changed to st.radio
    high_chol_display = st.radio(
        '**Diagnosed with High Cholesterol?**',
        options=['No', 'Yes'],
        horizontal=True,
        key='high_chol_radio'
    )
    high_chol = YES_NO_INPUT_MAP[high_chol_display] # Map back to numerical

with col_cholcheck:
    # Changed to st.radio
    chol_check_display = st.radio(
        '**Had Cholesterol Check in the past 5 years?**',
        options=['No', 'Yes'],
        horizontal=True,
        key='chol_check_radio'
    )
    chol_check = YES_NO_INPUT_MAP[chol_check_display] # Map back to numerical

col_heart, col_stroke, col_diffwalk = st.columns(3)
with col_heart:
    # Changed to st.radio
    heart_disease_attack_display = st.radio(
        '**Ever had Coronary Heart Disease or Myocardial Infarction?**',
        options=['No', 'Yes'],
        horizontal=True,
        key='heart_disease_attack_radio'
    )
    heart_disease_attack = YES_NO_INPUT_MAP[heart_disease_attack_display] # Map back to numerical

with col_stroke:
    # Changed to st.radio
    stroke_display = st.radio(
        '**Ever told you had a Stroke?**',
        options=['No', 'Yes'],
        horizontal=True,
        key='stroke_radio'
    )
    stroke = YES_NO_INPUT_MAP[stroke_display] # Map back to numerical

with col_diffwalk:
    # Changed to st.radio
    diff_walk_display = st.radio(
        '**Serious difficulty walking or climbing stairs?**',
        options=['No', 'Yes'],
        horizontal=True,
        key='diff_walk_radio'
    )
    diff_walk = YES_NO_INPUT_MAP[diff_walk_display] # Map back to numerical

st.markdown('---')

# --- Section 3: General Health & Lifestyle ---
st.header('🏃‍♀️ General Health & Lifestyle')
col_genhlth, col_physhlth, col_menthlth = st.columns(3)

with col_genhlth:
    gen_hlth = st.slider(
        '**In general, how would you rate your health?**', # Question in bold
        min_value=1, max_value=5, value=3,
        help="1=Excellent, 5=Poor",
        key='gen_hlth'
    )
with col_physhlth:
    phys_hlth = st.number_input(
        '**Days of poor physical health in past 30 days:**', # Question in bold
        min_value=0, max_value=30, value=0,
        help="Number of days your physical health was not good (past 30 days).",
        key='phys_hlth'
    )
with col_menthlth:
    ment_hlth = st.number_input(
        '**Days of poor mental health in past 30 days:**', # Question in bold
        min_value=0, max_value=30, value=0,
        help="Number of days your mental health was not good (past 30 days).",
        key='ment_hlth'
    )

col_smoke, col_physact, col_fruits = st.columns(3)
with col_smoke:
    # Changed to st.radio
    smoker_display = st.radio(
        '**Smoked at least 100 cigarettes in your lifetime?**',
        options=['No', 'Yes'],
        horizontal=True,
        key='smoker_radio'
    )
    smoker = YES_NO_INPUT_MAP[smoker_display] # Map back to numerical

with col_physact:
    # Changed to st.radio
    phys_activity_display = st.radio(
        '**Engaged in physical activity in past 30 days (not job-related)?**',
        options=['No', 'Yes'],
        horizontal=True,
        key='phys_activity_radio'
    )
    phys_activity = YES_NO_INPUT_MAP[phys_activity_display] # Map back to numerical

with col_fruits:
    # Changed to st.radio
    fruits_display = st.radio(
        '**Consume fruits 1 or more times per day?**',
        options=['No', 'Yes'],
        horizontal=True,
        key='fruits_radio'
    )
    fruits = YES_NO_INPUT_MAP[fruits_display] # Map back to numerical

col_veggies, col_alcohol, col_healthcare = st.columns(3)
with col_veggies:
    # Changed to st.radio
    veggies_display = st.radio(
        '**Consume vegetables 1 or more times per day?**',
        options=['No', 'Yes'],
        horizontal=True,
        key='veggies_radio'
    )
    veggies = YES_NO_INPUT_MAP[veggies_display] # Map back to numerical

with col_alcohol:
    # Changed to st.radio
    hvy_alcohol_consump_display = st.radio(
        '**Heavy alcohol consumption (men >14, women >7 drinks/week)?**',
        options=['No', 'Yes'],
        horizontal=True,
        key='hvy_alcohol_consump_radio'
    )
    hvy_alcohol_consump = YES_NO_INPUT_MAP[hvy_alcohol_consump_display] # Map back to numerical

with col_healthcare:
    # Changed to st.radio
    any_healthcare_display = st.radio(
        '**Have any kind of healthcare coverage?**',
        options=['No', 'Yes'],
        horizontal=True,
        key='any_healthcare_radio'
    )
    any_healthcare = YES_NO_INPUT_MAP[any_healthcare_display] # Map back to numerical

col_nodoc = st.columns(1)[0] # Single column for the last input
with col_nodoc:
    # Changed to st.radio
    no_doc_bc_cost_display = st.radio(
        '**Was there a time in past 12 months you needed to see a doctor but could not due to cost?**',
        options=['No', 'Yes'],
        horizontal=True,
        key='no_doc_bc_cost_radio'
    )
    no_doc_bc_cost = YES_NO_INPUT_MAP[no_doc_bc_cost_display] # Map back to numerical

st.markdown('---')

# --- Prediction Button ---
if st.button('✨ Get Prediction', key='predict_button', type="primary"):
    # Retrieve the name entered by the user
    entered_name = st.session_state.get('user_name_input', 'Person') # Default to 'Person' if no name entered
    if not entered_name.strip(): # Handle cases where user might leave name blank
        entered_name = "Person"

    # Create a DataFrame from the input values
    # IMPORTANT: The column names and order MUST match how your model was trained.
    feature_columns = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
        'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
        'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    input_data = pd.DataFrame([[
        high_bp, high_chol, chol_check, bmi, smoker, stroke,
        heart_disease_attack, phys_activity, fruits, veggies, hvy_alcohol_consump,
        any_healthcare, no_doc_bc_cost, gen_hlth, ment_hlth, phys_hlth,
        diff_walk, sex, age, education, income
    ]], columns=feature_columns)

    # --- Apply any necessary preprocessing (e.g., scaling, one-hot encoding) ---
    # IMPORTANT: If your model was trained on scaled or one-hot encoded data,
    # you MUST apply the SAME transformations to `input_data` here.
    # For example, if 'BMI' was scaled:
    # from sklearn.preprocessing import StandardScaler # You'd need to save/load this scaler
    # scaler = load_scaler_from_pickle('your_scaler.pkl') # Load your pre-trained scaler
    # input_data['BMI'] = scaler.transform(input_data[['BMI']])

    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data) # Get probabilities

        st.subheader('🔮 Prediction Results:')

        predicted_class = prediction[0]
        # Confidence is the probability of the predicted class
        confidence = prediction_proba[0][predicted_class]

        if predicted_class == 0:
            st.balloons() # Fun animation for a positive outcome
            st.success(f"**{entered_name}**, you are predicted to be **Healthy / No Diabetes**.")
            st.info(f"**Confidence Level:** {confidence:.2%}") # Display confidence as percentage
        elif predicted_class == 1:
            st.warning(f"**{entered_name}**, you are predicted to be **Prediabetic or Diabetic**.")
            st.info(f"**Confidence Level:** {confidence:.2%}") # Display confidence as percentage
        else:
            st.info(f"**{entered_name}**, unrecognized prediction outcome: {predicted_class}") # Fallback

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure all inputs are correct and match the format your model expects.")

st.markdown("""
---
*Disclaimer: This tool is for informational purposes only and should not be used as a substitute for professional medical advice. Always consult with a healthcare professional for diagnosis and treatment.*
""")