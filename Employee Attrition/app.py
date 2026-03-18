import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- STEP 1: MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="HR Strategic Dashboard", layout="wide")

# --- STEP 2: LOAD MODEL DATA ---
@st.cache_resource
def load_attrition_data():
    # Make sure this filename matches exactly what you saved in Notebook
    with open('attrition_system.pkl', 'rb') as f:
        return pickle.load(f)

# Initialize global variables
try:
    data_bundle = load_attrition_data()
    model = data_bundle['model']
    feature_names = data_bundle['feature_names']
except FileNotFoundError:
    st.error("Error: 'attrition_system.pkl' not found. Please run the Notebook code first.")
    st.stop()

# --- STEP 3: UI HEADER ---
st.title("🏢 Employee Attrition Intelligence System")
st.subheader("Final Year Project - HR Decision Support")
st.markdown("---")

# --- STEP 4: SIDEBAR INPUTS ---
st.sidebar.header("📝 Employee Profile Data")

def get_user_inputs():
    inputs = {}
    
    # Core Sliders for Prediction
    inputs['Age'] = st.sidebar.slider('Age', 18, 60, 35)
    inputs['MonthlyIncome'] = st.sidebar.slider('Monthly Income ($)', 1000, 20000, 8000)
    
    overtime_choice = st.sidebar.selectbox('Overtime', ['No', 'Yes'])
    inputs['OverTime'] = 1 if overtime_choice == 'Yes' else 0
    
    inputs['JobSatisfaction'] = st.sidebar.slider('Job Satisfaction (1-4)', 1, 4, 3)
    inputs['StockOptionLevel'] = st.sidebar.slider('Stock Option Level (0-3)', 0, 3, 1)
    inputs['YearsAtCompany'] = st.sidebar.slider('Years At Company', 0, 40, 5)
    inputs['TotalWorkingYears'] = st.sidebar.slider('Total Working Years', 0, 40, 10)
    inputs['WorkLifeBalance'] = st.sidebar.slider('Work Life Balance (1-4)', 1, 4, 3)
    
    # Logic to align with Notebook Features
    full_input_dict = {}
    for name in feature_names:
        if name in inputs:
            full_input_dict[name] = inputs[name]
        else:
            # Set hidden features to a neutral/common value to prevent bias
            full_input_dict[name] = 0 
            
    # Convert to DataFrame with exact column order from training
    final_df = pd.DataFrame([full_input_dict])[feature_names]
    return final_df, overtime_choice, inputs['MonthlyIncome']

input_df, ot_label, income_val = get_user_inputs()

# --- STEP 5: PREDICTION ENGINE ---
if st.button('🔍 Run Attrition Diagnostic'):
    # Get Probability
    probability = model.predict_proba(input_df)[0][1] * 100
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if probability > 60:
            st.error(f"### 🚨 High Risk Detected: {probability:.1f}%")
            st.write("**Assessment:** High probability of resignation.")
        elif probability > 35:
            st.warning(f"### ⚠️ Medium Risk Detected: {probability:.1f}%")
            st.write("**Assessment:** Moderate risk; employee needs engagement.")
        else:
            st.success(f"### ✅ Low Risk / Stable: {probability:.1f}%")
            st.write("**Assessment:** Strong alignment with retention factors.")
            st.balloons()

    with col2:
        st.subheader("💡 Strategic Insights")
        if ot_label == 'Yes':
            st.info("• **Workload:** Reducing Overtime can decrease risk significantly.")
        if income_val < 5000:
            st.info("• **Compensation:** Salary is below the stability benchmark.")
        if probability < 35:
            st.write("• **Retention Logic:** Positive factors like work-life balance and stability are dominating the profile.")

st.markdown("---")
st.caption("AI Model: Random Forest Classifier | Domain: Strategic Human Resources")



# --------------------------------
# import streamlit as st  # Streamlit library import ki, web interface banane ke liye
# import pandas as pd     # Pandas data ko table/dataframe format mein handle karne ke liye
# import numpy as np      # Numpy mathematical calculations aur array handling ke liye
# import pickle           # Pickle save kiye huye model file ko load karne ke liye

# # --- PAGE CONFIGURATION ---
# # Ye line browser ke tab ka naam aur layout (wide matlab puri screen) set karti hai
# # Ye hamesha sabse pehli Streamlit command honi chahiye
# st.set_page_config(page_title="HR Strategic Dashboard", layout="wide")

# # --- LOAD MODEL DATA ---
# # '@st.cache_resource' ka matlab hai ki model baar-baar load na ho, ek baar memory mein save ho jaye
# @st.cache_resource
# def load_attrition_data():
#     # Humne jo 'attrition_system.pkl' notebook se banayi thi, use 'read binary' (rb) mode mein open kar rahe hain
#     with open('attrition_system.pkl', 'rb') as f:
#         return pickle.load(f) # Model aur features ka data return kar rahe hain

# # Error handling ke liye 'try-except' block taaki agar file na mile toh app crash na ho
# try:
#     data_bundle = load_attrition_data()    # Bundle ko load kiya
#     model = data_bundle['model']           # Bundle se trained Random Forest model nikala
#     feature_names = data_bundle['feature_names'] # Bundle se columns ke exact names nikale
# except FileNotFoundError:
#     st.error("Error: 'attrition_system.pkl' nahi mili. Pehle Notebook run karein.")
#     st.stop() # Agar file nahi hai toh app yahi ruk jayegi

# # --- UI HEADER ---
# st.title("🏢 Employee Attrition Intelligence System") # Main heading
# st.subheader("Final Year Project - HR Decision Support") # Sub-heading
# st.markdown("---") # Ek horizontal line partition ke liye

# # --- SIDEBAR INPUTS ---
# st.sidebar.header("📝 Employee Profile Data") # Sidebar ki heading

# def get_user_inputs():
#     inputs = {} # Ek khali dictionary banayi user ka data store karne ke liye
    
#     # Sidebar mein sliders aur dropdowns bana rahe hain
#     # Syntax: slider(label, min_value, max_value, default_value)
#     inputs['Age'] = st.sidebar.slider('Age', 18, 60, 35)
#     inputs['MonthlyIncome'] = st.sidebar.slider('Monthly Income ($)', 1000, 20000, 8000)
    
#     # Dropdown menu 'Yes/No' ke liye
#     overtime_choice = st.sidebar.selectbox('Overtime', ['No', 'Yes'])
#     # Model numbers samajhta hai, isliye 'Yes' ko 1 aur 'No' ko 0 mein convert kiya
#     inputs['OverTime'] = 1 if overtime_choice == 'Yes' else 0
    
#     inputs['JobSatisfaction'] = st.sidebar.slider('Job Satisfaction (1-4)', 1, 4, 3)
#     inputs['StockOptionLevel'] = st.sidebar.slider('Stock Option Level (0-3)', 0, 3, 1)
#     inputs['YearsAtCompany'] = st.sidebar.slider('Years At Company', 0, 40, 5)
#     inputs['TotalWorkingYears'] = st.sidebar.slider('Total Working Years', 0, 40, 10)
#     inputs['WorkLifeBalance'] = st.sidebar.slider('Work Life Balance (1-4)', 1, 4, 3)
    
#     # FEATURE ALIGNMENT LOGIC:
#     # Model ne training ke waqt 30 columns dekhe the, lekin hum app mein sirf 8-9 mang rahe hain.
#     # Is loop se hum baaki bache huye 22 columns ko '0' set kar rahe hain taaki model ko pura 30 ka set mile.
#     full_input_dict = {}
#     for name in feature_names: # Jo names notebook se aaye hain unpe loop chalaya
#         if name in inputs:
#             full_input_dict[name] = inputs[name] # Agar slider hai toh slider ki value daalo
#         else:
#             full_input_dict[name] = 0 # Warna us column mein 0 daal do
            
#     # Dictionary ko DataFrame mein convert kiya aur columns ka order wahi rakha jo model chahta hai
#     final_df = pd.DataFrame([full_input_dict])[feature_names]
#     return final_df, overtime_choice, inputs['MonthlyIncome']

# # Function call karke data return liya
# input_df, ot_label, income_val = get_user_inputs()

# # --- PREDICTION ENGINE ---
# # Jab user is button par click karega:
# if st.button('🔍 Run Attrition Diagnostic'):
    
#     # 'predict_proba' se humein 'Jaane' (1) aur 'Rukne' (0) ki percentage milti hai
#     # [0][1] ka matlab hai humein 'Attrition = Yes' ki probability chahiye
#     probability = model.predict_proba(input_df)[0][1] * 100
    
#     st.divider() # Design ke liye line
    
#     # Screen ko do columns mein divide kiya results dikhane ke liye
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         # Probability ke basis par color coding aur status dikhana
#         if probability > 60:
#             st.error(f"### 🚨 High Risk Detected: {probability:.1f}%")
#             st.write("**Assessment:** High probability of resignation.")
#         elif probability > 35:
#             st.warning(f"### ⚠️ Medium Risk Detected: {probability:.1f}%")
#             st.write("**Assessment:** Moderate risk; employee needs engagement.")
#         else:
#             st.success(f"### ✅ Low Risk / Stable: {probability:.1f}%")
#             st.write("**Assessment:** Strong alignment with retention factors.")
#             st.balloons() # Khushi manane ke liye balloons animation

#     with col2:
#         # HR ke liye actionable suggestions
#         st.subheader("💡 Strategic Insights")
#         if ot_label == 'Yes':
#             st.info("• **Workload:** Overtime kam karne se risk score gir sakta hai.")
#         if income_val < 5000:
#             st.info("• **Compensation:** Salary industry standard se kam lag rahi hai.")
#         if probability < 35:
#             st.write("• **Retention Logic:** Ye employee long-term asset ho sakta hai.")

# st.markdown("---")
# # Footer credit
# st.caption("AI Model: Random Forest Classifier | Domain: Strategic Human Resources")