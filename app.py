import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Premium Styling
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
    }
    
    /* Main Headers */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 16px;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Glassmorphism Cards */
    .css-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    /* Input Section Containers */
    .stSelectbox, .stSlider, .stNumberInput {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Styled Submit Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #2563EB 0%, #7C3AED 100%);
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 18px;
        padding: 14px 20px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Metrics Override */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

# Plotly Theme Defaults
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = px.colors.sequential.Purples

# ---------------------------------------------------------
# Load Resources
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_student_model.pkl")
    except Exception as e:
        return None

@st.cache_data
def load_data():
    try:
        return pd.read_csv("student_data.csv")
    except Exception as e:
        return None

model = load_model()
df = load_data()

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/illustrations/200/graduation-cap.png", width=120)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to page:",
    ["Student Prediction", "Data Analytics & Insights", "Model Performance"]
)

# ---------------------------------------------------------
# Page 1: Student Prediction
# ---------------------------------------------------------
if menu == "Student Prediction":
    st.markdown("<h1 class='main-title'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Predict student academic performance category based on behavioral and demographic factors.</p>", unsafe_allow_html=True)
    
    if model is None:
        st.error("Error: Could not load `best_student_model.pkl`. Please ensure the file is in the same directory.")
    else:
        st.subheader("📋 Enter Student Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sex = st.selectbox("Gender", ["F", "M"], help="F: Female, M: Male")
            age = st.slider("Age", 15, 22, 17)
            address = st.selectbox("Address Type", ["U", "R"], help="U: Urban, R: Rural")
            famsize = st.selectbox("Family Size", ["LE3", "GT3"], help="LE3: Less/Equal 3, GT3: Greater than 3")
            pstatus = st.selectbox("Parent Cohabitation Status", ["T", "A"], help="T: Living together, A: Apart")

        with col2:
            medu = st.select_slider("Mother Education Level", options=[0, 1, 2, 3, 4], value=2)
            fedu = st.select_slider("Father Education Level", options=[0, 1, 2, 3, 4], value=2)
            mjob = st.selectbox("Mother Job", ["teacher", "health", "services", "at_home", "other"])
            fjob = st.selectbox("Father Job", ["teacher", "health", "services", "at_home", "other"])
            reason = st.selectbox("Reason to Choose School", ["home", "reputation", "course", "other"])

        with col3:
            guardian = st.selectbox("Guardian", ["mother", "father", "other"])
            traveltime = st.select_slider("Travel Time Level", options=[1, 2, 3, 4], value=1)
            studytime = st.select_slider("Weekly Study Time Level", options=[1, 2, 3, 4], value=2)
            failures = st.select_slider("Past Class Failures", options=[0, 1, 2, 3], value=0)
            schoolsup = st.selectbox("Extra Educational Support", ["yes", "no"])

        st.markdown("---")
        col4, col5, col6 = st.columns(3)

        with col4:
            famsup = st.selectbox("Family Educational Support", ["yes", "no"])
            paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
            activities = st.selectbox("Extra-curricular Activities", ["yes", "no"])
            nursery = st.selectbox("Attended Nursery", ["yes", "no"])

        with col5:
            higher = st.selectbox("Wants Higher Education", ["yes", "no"])
            internet = st.selectbox("Internet Access at Home", ["yes", "no"])
            romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"])
            famrel = st.slider("Family Relationship Quality", 1, 5, 4)

        with col6:
            freetime = st.slider("Free Time After School", 1, 5, 3)
            goout = st.slider("Going Out with Friends", 1, 5, 3)
            dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)
            walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1)
            health = st.slider("Current Health Status", 1, 5, 4)
            absences = st.number_input("Number of Absences", min_value=0, max_value=93, value=4)
            school = st.selectbox("School", ["GP", "MS"])

        st.markdown("---")
        
        # Assemble input dictionary
        input_data = {
            "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
            "Pstatus": pstatus, "Medu": medu, "Fedu": fedu, "Mjob": mjob, "Fjob": fjob,
            "reason": reason, "guardian": guardian, "traveltime": traveltime, "studytime": studytime,
            "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
            "activities": activities, "nursery": nursery, "higher": higher, "internet": internet,
            "romantic": romantic, "famrel": famrel, "freetime": freetime, "goout": goout,
            "Dalc": dalc, "Walc": walc, "health": health, "absences": absences
        }

        input_df = pd.DataFrame([input_data])

        if st.button("🚀 Predict Student Performance"):
            prediction = model.predict(input_df)[0]
            
            st.markdown("### 🎯 Prediction Results")
            
            if prediction == "High":
                st.balloons()
                st.success(f"🌟 **Predicted Category:** **{prediction} Performance**")
                st.info("This student demonstrates high potential and favorable study conditions.")
            elif prediction == "Medium":
                st.warning(f"📊 **Predicted Category:** **{prediction} Performance**")
                st.info("This student shows moderate performance with room for academic support.")
            else:
                st.error(f"⚠️ **Predicted Category:** **{prediction} Performance**")
                st.info("This student may require targeted intervention and extra support.")

# ---------------------------------------------------------
# Page 2: Data Analytics & Insights
# ---------------------------------------------------------
elif menu == "Data Analytics & Insights":
    st.markdown("<h1 class='main-title'>📊 Exploratory Data Analytics</h1>", unsafe_allow_html=True)
    
    if df is None:
        st.error("Error: Could not load `student_data.csv`.")
    else:
        st.subheader("Key Dataset Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Students", len(df))
        m2.metric("Average Absences", f"{df['absences'].mean():.1f}")
        m3.metric("Avg G3 Score", f"{df['G3'].mean():.1f}")
        m4.metric("Higher Ed Desire %", f"{(df['higher'] == 'yes').mean() * 100:.1f}%")

        st.markdown("---")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Study Time vs Final Grade (G3)")
            fig1 = px.box(
                df, x="studytime", y="G3", color="studytime",
                labels={"studytime": "Study Time Level", "G3": "Final Score"},
                title="Final Grade Distribution across Study Time Levels",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.subheader("Absences vs Final Grade (G3)")
            fig2 = px.scatter(
                df, x="absences", y="G3", color="sex",
                title="Impact of Absences on Final Grade",
                color_discrete_sequence=["#3B82F6", "#EC4899"]
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# Page 3: Model Performance
# ---------------------------------------------------------
elif menu == "Model Performance":
    st.markdown("<h1 class='main-title'>⚡ Model Evaluation Metrics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Comparison across trained machine learning models</p>", unsafe_allow_html=True)

    # Static metric comparison data
    metrics_data = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression", "SVM", "Decision Tree", "KNN"],
        "Accuracy": [0.72, 0.68, 0.67, 0.61, 0.58],
        "F1 Score": [0.71, 0.67, 0.66, 0.60, 0.57]
    })

    fig = px.bar(
        metrics_data, x="F1 Score", y="Model", orientation="h",
        color="F1 Score", color_continuous_scale="Plasma",
        title="F1 Score Comparison"
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(metrics_data.style.highlight_max(axis=0, color="#1E3A8A"), use_container_width=True)