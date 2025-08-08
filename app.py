# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Student Digital Wellness",
    layout="wide",
    initial_sidebar_state="expanded"
) 

# --- CACHED FUNCTIONS FOR PERFORMANCE ---
@st.cache_data
def load_and_preprocess_data():
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv('Students Social Media Addiction.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'student_data.csv' not found. Please upload the data file.")
        return None

@st.cache_resource
def train_model(df_train):
    """Trains the predictive model using the full dataset."""
    features = ['Avg_Daily_Usage_Hours', 'Addicted_Score', 'Sleep_Hours_Per_Night', 
                'Gender', 'Academic_Level', 'Most_Used_Platform', 
                'Country', 'Relationship_Status', 'Conflicts_Over_Social_Media']
    target = 'Mental_Health_Score'

    X = df_train[features]
    y = df_train[target]

    categorical_features = ['Gender', 'Academic_Level', 'Most_Used_Platform', 'Country', 'Relationship_Status']
    numerical_features = ['Avg_Daily_Usage_Hours', 'Addicted_Score', 'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
    
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', LinearRegression())])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)
    
    y_pred = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model_pipeline, mse, r2

def generate_recommendations(prediction, new_student_data):
    """Generates dynamic and specific recommendations based on the forecast."""
    recommendations = []
    
    if prediction < 5:
        recommendations.append(f"Mental Health Alert: The model predicts a low score of {prediction:.2f}. This student may be at risk. Immediate intervention and counseling are highly recommended.")
    elif prediction >= 5 and prediction < 8:
        recommendations.append(f"Wellness Check: The model predicts a moderate score of {prediction:.2f}. Encourage a balanced approach to social media usage.")
    else:
        recommendations.append(f"Positive Outlook: The model predicts a healthy mental health score of {prediction:.2f}. This student appears to be managing their digital life well.")

    if new_student_data['Avg_Daily_Usage_Hours'].iloc[0] > 6:
        recommendations.append(f"High Usage Warning: With {new_student_data['Avg_Daily_Usage_Hours'].iloc[0]} hours of daily usage, suggest setting app timers or usage goals.")
        
    if new_student_data['Addicted_Score'].iloc[0] > 7:
        recommendations.append(f"Addiction Risk: An Addicted Score of {new_student_data['Addicted_Score'].iloc[0]} is a red flag. Recommend a digital detox challenge or a consultation with a wellness coach.")
        
    if new_student_data['Sleep_Hours_Per_Night'].iloc[0] < 6:
        recommendations.append(f"Sleep Deficit: The student reports only {new_student_data['Sleep_Hours_Per_Night'].iloc[0]} hours of sleep. Advise on a consistent sleep schedule and avoiding screens an hour before bed.")
    
    if new_student_data['Most_Used_Platform'].iloc[0] == 'TikTok':
        recommendations.append("Platform Insight: The primary platform is TikTok. Help them find resources on conscious consumption and setting boundaries with fast-paced media.")

    if new_student_data['Conflicts_Over_Social_Media'].iloc[0] > 3:
        recommendations.append(f"Relationship Strain: The student reports a high number of conflicts due to social media. Offer resources on healthy digital communication and boundary-setting.")

    return recommendations

# --- MAIN APP LAYOUT ---
st.title("Student Digital Wellness Project")
st.markdown("### An analytical dashboard to explore and predict student well-being.")

# Load data and train model
df = load_and_preprocess_data()

# Only proceed if data was loaded successfully
if df is not None:
    model, mse, r2 = train_model(df)


    # --- EDA Section ---
    st.header("1. Exploratory Data Analysis (EDA)")
    st.write("This section visualizes key trends and relationships in the dataset.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Social Media Usage Distribution")
        fig_hist = px.histogram(df, x='Avg_Daily_Usage_Hours', nbins=20, color_discrete_sequence=['#4B0082'])
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Usage vs. Mental Health")
        fig_scatter = px.scatter(df, x='Avg_Daily_Usage_Hours', y='Mental_Health_Score', color='Affects_Academic_Performance', symbol='Gender',
                                 labels={'Avg_Daily_Usage_Hours': 'Avg. Usage Hours', 'Mental_Health_Score': 'Mental Health Score'})
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Correlation Matrix of Key Variables")
    corr = df[['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Addicted_Score', 'Conflicts_Over_Social_Media']].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', range_color=[-1,1])
    st.plotly_chart(fig_corr, use_container_width=True)


    # --- EDA Summary and Recommendations ---
    st.header("Insights and Key Findings from EDA")
    st.markdown("Based on the data, here is a summary of the key trends:")
    
    high_usage_count = len(df[df['Avg_Daily_Usage_Hours'] > df['Avg_Daily_Usage_Hours'].mean() + df['Avg_Daily_Usage_Hours'].std()])
    st.info(f"A significant number of students ({high_usage_count} students) report high social media usage, exceeding the average. This suggests a need for targeted workshops on digital time management.")

    low_mental_health_count = len(df[df['Mental_Health_Score'] < 5])
    if low_mental_health_count > len(df) * 0.2:
        st.warning(f"Over 20% of students ({low_mental_health_count} students) have a mental health score below 5. This suggests a potential mental health crisis, and it's crucial to investigate the underlying factors with more detail.")
    else:
        st.info("Mental health scores are generally well-distributed, but a small group may need targeted support. Focus on resources for students scoring below 5.")
    
    conflicts_in_relationship = df[df['Relationship_Status'] == 'In Relationship']['Conflicts_Over_Social_Media'].mean()
    st.info(f"On average, students in a relationship report {conflicts_in_relationship:.1f} conflicts due to social media. Providing resources on healthy digital communication could be beneficial for this group.")

    # --- Model Evaluation Section ---
    st.header("2. Predictive Model Evaluation")
    st.info(f"The predictive model uses a Linear Regression approach. The model's Mean Squared Error is **{mse:.2f}** and the R-squared score is **{r2:.2f}**.")


    # --- Forecasting Section ---
    st.header("3. Student Digital Wellness Forecasting")
    st.write("Enter a new student's data below to predict their Mental Health Score and receive personalized recommendations.")

    # --- Sidebar for User Input ---
    st.sidebar.header("Input New Student Data")

    avg_usage = st.sidebar.slider("Avg. Daily Usage Hours", 0.0, 10.0, 5.0, 0.1)
    addicted_score = st.sidebar.slider("Addiction Score (1-10)", 1, 10, 5, 1)
    sleep_hours = st.sidebar.slider("Sleep Hours Per Night", 3.0, 12.0, 7.0, 0.1)
    
    # Dynamically populate select boxes from the loaded data
    gender = st.sidebar.selectbox("Gender", df['Gender'].unique())
    academic_level = st.sidebar.selectbox("Academic Level", df['Academic_Level'].unique())
    most_used_platform = st.sidebar.selectbox("Most Used Platform", df['Most_Used_Platform'].unique())
    country = st.sidebar.selectbox("Country", df['Country'].unique())
    relationship_status = st.sidebar.selectbox("Relationship Status", df['Relationship_Status'].unique())
    
    conflicts = st.sidebar.slider("Conflicts Over Social Media", 0, 10, 2, 1)

    input_data = pd.DataFrame({
        'Avg_Daily_Usage_Hours': [avg_usage],
        'Addicted_Score': [addicted_score],
        'Sleep_Hours_Per_Night': [sleep_hours],
        'Gender': [gender],
        'Academic_Level': [academic_level],
        'Most_Used_Platform': [most_used_platform],
        'Country': [country],
        'Relationship_Status': [relationship_status],
        'Conflicts_Over_Social_Media': [conflicts]
    })

    if st.sidebar.button("Predict Score & Get Recommendations"):
        prediction = model.predict(input_data)[0]
        st.subheader("Prediction & Recommendations")
        
        # Color-coding logic for the prediction
        if prediction < 5:
            st.markdown(f'### <span style="color:red; font-weight:bold;">Predicted Mental Health Score: {prediction:.2f}</span>', unsafe_allow_html=True)
        elif prediction >= 8:
            st.markdown(f'### <span style="color:green; font-weight:bold;">Predicted Mental Health Score: {prediction:.2f}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'### Predicted Mental Health Score: {prediction:.2f}', unsafe_allow_html=True)

        recommendations = generate_recommendations(prediction, input_data)
        
        st.markdown("---")
        st.markdown("### Personalized Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"**{i}.** {rec}")
    
    # --- FOOTER ---
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #808080;">
            <p>Developed By <b>Andiswa Mabuza</b> | 
            Email: <a href="mailto:Amabuza53@gmail.com">Amabuza53@gmail.com</a> | 
            Visit: <a href="https://andiswamabuza.vercel.app" target="_blank">Developer Site</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )
