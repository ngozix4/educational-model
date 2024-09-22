import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import datetime

# Function to load the model
def load_model():
    with open("Student_Performance_Model.sav", "rb") as file:
        model = pickle.load(file)
    return model

# Load the model
model = load_model()
st.write(type(model))
# Define the input feature columns
df_input = ["Grade Level", 'English', 'Afrikaans', 'Mathematics', 'Natural Sciences',
            'Social Sciences', 'EMS', 'Technology', 'Life Orientation', 'Creative Arts',
            'Life Sciences', 'Physical Sciences', 'Accounting', 'Business Studies',
            'Geography', 'History', 'Visual Arts', 'Information Technology',
            'Attendance Rate', 'Absenteeism Rate', 'Resources']

# Elective subjects
elective_subjects = ['Life Sciences', 'Physical Sciences', 'Accounting', 'Business Studies',
                     'Geography', 'History', 'Visual Arts', 'Information Technology']

def main_page():
    # Streamlit app interface (prediction section)
    st.write("Welcome to the Student Performance Prediction Model")
    st.write("This model uses past records of the student and their environment to predict how they will fare in the next exam.")

    # Initialize inputs dictionary
    inputs = {}

    # Input fields for each feature
    for feature in df_input:
        if feature == "Grade Level":
            inputs[feature] = st.selectbox(feature, options=[9, 10, 11, 12])

        elif feature in ['English', 'Afrikaans', 'Mathematics', 'Natural Sciences',
                        'Social Sciences', 'EMS', 'Technology', 'Life Orientation',
                        'Creative Arts']:
            inputs[feature] = st.number_input(feature, min_value=0, max_value=100)

        elif feature == 'Attendance Rate':
            inputs[feature] = st.slider('Attendance Rate', min_value=0, max_value=100)
            inputs['Absenteeism Rate'] = 100 - inputs['Attendance Rate']
            st.write(f"Absenteeism Rate: {inputs['Absenteeism Rate']}")

        elif feature == 'Resources':
            inputs[feature] = st.selectbox(feature, options=["Good", "Limited"])

    # Ask if the student is taking any elective modules
    taking_electives = st.radio("Are you taking any elective modules?", options=["Yes", "No"])

    # If the student is taking electives, show the elective input fields
    if taking_electives == "Yes":
        for elective in elective_subjects:
            inputs[elective] = st.number_input(elective, min_value=0, max_value=100)
    else:
        # Set elective subjects to 0 if the student is not taking any
        for elective in elective_subjects:
            inputs[elective] = 0

    # Convert inputs to a DataFrame for prediction
    input_data = pd.DataFrame([inputs])

    # Preprocess input data (if required, based on how the model was trained)
    input_data_encoded = pd.get_dummies(input_data)
    
    # Ensure that input_data_encoded has the same columns as df_input and fill missing columns with 0
    missing_cols = set(df_input) - set(input_data_encoded.columns)
    for col in missing_cols:
        input_data_encoded[col] = 0
    
    # Reorder columns to match the model's training data
    input_data_encoded = input_data_encoded[df_input]

    # Make prediction
    def make_prediction():
        if st.button('Predict'):
            prediction = model.predict(input_data_encoded)
            st.markdown(f"**Predicted Performance:** {prediction[0]}")

            # For recommendation, we assume you filter subjects with scores below 65
            subjects = ['English', 'Afrikaans', 'Mathematics', 'Natural Sciences',
                        'Social Sciences', 'EMS', 'Technology', 'Life Orientation',
                        'Creative Arts', 'Life Sciences', 'Physical Sciences',
                        'Accounting', 'Business Studies', 'Geography', 'History',
                        'Visual Arts', 'Information Technology']

            recommendations = []
            for subject in subjects:
                score = inputs.get(subject, 0)
                if 0 < score < 65:
                    if score < 40:
                        recommendations.append({
                            "Subject": subject,
                            "Score": score,
                            "Recommendation": f"Attend extra classes and use online resources to improve in {subject}. Consider using apps like Khan Academy for personalized exercises."
                        })
                    elif 40 <= score < 50:
                        recommendations.append({
                            "Subject": subject,
                            "Score": score,
                            "Recommendation": f"Seek help from a tutor for {subject}. Also, dedicate extra study time with specific focus on problem areas."
                        })
                    elif 50 <= score < 55:
                        recommendations.append({
                            "Subject": subject,
                            "Score": score,
                            "Recommendation": f"Review the key concepts and practice past papers for {subject}. Watch educational videos on YouTube for better understanding."
                        })
                    elif 55 <= score < 60:
                        recommendations.append({
                            "Subject": subject,
                            "Score": score,
                            "Recommendation": f"Engage in regular self-assessment quizzes to identify and work on weak areas in {subject}."
                        })
                    elif 60 <= score < 65:
                        recommendations.append({
                            "Subject": subject,
                            "Score": score,
                            "Recommendation": f"Participate in study groups and focus on weak areas in {subject}. Utilize online forums for discussions and ask questions."
                        })

            # Add motivational messages and time management tips
            if prediction[0] < 50:
                st.write("**Motivational Message:** Don't give up! Remember that every step you take towards improving your performance is a step closer to your goals. You have the potential to excel!")
                recommendations.append({
                    "Subject": "Time Management",
                    "Score": "N/A",
                    "Recommendation": "Create a study schedule that breaks down your tasks into manageable chunks. Use tools like Pomodoro timers to keep yourself on track."
                })
            elif prediction[0] < 65:
                st.write("**Motivational Message:** You're making progress! Keep pushing forward, and don't hesitate to ask for help when needed. Your hard work will pay off.")
                recommendations.append({
                    "Subject": "Time Management",
                    "Score": "N/A",
                    "Recommendation": "Prioritize your tasks and focus on the most challenging subjects first. Balance study with breaks to maintain productivity."
                })
            else:
                st.write("**Motivational Message:** Great job so far! Stay consistent and continue to refine your study habits. Consistency is key to achieving your goals.")
                recommendations.append({
                    "Subject": "Time Management",
                    "Score": "N/A",
                    "Recommendation": "Maintain a steady study routine and allocate time for review sessions to reinforce your knowledge."
                })

            if recommendations:
                st.session_state.df_filtered = pd.DataFrame(recommendations)
            else:
                st.session_state.df_filtered = pd.DataFrame([], columns=["Subject", "Score", "Recommendation"])

            # Generate a personalized study timetable
            st.write("### Suggested Study Timetable")
            timetable = generate_timetable(subjects, inputs)
            st.write(timetable)
            st.download_button(label="Download Timetable", data=timetable.to_csv(), file_name="study_timetable.csv", mime="text/csv")

    make_prediction()

def generate_timetable(subjects, inputs):
    # Define a basic timetable structure (6 days a week, 4 sessions per day)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    sessions = ["Morning", "Afternoon", "Evening"]
    
    # Initialize timetable dictionary
    timetable = {day: {session: "" for session in sessions} for day in days}
    
    # Allocate time based on the subject scores
    for day in days:
        for session in sessions:
            for subject in subjects:
                if inputs.get(subject, 0) < 65:  # Focus on subjects where the score is below 65
                    timetable[day][session] = subject
                    subjects.remove(subject)
                    break
            else:
                timetable[day][session] = "Revision / Break"

    # Convert the timetable to a DataFrame for display
    timetable_df = pd.DataFrame(timetable)
    return timetable_df

def recommendations_page():
    # Check if the prediction has been made
    if 'df_filtered' not in st.session_state or st.session_state.df_filtered.empty:
        st.error("You haven't made a prediction yet or no subjects are below 65. Please go to the Prediction page to predict first.")
        return

    # Streamlit app (recommendations section)
    st.title('Subjects with Scores Below 65')

    # Display pie chart
    fig = px.bar(st.session_state.df_filtered, x='Subject', y='Score', title='Subjects with Scores Below 65')
    st.plotly_chart(fig)

    # Display recommendations
    st.subheader('Recommendations for Improvement')
    for index, row in st.session_state.df_filtered.iterrows():
        st.write(f"**{row['Subject']}**: {row['Recommendation']}")

# Navigation using Streamlit's sidebar
selected_page = st.sidebar.selectbox("Select Page", ["Prediction", "Recommendations"])

if selected_page == "Prediction":
    main_page()
elif selected_page == "Recommendations":
    recommendations_page()
