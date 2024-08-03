import streamlit as st
from utils import PrepProcesor, columns

import numpy as np
import pandas as pd
import joblib

# Load the model
model = joblib.load('xgbpipe1.joblib')

# Set the title and image
st.title('Will you survive ??? :ship:')
st.image('https://services.meteored.com/img/article/titanic-10-curiosidades-sobre-el-naufragio-mas-famoso-de-la-historia-1681429632845_1024.jpg')
st.markdown('The Titanic struck an iceberg on 15 April 1912 that caused its sinking, leading to the death of more than 1,500 people. This made it one of the deadliestsinking of a single ship.')
st.markdown('We will try to predict whether a particular person on Titanic survived or not using 11 features about them.')
st.markdown('The Data was trained by XGBOOST algorithm with 85% accuracy.')
# Get user inputs
passengerid = st.text_input("Input Passenger ID", '8585') 
pclass = st.selectbox("Choose class", [1, 2, 3])
name = st.text_input("Input Passenger Name", 'Geonik Arakelyan')
sex = st.selectbox("Choose sex", ['male', 'female'])
age = st.slider("Choose age", 0, 100)
sibsp = st.slider("How many siblings or spouses are traveling with you?", 0, 10)
parch = st.slider("How many parents or children are traveling with you", 0, 10)
ticket = st.text_input("Input Ticket Number", "8585") 
fare = st.number_input("Input Fare Price (in 1912)", 0, 1000)
cabin = st.text_input("Input Cabin", "C52") 
embarked = st.radio("Boarded Location", ['Southampton', 'Cherbourg', 'Queenstown'])

# Initialize session state for prediction result
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

def predict():
    row = np.array([passengerid, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked])
    X = pd.DataFrame([row], columns=columns)
    prediction = model.predict(X)
    
    # Store the prediction in session state
    st.session_state.prediction = prediction[0]

# Button to trigger prediction
trigger = st.button('Predict', on_click=predict)

# Placeholder for the result
result_placeholder = st.empty()

# Display the prediction result
if st.session_state.prediction is not None:
    with result_placeholder.container():
        if st.session_state.prediction == 1:
            st.success('Passenger Survived :thumbsup:')
            st.markdown("![Alt Text](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExc3g1azE1Y3BnYmpiaDRnN2x3c2N0bm11cDRvaHplN2lzZWR6MWFqdiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/vosPjrvZwobXsQj3l4/giphy.gif)")
        else:
            st.error('Passenger did not Survive :thumbsdown:')
            st.markdown("![Alt Text](https://c.tenor.com/bl_3kYhSzsgAAAAC/tenor.gif)")
