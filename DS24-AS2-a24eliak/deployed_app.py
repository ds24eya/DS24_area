#Importing necessary libraries and modules
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import joblib

#Loading the model, the encoders for the categorical values and also the scaler from the Jupyter Notebook file
modelX = load_model('neunet.h5')
enc_onh_month = joblib.load('enc_month.pkl')
enc_onh_day = joblib.load('enc_day.pkl')
scl = joblib.load('scalerx.pkl')  

#Defining the columns that are numerical within the dataset
num_columns = ['X', 'Y', 'temp', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI', 'RH']

st.title('Predict the burned area')

#Setting up the input data from the dataset lined up into 4 columns
line_1, line_2, line_3, line_4 = st.columns(4)

#Lining up the columns with the different columns within the dataset into each 'column tab'
with line_1:
    st.subheader('Date / General')
    month = st.text_input('Enter month (jan - dec): ', 'jan')
    day = st.text_input('Enter day (mon - sun): ', 'mon')
    X = st.number_input('Enter X: ', value = 4)

with line_2:
    st.subheader('General / Weather')
    Y = st.number_input('Enter Y: ', value = 5)
    temp = st.number_input('Enter temp: ', value = 10.0)
    wind = st.number_input('Enter wind speed: ', value = 2.0)

with line_3:
    st.subheader('Weather / Index')
    rain = st.number_input('Enter outside rain', value = 1.5)
    FFMC = st.number_input('Enter FFMC index', value = 70.0)
    DMC = st.number_input('Enter DMC index', value = 11.0)

with line_4:
    st.subheader('Index')
    DC = st.number_input('Enter DC index', value = 20.0)
    ISI = st.number_input('Enter ISI index', value = 16.0)
    RH = st.number_input('Enter RH (relative humidity)', value = 40.0)

#Predicting the area by encoding, combining, and generating the prediction
def area_predict():
        #Encoding the categorical values
        encoded_month = enc_onh_month.transform([[month]])
        encoded_day = enc_onh_day.transform([[day]])

        #The rest of the columns that are numerical are getting prepared
        rest_columns = np.array([[X, Y, temp, wind, rain, FFMC, DMC, DC, ISI, RH]])

        #All inputs combined within the dataset and the scaled input of it
        tot_input = np.hstack([encoded_month, encoded_day, rest_columns])

        scl_input = scl.transform(tot_input)

        #Dataframe created in order to agree with the input of the model for the encoded columns,
        #and also the numerical columns to the dataframe
        sum_columns = list(enc_onh_month.get_feature_names_out(['month'])) + \
                      list(enc_onh_day.get_feature_names_out(['day'])) + \
                      num_columns
        
        df_input = pd.DataFrame(scl_input, columns = sum_columns)
        
        #A prediction is being conducted
        prediction = modelX.predict(df_input)

        #Showing off the result of the predicted area
        st.success(f"The predicted burned area is: {prediction[0][0]}")

#Starting the prediction
if st.button("Predict the area"):
    area_predict()
