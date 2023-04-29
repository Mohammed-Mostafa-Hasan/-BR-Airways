import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

st.write("""
# Booking Prediction App

This app predicts the **Booking status for a customer**!

Data obtained from  [project repo](https://github.com/Mohammed-Mostafa-Hasan/BR-Airways/blob/main/Data%20preparation%20for%20Model/data/customer_booking.csv)

""")
Booking_df = pd.read_csv('customer_booking.csv',encoding = 'ISO-8859-1')




st.sidebar.header('User Input Features')
# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
#read raw data
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():

        Booking_origin = st.sidebar.selectbox("Select a country", pd.unique(Booking_df['booking_origin']))

        rout_path = st.sidebar.selectbox('Rout',pd.unique(Booking_df['route']))

        sales_channel = st.sidebar.selectbox('sales_channel',('Internet','Mobile'))

        trip_type = st.sidebar.selectbox('trip_type',('RoundTrip','CircleTrip','OneWay'))

        flight_day = st.sidebar.selectbox('flight_day',('Sat', 'Wed', 'Thu', 'Mon', 'Sun', 'Tue', 'Fri'))
       
        purchase_lead  = st.sidebar.number_input('purchase_lead', 0,867,80)

        length_of_stay = st.sidebar.number_input('length_of_stay', 0,778,23)

        flight_hour = st.sidebar.slider('flight_hour',0,23,7)

        num_passengers = st.sidebar.slider('num_passengers', 1,9,3)

        extra_baggage = st.sidebar.selectbox('wants_extra_baggage', (0,1))

        preferred_seat = st.sidebar.selectbox('wants_preferred_seat', (0,1))

        flight_meals = st.sidebar.selectbox('wants_in_flight_meals', (0,1))

        flight_duration = st.sidebar.slider('flight_duration',4.67,9.5)


     
        data = {'num_passengers': num_passengers,
                'sales_channel': sales_channel,
                'trip_type': trip_type,
                'purchase_lead': purchase_lead,
                'length_of_stay': length_of_stay,
                'flight_hour': flight_hour,
                'flight_day': flight_day,
                'route': rout_path,
                'booking_origin': Booking_origin,
                'wants_extra_baggage': extra_baggage,
                'wants_preferred_seat': preferred_seat,
                'wants_in_flight_meals': flight_meals,
                'flight_duration': flight_duration,
                }
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()
# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
Booking_df = Booking_df.drop(columns=['booking_complete'])
df = pd.concat([input_df,Booking_df],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# encode = ['sales_channel','trip_type','flight_day','route','booking_origin']
# for col in encode:
#     dummy = pd.get_dummies(df[col])
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]
le = LabelEncoder()
df_le = df.apply(le.fit_transform)
df = df_le[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = joblib.load('RF_model.pkl')

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_species = np.array(['Booking_not_complete','Booking_complete'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)