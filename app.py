import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe2.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Pune house rent predictor")


Seller = st.selectbox('seller_type', df['seller_type'].unique())

Bedroom = st.selectbox('bedroom', df['bedroom'].unique())

Layout_type = st.selectbox('layout_type', df['layout_type'].unique())

property_type = st.selectbox('property_type', df['property_type'].unique())

Location = st.selectbox('locality', df['locality'].unique())

Area = st.number_input('Sq-ft')

Furnish_type = st.selectbox('furnish_type', df['furnish_type'].unique())

Bathroom = st.selectbox('bathroom', df['bathroom'].unique())

if st.button('Predict Price'):
    query = np.array([Seller, Bedroom, Layout_type, property_type, Location, Area, Furnish_type, Bathroom])
    query = query.reshape(1, 8)
    st.title("₹" + str(np.round(pipe.predict(query)[0], 2)))







