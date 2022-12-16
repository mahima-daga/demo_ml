import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
st.title('Medical Diagnostic Web App')

# Step 1: Load model
model=open('rfc.pickle','rb')
rfc_model=pickle.load(model)
model.close()

# Step 2: Create UI for front end user
pregs=st.slider('Pregnancies',0,20,step=1)
glucose=st.slider('Glucose',40,200,40) 
bp=st.slider('BloodPressure',24,240,24)
skin=st.slider('SkinThickness',5,100,5) 
insulin=st.slider('Insulin',14,900,14)
bmi=st.slider('BMI',15,70,15) 
dpf=st.slider('DiabetesPedigreeFunction',0.05,2.5,0.05) 
age=st.slider('Age',21,90,21)

# step 3: change user input to model input data

data={'Pregnancies':pregs,
     'Glucose':glucose,
     'BloodPressure':bp,
     'SkinThickness':skin,
     'Insulin':insulin,
     'BMI':bmi,
     'DiabetesPedigreeFunction':dpf,
     'Age':age}

input_data=pd.DataFrame([data])

# STEP4 : get predictions and print the result
predictions=rfc_model.predict(input_data)[0]
st.write(predictions)
if st.button('Predict'):
    if predictions==0:
        st.success('diabetes Free')
    if predictions==1:
        st.error('Has Diabetes')
