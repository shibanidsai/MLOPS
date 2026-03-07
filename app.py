import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

carmodel = load('carsmodel.pkl')

def predict_price(model, 
                  fuelType,
                  transmission, 
                  ownerType,
                  kmDriven, 
                  age):

    inputs_dict = {'KM_Driven' : float(kmDriven), 
                   'Fuel_Type': fuelType, 
                   'Age': float(age), 
                   'Transmission': transmission, 
                   'Owner_Type': ownerType, 
                   'Model': model.lower()}

    df = pd.DataFrame(inputs_dict, index = [0])


    price = carmodel.predict(df)[0]
    return price 


#function to define the app_layout
def app_layout():

    st.title('Car Resale Value Prediction')
    st.header('Enter car detail:')  

    ## Creating the user input fields 

    model = st.selectbox('Model:', 
                         ['ertiga', 'swift', 'alto', 'wagon', 
                          'vitara', 'zen', 'ritz',
                          'omni', 'eeco', 'ciaz', 'baleno', 
                          'a-star', 'celerio', 'dzire'])

    transmission = st.radio('Transmission:', 
                            ['Manual', 'Automatic'], 
                            horizontal=True)

    fueltype = st.radio('Fuel Type:', 
                        ['Petrol', 'Diesel'], 
                        horizontal=True)

    ownertype = st.radio('Owner Type:', 
                         ['First', 'Second', 'Third'], 
                         horizontal=True)

    age = st.number_input('Age:', 
                          min_value=1,
                          max_value=10, 
                          value=1)

    kmdriven = st.number_input('KM Driven in 1000 kms.:', 
                               min_value=1.0, 
                               max_value=150.0, 
                               value=10.0)

    if st.button('Predict Price'):
        price = predict_price(model, 
                              fueltype, 
                              transmission, 
                              ownertype, 
                              kmdriven, 
                              age)
        st.success(f'Explected resale value of the car is : INR {np.round(price, 2)} Lakhs')

if __name__=='__main__':
  app_layout()
