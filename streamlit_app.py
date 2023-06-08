pip install --upgrade pip
pip install seaborn
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import streamlit as st
import pickle

def main():
    style = """<div style='background-color:pink; padding:12px'>
              <h1 style='color:black'>Rental Price Prediction App</h1>
       </div>"""
    st.markdown(style, unsafe_allow_html=True)
    left, right = st.columns((2,2))

    # x, numeric = rooms, parking, size	
    rooms = left.number_input('Enter the number of rooms',
                              min_value=1, max_value=10, step=1, format="%.1f", value=1)
    parking = right.number_input('Enter the number of parking available',
                                min_value=0, max_value=5, step=1, format="%.1f", value=1)
    size = left.number_input('Enter the size of the whole property in sqft',
                            min_value=1, max_value=10000, step=1, format="%.2f", value=100)

    # x, category = property_type, furnished, region
    property_type = st.selectbox('Select the type of your property',
                    ('Condominium', 'Service Residence', 'Apartment', 'Flat', 'Studio', 'Others', 'Duplex', 'Townhouse Condo'))
  
    furnished = st.selectbox('Select the furnish status of your property',
                    ('Fully Furnished', 'Not Furnished', 'Partially Furnished'))

    region = st.selectbox('Located at KL or Selangor',
                    ('Kuala Lumpur', 'Selangor'))
    # button
    button = st.button('Predict')
    
    # if button is pressed
    if button:
        # make prediction
        result = predict(rooms, parking, size, property_type, furnished, region)
        st.success(f'The estimate rental price is ${result}')
        
# load the train model
with open('rf_model.pkl', 'rb') as rf:
    model = pickle.load(rf)

# load the StandardScaler
with open('scaler.pkl', 'rb') as stds:
    scaler = pickle.load(stds)


def predict(rooms, parking, size, property_type, furnished, region):
    # processing user input
    property_type = 0 if property_type == 'Condominium' else 1 if property_type == 'Service Residence' else 2 if property_type == 'Apartment' else 3 if property_type == 'Flat' else 4 if property_type == 'Studio' else 5 if property_type == 'Others' else 6 if property_type == 'Duplex' else 7
    furnished = 0 if furnished == 'Fully Furnished' else 1 if furnished == 'Partially Furnished' else 2 if furnished == 'Not Furnished' else 3 
    region = 0 if region == 'Kuala Lumpur' else 1 if region == 'Selangor' else 2
    lists = [rooms, parking, size, property_type, furnished, region]
    df = pd.DataFrame(lists).transpose()
    
    # scaling the data
    scaler.transform(df)
    
    # making predictions using the train model
    prediction = model.predict(df)
    result = int(prediction)
    return result
  
if __name__ == '__main__':
    main()
