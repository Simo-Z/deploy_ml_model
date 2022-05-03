import streamlit as st
import pickle
import pandas as pd

st.image("https://www.ihre-vorsorge.de/fileadmin/_processed_/8/a/csm_adobestock-186462448-hausmodell-muenzstapel-stift-taschenrechner-kaufvertrag-bauplan_89685b45ea.jpg")
 
st.title("Housing Prices Prediction")
 
st.write("""
### Project description
We have trained several models to predict the price of a house based on features such as the area of the house and the condition and quality of their different rooms.
 
""")

LotArea = st.number_input("Lot Area")
TotalBsmtSF = st.number_input("Basement Square Feet")
BedroomAbvGr = st.number_input("Number of Bedrooms")
GarageCars = st.number_input("Car spaces in Garage")

model = pickle.load(open('/Users/simo/Home/WBS-Bootcamp/deploy_ml_model/models/trained_pipe_knn.sav', 'rb'))
 
new_house = pd.DataFrame({
    'LotArea':[LotArea],
    'TotalBsmtSF':[TotalBsmtSF], 
    'BedroomAbvGr':[BedroomAbvGr], 
    'GarageCars':[GarageCars]
})

prediction = model.predict(new_house)
 
st.write("The price of the house is:", prediction)