import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import math 

# Start the app

st.title("Anthropometrics")
st.markdown("""
Here you can specify the anthropometrics of the person you want to make simulations for.
""")
anthropometrics["sex"] = st.selectbox("Sex:", ["Man", "Woman"], ["Man", "Woman"].index('Man'), key="sex")
anthropometrics["weight"] = st.number_input("Weight (kg):", 0.0, 1000.0, 67.6, key="weight") # max, min 
anthropometrics["age"] = st.number_input("Age (years):", 0.0, 100.0, 30.0, key="age") # max, min 
anthropometrics["height"] = st.number_input("Height (m):", 0.0, 2.5, 1.85,  key="height") # st.session_state['height'], 0.1, 


if 'sex' not in st.session_state:
    st.session_state['sex'] = anthropometrics["sex"] # 'Man'
if 'weight' not in st.session_state:
    st.session_state['weight'] = anthropometrics["weight"] # 67.6
if 'height' not in st.session_state:
    st.session_state['height'] = anthropometrics["height"]
if 'age' not in st.session_state:
    st.session_state['age'] = anthropometrics["age"] # 30.0
st.session_state['Ginit'] = (1.0 + 2.7)*0.5
st.session_state['ECFinit'] = 0.7*0.235*st.session_state['weight']  
if 'Finit' not in st.session_state:
    if st.session_state['sex']== 'Woman':
        st.session_state['Finit'] = (st.session_state['weight']/100.0)*(0.14*st.session_state['age'] + 39.96*math.log(st.session_state['weight']/((st.session_state['height'])**2.0)) - 102.01)
    elif st.session_state['sex']== 'Man': 
        st.session_state['Finit'] = (st.session_state['weight']/100.0)*(0.14*st.session_state['age'] + 37.31*math.log(st.session_state['weight']/((st.session_state['height'])**2.0)) - 103.95) 
if 'Linit' not in st.session_state:
    st.session_state['Linit'] = st.session_state['weight'] - (st.session_state['Finit'] + (1.0 + 2.7)*st.session_state['Ginit'] + st.session_state['ECFinit'])

#anthropometrics = {"weight": st.session_state['weight'], "height": st.session_state['height'], 
#                   "Finit": st.session_state['Finit'], "Linit": st.session_state['Linit'],
#                   "Ginit": st.session_state['Ginit'], "ECFinit": st.session_state['ECFinit']
#                   }
# "ECFinit": st.session_state['ECFinit'], "Ginit": st.session_state['Ginit'], 
#                   "sex": st.session_state['sex'], "age": st.session_state['age'],
                   
# anthropometrics["sex"] = float(anthropometrics["sex"].lower() in ["male", "man", "men", "boy", "1", "chap", "guy"]) #Converts to a numerical representation

fat_known = st.checkbox("Do you know your fat mass?")
if fat_known:
    st.session_state["Finit"] = st.number_input("Fat mass (kg):", 0.0, 1000.0, st.session_state.Finit, 0.1, key="Finit")

lean_known = st.checkbox("Do you know your lean mass?")

if lean_known:
   st.session_state["Linit"] = st.number_input("Lean mass (kg):", 0.0, 1000.0, st.session_state.Linit, 0.1, key="Linit")
