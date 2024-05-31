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

if 'sex' not in st.session_state:
    st.session_state['sex'] = 'Man'
if 'weight' not in st.session_state:
    st.session_state['weight'] = 67.6
if 'height' not in st.session_state:
    st.session_state['height'] = 1.85
if 'age' not in st.session_state:
    st.session_state['age'] = 30.0
st.session_state['Ginit'] = (1.0 + 2.7)*0.5
st.session_state['ECFinit'] = 0.7*0.235*st.session_state['weight']  
if 'Finit' not in st.session_state:
    if st.session_state['sex']== 'Woman':
        st.session_state['Finit'] = (st.session_state['weight']/100.0)*(0.14*st.session_state['age'] + 39.96*math.log(st.session_state['weight']/((st.session_state['height'])**2.0)) - 102.01)
    elif st.session_state['sex']== 'Man': 
        st.session_state['Finit'] = (st.session_state['weight']/100.0)*(0.14*st.session_state['age'] + 37.31*math.log(st.session_state['weight']/((st.session_state['height'])**2.0)) - 103.95) 
if 'Linit' not in st.session_state:
    st.session_state['Linit'] = st.session_state['weight'] - (st.session_state['Finit'] + (1.0 + 2.7)*st.session_state['Ginit'] + st.session_state['ECFinit'])
           

sex = st.selectbox("Sex:", ["Man", "Woman"], ["Man", "Woman"].index(st.session_state['sex']), key="sex")
weight = st.number_input("Weight (kg):", 0.0, 1000.0, st.session_state['weight'], key="weight") # max, min 
age = st.number_input("Age (years):", 0.0, 100.0, st.session_state['age'], key="age") # max, min 
height = st.number_input("Height (m):", 0.0, 2.5, st.session_state['height'],  key="height") # st.session_state['height'], 0.1, 
fat_known = st.checkbox("Do you know your fat mass?")
if fat_known:
    Finit = st.number_input("Fat mass (kg):", 0.0, 1000.0, st.session_state.Finit, 0.1, key="Finit")

lean_known = st.checkbox("Do you know your lean mass?")
if lean_known:
   Linit = st.number_input("Lean mass (kg):", 0.0, 1000.0, st.session_state.Linit, 0.1, key="Linit")
