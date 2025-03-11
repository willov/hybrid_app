
import streamlit as st

# Install sund in a custom location
import subprocess
import sys
import os 

os.makedirs('./custom_package', exist_ok=True)

if "sund" not in os.listdir('./custom_package'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=./custom_package", 'https://isbgroup.eu/sund-toolbox/releases/sund-1.2.24.tar.gz'])

# Runthe app

st.title("Digital twins and hybrid modelling for simulation of physiological variables and stroke risk")
st.markdown("""This app can be used to simulate the mechanistic part of our hybrid physiological digital twin, that combines mechanistic models with a machine learning model. 
The mechanistic part can simulate the evolution of blood pressure as well as the development of type 2 diabetes and related risk factors (such as weight, fasting plasma glucose) through time, under different intervention scenarios, involving a change in diet, exercise, and certain medications. These forecast trajectories of the physiological risk factors are then used by the machine learning model to calculate the 5-year risk of stroke, which thus also can be calculated for each timepoint in the simulated scenarios.

The twin can be used to simulate both long-term scenarios - weight change or blood pressure change with and without medication, - and short-term scenarios in the form of a meal.           
                        
We hope that our hybrid digital twin can help improve patients’ understanding of their body and health, and therefore serve as a valuable tool for patient education.

Please note that this application is only for research and visualization purposes, and should not be used to make medical decisions.  

This application is a companion-application to the publication titled \"Digital twins and hybrid modelling for simulation of physiological variables and stroke risk\", [available as a preprint on bioRxiv](https://www.biorxiv.org/content/10.1101/2022.03.25.485803v1).                    
""")

         