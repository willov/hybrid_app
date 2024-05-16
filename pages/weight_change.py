import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import math 

# testing testing

# Install sund in a custom location
import subprocess
import sys
if "sund" not in os.listdir('./custom_package'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=./custom_package", 'https://isbgroup.eu/edu/assets/sund-1.0.1.tar.gz#sha256=669a1d05c5c8b68500086e183d831650277012b3ea57e94356de1987b6e94e3e'])

sys.path.append('./custom_package')
import sund

st.elements.utils._shown_default_value_warning=True # This is not a good solution, but it hides the warning of using default values and sessionstate api

# Setup the models

def setup_model(model_name):
    sund.installModel(f"./models/{model_name}.txt")
    model_class = sund.importModel(model_name)
    model = model_class() 

    fs = []
    for path, subdirs, files in os.walk('./results'):
        for name in files:
            if model_name in name.split('(')[0] and "ignore" not in path:
                fs.append(os.path.join(path, name))
    fs.sort()
    with open(fs[0],'r') as f:
        param_in = json.load(f)
        params = param_in['x']

    model.parametervalues = params
    features = model.featurenames
    return model, features

model, model_features = setup_model('alcohol_model')

# Define functions needed

def flatten(list):
    return [item for sublist in list for item in sublist]

def simulate(m, anthropometrics, stim):
    act = sund.Activity(timeunit = 'days')
    pwc = sund.PIECEWISE_CONSTANT # space saving only
    const = sund.CONSTANT # space saving only

    for key,val in stim.items():
        act.AddOutput(name = key, type=pwc, tvalues = val["t"], fvalues = val["f"]) 
    for key,val in anthropometrics.items():
        act.AddOutput(name = key, type=const, fvalues = val) 
    
    sim = sund.Simulation(models = m, activities = act, timeunit = 'days')
    
    sim.ResetStatesDerivatives()
    t_start = anthropometrics.age
    # steady state first
    sim.Simulate(timevector = np.linspace(t_start, max(stim["EIchange"]["t"]), 10000))
    
    sim_results = pd.DataFrame(sim.featuredata,columns=sim.featurenames)
    sim_results.insert(0, 'Time', sim.timevector)

    #t_start_diet = 

    # sim_diet_results = sim_results[(sim_results['Time']>=t_start_diet)]
    #return sim_diet_results
    return sim_results

# Start the app

st.title("Simulation weight change")
st.markdown("""Using the model for insulin resistance and blood pressure, you can simulate the dynamics of different changes in energy intake and blood pressure medication based on custom anthropometrics. 

Below, you can specify how big change in energy intake you want to simulate and when/how big meals to simulate.

""")
   
# Anthropometrics
anthropometrics = {"sex": st.session_state['sex'], "weight": st.session_state['weight'], 
                   "height": st.session_state['height'], "age": st.session_state['age'], 
                   "Finit": st.session_state['Finit'], "Linit": st.session_state['Linit'],
                   "Ginit": st.session_state['Ginit'], "ECFinit": st.session_state['ECFinit']}

# Specifying diet
st.subheader("Diet")

#diet_time = []
EIchange = []
diet_length = []
t_long = []

st.divider()
start_time = st.session_state['age']

# diet_time(st.number_input("Start of diet (age): ", 0.0, 100.0, start_time, 0.1, key=f"diet_time"))
diet_length(st.number_input("Diet length (years): ", 0.0, 100.0, 20.0, 0.1, key=f"diet_length"))
EIchange(st.number_input("Change in kcal of diet (kcal): ", 0.0, 1000.0, 400, 1.0, key=f"EIchange"))
t_long(st.number_input("How long to simulate (years): ", 0.0, 100.0, 45.0, 1.0, key=f"t_long"))
t_long = t_long+anthropometrics.age
st.divider()

st.subheader(f"Meals")

meal_times = []
meal_kcals = []

n_meals = st.slider("Number of (solid) meals:", 0, 5, 1)

for i in range(n_meals):
    st.markdown(f"**Meal {i+1}**")
    meal_times.append(st.number_input("Time of meal (years): ", 0.0, diet_length, 0.1, key=f"meal_times{i}"))
    meal_kcals.append(st.number_input("Size of meal (kcal): ",0, 10000, 312, key=f"diet_kcals{i}"))
    start_time += 0.1
    st.divider()
if n_meals < 1.0:
    st.divider()

meal_amount = meal_kcals/4*1000 # converting from kcal to mg glucose
meal_amount = [0]+[k*on for k in meal_amount for on in [1 , 0]]
meal_times = [0]+[n*on for n in meal_times for on in [1 , 0]]

t_meal = [t_meal+(l/60)*on for t_meal,l in zip(meal_times, 0.3) for on in [0,1]]

# Setup stimulation to the model

stim_long = {
    "EIchange": {"t": t_long, "f": EIchange},
    "meal": {"t": t_long, "f": 0}
    }

stim_meal = {
    "meal_amount": {"t": t_meal, "f": meal_amount},
    "meal": {"t": t_meal, "f": 1}
    }

# Plotting weight change and meals

sim_long = simulate(model, anthropometrics, stim_long)
sim_meal = simulate(model, anthropometrics, stim_meal)

st.subheader("Plotting long term simulation of weight change")

feature = st.selectbox("Feature of the model to plot", model_features)
st.line_chart(sim_long, x="Time", y=feature)

st.subheader("Plotting meal simulations based on time points chosen in long term simulation")
feature = st.selectbox("Feature of the model to plot", model_features)
st.line_chart(sim_meal, x="Time", y=feature)