import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import math 
import altair as alt
from array import array

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

    # model.parametervalues = params
    features = model.featurenames
    return model, features

model, model_features = setup_model('insres_model')

# Define functions needed

def flatten(list):
    return [item for sublist in list for item in sublist]

def simulate(m, anthropometrics, stim):
    act = sund.Activity(timeunit = 'd')
    pwc = sund.PIECEWISE_CONSTANT # space saving only
    const = sund.CONSTANT # space saving only

    for key,val in stim.items():
        act.AddOutput(name = key, type=pwc, tvalues = val["t"], fvalues = val["f"]) 
    for key,val in anthropometrics.items():
        act.AddOutput(name = key, type=const, fvalues = val) 

    sim = sund.Simulation(models = m, activities = act, timeunit = 'd')

    sim.ResetStatesDerivatives()
    t_start_diet = anthropometrics['age']*365.0-10

    fs = []
    for path, subdirs, files in os.walk('./results'):
        for name in files:
            if 'inits' in name.split('(')[0] and "ignore" not in path:
                fs.append(os.path.join(path, name))
    fs.sort()
    with open(fs[0],'r') as f:
        inits_in = json.load(f)
        inits = inits_in['x']
    np.disp(inits[1:4])
    inits[1:4] = [anthropometrics[i] for i in ['Ginit','ECFinit','Finit','Linit']]
    np.disp([anthropometrics[i] for i in ['Ginit','ECFinit','Finit','Linit']])
    np.disp(type([anthropometrics[i] for i in ['Ginit','ECFinit','Finit','Linit']]))
    np.disp(len(inits))
    np.disp(type(inits))
    np.disp(inits)
    test = [1, 2, 3]
    np.disp(type(test))
    # inits = array("f", inits)

    sim.Simulate(timevector = np.linspace(min(stim["ss_x"]["t"]), max(stim["ss_x"]["t"]), 10000), statevalues = inits)
    
    sim_results = pd.DataFrame(sim.featuredata,columns=sim.featurenames)
    sim_results.insert(0, 'Time', sim.timevector)

    sim_diet_results = sim_results[(sim_results['Time']>=t_start_diet)]
    #return sim_diet_results
    return sim_diet_results

# Start the app

st.title("Simulation weight change")
st.markdown("""Using the model for insulin resistance and blood pressure, you can simulate the dynamics of different changes in energy intake and blood pressure medication based on custom anthropometrics. 

Below, you can specify how big change in energy intake you want to simulate and when/how big meals to simulate.

""")

# Setting anthropometrics
st.divider()   
st.subheader("Anthropometrics")
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
    st.session_state['age'] = 40.0
st.session_state['Ginit'] = (1.0 + 2.7)*0.5
st.session_state['ECFinit'] = 0.7*0.235*st.session_state['weight']  
if 'Finit' not in st.session_state:
    if st.session_state['sex']== 'Woman':
        st.session_state['Finit'] = (st.session_state['weight']/100.0)*(0.14*st.session_state['age'] + 39.96*math.log(st.session_state['weight']/((st.session_state['height'])**2.0)) - 102.01)
    elif st.session_state['sex']== 'Man': 
        st.session_state['Finit'] = (st.session_state['weight']/100.0)*(0.14*st.session_state['age'] + 37.31*math.log(st.session_state['weight']/((st.session_state['height'])**2.0)) - 103.95) 
if 'Linit' not in st.session_state:
    st.session_state['Linit'] = st.session_state['weight'] - (st.session_state['Finit'] + (1.0 + 2.7)*st.session_state['Ginit'] + st.session_state['ECFinit'])

anthropometrics = {"weight": st.session_state['weight'], "ECFinit": st.session_state['ECFinit'], 
                   "height": st.session_state['height'], "age": st.session_state['age'], 
                   "Finit": st.session_state['Finit'], "Linit": st.session_state['Linit'],
                   "Ginit": st.session_state['Ginit']}  

anthropometrics["sex"] = st.selectbox("Sex:", ["Man", "Woman"], ["Man", "Woman"].index(st.session_state['sex']), key="sex")
anthropometrics["weight"] = st.number_input("Weight (kg):", 0.0, 1000.0, st.session_state['weight'], key="weight") # max, min 
anthropometrics["age"] = st.number_input("Age (years):", 0.0, 100.0, st.session_state['age'], key="age") # max, min 
anthropometrics["height"] = st.number_input("Height (m):", 0.0, 2.5, st.session_state['height'],  key="height") # st.session_state['height'], 0.1, 
anthropometrics["ECFinit"] = st.session_state['ECFinit']

fat_known = st.checkbox("Do you know your fat mass?")
if fat_known:
    st.session_state['Finit'] = st.number_input("Fat mass (kg):", 0.0, 1000.0, st.session_state.Finit, 0.1, key="Finit")

lean_known = st.checkbox("Do you know your lean mass?")
if lean_known:
   st.session_state['Linit'] = st.number_input("Lean mass (kg):", 0.0, 1000.0, st.session_state.Linit, 0.1, key="Linit")

anthropometrics["Finit"] = st.session_state['Finit']
anthropometrics["Linit"] = st.session_state['Linit']

anthropometrics["sex"] = float(anthropometrics["sex"].lower() in ["male", "man", "men", "boy", "1", "chap", "guy"]) #Converts to a numerical representation


# Specifying diet
st.divider()
st.subheader("Diet")

#diet_time = []
EIchange = []
diet_length = []
diet_start = []
t_long = []

start_time = st.session_state['age']

diet_start = st.number_input("Diet start (age): ", st.session_state['age'], 100.0, 40.0, 0.1, key=f"diet_start")
diet_length = st.number_input("Diet length (years): ", 0.0, 100.0, 20.0, 0.1, key=f"diet_length")
EIchange = st.number_input("Change in kcal of diet (kcal): ", -1000.0, 1000.0, 400.0, 1.0, key=f"EIchange")
EIchange = [0.0] + [0.0] + [0.0] + [EIchange] + [0.0] 
# t_long = st.number_input("How long to simulate (years): ", 0.0, 100.0, 45.0, 1.0, key=f"t_long")
t_long = [st.session_state['age']*365.0-10] + [st.session_state['age']*365.0] + [diet_start*365.0] + [(st.session_state['age']+diet_length)*365.0] 
ss_x = [0] + [0] + [1] + [1] + [0] 

st.divider()
st.subheader("Meals")

meal_times = []
meal_amounts = []

n_meals = st.slider("Number of (solid) meals:", 0, 5, 1)

for i in range(n_meals):
    st.markdown(f"**Meal {i+1}**")
    meal_times.append(st.number_input("Time of meal (years): ", 0.0, diet_length, 0.1, key=f"meal_times{i}"))
    meal_amounts.append(st.number_input("Size of meal (kcal): ",0.0, 10000.0, 312.0, key=f"diet_kcals{i}"))
    start_time += 0.1
    st.divider()
if n_meals < 1:
    st.divider()
# meal_amount = [0]+[k*on for k in meal_amount for on in [1 , 0]]
# meal_times = [0]+[n*on for n in meal_times for on in [1 , 0]]

# t_meal = [t_meal+(l/60)*on for t_meal,l in zip(meal_times, 0.3) for on in [0,1]] # varje gång något ska ändras

# Setup stimulation to the model

stim_long = {
    "EIchange": {"t": t_long, "f": EIchange},
    "ss_x": {"t": t_long, "f": ss_x},
    }

# Plotting weight change and meals

sim_long = simulate(model, anthropometrics, stim_long)

st.subheader("Plotting long term simulation of weight change")

feature_long = st.selectbox("Feature of the model to plot", model_features, key="long_plot")
# st.line_chart(sim_long, x="Time", y=feature_long)

l = (
    alt.Chart(sim_long).mark_point().encode(
    x = alt.X('Time').scale(zero=False),
    y = alt.Y(feature_long).scale(zero=False)
))

st.altair_chart(l, use_container_width=True)

st.divider()

st.subheader("Plotting meal simulations based on time points chosen in long term simulation")
feature_meal = st.selectbox("Feature of the model to plot", model_features, key="meal_plot")

for i in range(n_meals):
    meal_amount = [0.0] + [meal_amounts[i]] + [meal_amounts[i]] + [0]
    meal = [0.0] + [1.0] + [1.0] + [0.0]
    ss_x = [0.0] + [1.0] + [1.0] + [0.0]

    meal_time = [meal_times[i]-10] + [meal_times[i]] + [meal_times[i] + 0.3]
    stim_meal = {
    "meal_amount": {"t": meal_time, "f": meal_amount},
    "meal": {"t": meal_time, "f": meal},
    "ss_x": {"t": meal_time, "f": ss_x}
    }
    sim_meal = simulate(model, anthropometrics, stim_meal)
    # st.line_chart(sim_meal, x="Time", y=feature_meal)
    m = (
    alt.Chart(sim_meal).mark_point().encode(
    x = alt.X('Time').scale(zero=False),
    y = alt.Y(feature_meal).scale(zero=False)
        ))