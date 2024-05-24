import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import math 

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
    #with open(fs[0],'r') as f:
    #    param_in = json.load(f)
    #    params = param_in['x']

    # model.parametervalues = params
    features = model.featurenames
    return model, features

model, model_features = setup_model('insres_model')

# Define functions needed

def flatten(list):
    return [item for sublist in list for item in sublist]

def simulate(m, anthropometrics, stim):
    act = sund.Activity(timeunit = 'days')
    pwc = sund.PIECEWISE_CONSTANT # space saving only
    const = sund.CONSTANT # space saving only

    np.disp(anthropometrics.items())
    for key,val in stim.items():
        act.AddOutput(name = key, type=pwc, tvalues = val["t"], fvalues = val["f"]) 
    for key,val in anthropometrics.items():
        act.AddOutput(name = key, type=const, fvalues = val) 
    
    sim = sund.Simulation(models = m, activities = act, timeunit = 'days')
    
    np.disp(model.initialvalues)
    np.disp(model)
    sim.ResetStatesDerivatives()
    t_start = min(stim["EIchange"]["t"])
    # TODO steady state 
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

if 'sex' not in st.session_state:
    st.session_state['sex'] = 'Man'
if 'weight' not in st.session_state:
    st.session_state['weight'] = 90.0
if 'height' not in st.session_state:
    st.session_state['height'] = 1.85
if 'age' not in st.session_state:
    st.session_state['age'] = 40.0
st.session_state['Ginit'] = (1.0 + 2.7)*0.5
st.session_state['ECFinit'] = 0.7*0.235*st.session_state['weight']  
if 'Finit' not in st.session_state:
    if st.session_state['sex']== 'Woman':
        st.session_state['Finit'] = (st.session_state['weight']/100.0)*(0.14*st.session_state['age'] + 39.96*math.log(st.session_state['weight']/((0.01*st.session_state['height'])**2.0)) - 102.01)
    elif st.session_state['sex']== 'Man': 
        st.session_state['Finit'] = (st.session_state['weight']/100.0)*(0.14*st.session_state['age'] + 37.31*math.log(st.session_state['weight']/((0.01*st.session_state['height'])**2.0)) - 103.95) 
if 'Linit' not in st.session_state:
    st.session_state['Linit'] = st.session_state['weight'] - (st.session_state['Finit'] + (1.0 + 2.7)*st.session_state['Ginit'] + st.session_state['ECFinit'])

anthropometrics = {"weight": st.session_state['weight'], "ECFinit": st.session_state['ECFinit', 
                   "height": st.session_state['height'], "age": st.session_state['age'], 
                   "Finit": st.session_state['Finit'], "Linit": st.session_state['Linit'],
                   "Ginit": st.session_state['Ginit']]} # , "sex": st.session_state['sex']

np.disp(anthropometrics)

# Specifying diet
st.divider()
st.subheader("Diet")

#diet_time = []
EIchange = []
diet_length = []
diet_start = []
t_long = []

start_time = st.session_state['age']

# diet_time(st.number_input("Start of diet (age): ", 0.0, 100.0, start_time, 0.1, key=f"diet_time"))
diet_start = st.number_input("Diet start (years): ", st.session_state['age'], 100.0, 40.0, 0.1, key=f"diet_start")
diet_length = st.number_input("Diet length (age): ", 0.0, 100.0, 20.0, 0.1, key=f"diet_length")
EIchange = st.number_input("Change in kcal of diet (kcal): ", -1000.0, 1000.0, 400.0, 1.0, key=f"EIchange")
EIchange = [0.0] + [0.0] + [EIchange] + [0.0]
np.disp(EIchange)
# t_long = st.number_input("How long to simulate (years): ", 0.0, 100.0, 45.0, 1.0, key=f"t_long")
t_long = [st.session_state['age']] + [diet_start] + [st.session_state['age']+diet_length] 
np.dips(t_long)

st.divider()
st.subheader("Meals")

meal_times = []
meal_kcals = []

n_meals = st.slider("Number of (solid) meals:", 0, 5, 1)

for i in range(n_meals):
    st.markdown(f"**Meal {i+1}**")
    meal_times.append(st.number_input("Time of meal (years): ", 0.0, diet_length, 0.1, key=f"meal_times{i}"))
    meal_kcals.append(st.number_input("Size of meal (kcal): ",0.0, 10000.0, 312.0, key=f"diet_kcals{i}")/4.0*1000.0)
    start_time += 0.1
    st.divider()
if n_meals < 1:
    st.divider()

meal_amount = meal_kcals #/4*1000 # converting from kcal to mg glucose
meal = [0.0] + [0.0] + [0.0] + [0.0] 
# meal_amount = [0]+[k*on for k in meal_amount for on in [1 , 0]]
# meal_times = [0]+[n*on for n in meal_times for on in [1 , 0]]

# t_meal = [t_meal+(l/60)*on for t_meal,l in zip(meal_times, 0.3) for on in [0,1]] # varje gång något ska ändras

# Setup stimulation to the model

stim_long = {
    "EIchange": {"t": t_long, "f": EIchange},
    "meal": {"t": t_long, "f": meal},
    }

# Plotting weight change and meals

sim_long = simulate(model, anthropometrics, stim_long)

st.subheader("Plotting long term simulation of weight change")

feature = st.selectbox("Feature of the model to plot", model_features)
st.line_chart(sim_long, x="Time", y=feature)

st.subheader("Plotting meal simulations based on time points chosen in long term simulation")
feature = st.selectbox("Feature of the model to plot", model_features)

for i in range(n_meals):
    stim_meal = {
    "meal_amount": {"t": meal_times[i], "f": meal_amount[i]},
    "meal": {"t": meal_times[i], "f": 1.0}
    }
    sim_meal = simulate(model, anthropometrics, stim_meal)
    st.line_chart(sim_meal, x="Time", y=feature)