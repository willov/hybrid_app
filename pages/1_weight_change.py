import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import math 
import altair as alt
from array import array
import plotly.express as px
import plotly.graph_objects as go

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

    features = model.featurenames
    return model, features

model, model_features = setup_model('insres_model')

# Define functions needed

def flatten(list):
    return [item for sublist in list for item in sublist]

def simulate(m, anthropometrics, stim, t_start_sim, n):
    act = sund.Activity(timeunit = 'd')
    pwc = sund.PIECEWISE_CONSTANT # space saving only
    const = sund.CONSTANT # space saving only

    for key,val in stim.items():
        act.AddOutput(name = key, type=pwc, tvalues = val["t"], fvalues = val["f"]) 
    for key,val in anthropometrics.items():
        act.AddOutput(name = key, type=const, fvalues = val) 

    sim = sund.Simulation(models = m, activities = act, timeunit = 'd')
    
    sim.ResetStatesDerivatives()

    # Getting initial values
    
    fs = []
    for path, subdirs, files in os.walk('./results'):
        for name in files:
            if 'inits' in name.split('(')[0] and "ignore" not in path:
                fs.append(os.path.join(path, name))
    fs.sort()
    with open(fs[0],'r') as f:
        inits_in = json.load(f)
        inits = inits_in['x']
  
    inits[1:5] = [anthropometrics[i] for i in ['Ginit','ECFinit','Finit','Linit']]
    sim.Simulate(timevector = np.linspace(min(stim["ss_x"]["t"]), max(stim["ss_x"]["t"]), 10000), statevalues = inits)
   
    sim_results = pd.DataFrame(sim.featuredata,columns=sim.featurenames)
    sim_results.insert(0, 'Time', sim.timevector)

    sim_diet_results = sim_results[(sim_results['Time']>=t_start_sim)]
    inits = sim.statevalues
    return sim_diet_results, inits

def simulate_meal(m, anthropometrics, stim, inits, t_start_sim, n):
    act = sund.Activity(timeunit = 'd')
    pwc = sund.PIECEWISE_CONSTANT # space saving only
    const = sund.CONSTANT # space saving only

    for key,val in stim.items():
        act.AddOutput(name = key, type=pwc, tvalues = val["t"], fvalues = val["f"]) 
    for key,val in anthropometrics.items():
        act.AddOutput(name = key, type=const, fvalues = val)

    sim = sund.Simulation(models = m, activities = act, timeunit = 'd')

    sim.Simulate(timevector = np.linspace(min(stim["ss_x"]["t"]), max(stim["ss_x"]["t"]), 10000), statevalues = inits)
   
    sim_results = pd.DataFrame(sim.featuredata,columns=sim.featurenames)


    with open(f'simulate_meal_{n}_simres_t_{t_start_sim}.json', 'w') as f:
        json.dump(sim_results.to_dict(), f, cls=NumpyArrayEncoder )

    sim_results.insert(0, 'Time', sim.timevector)
    sim_meal_results = sim_results[(sim_results['Time']>=t_start_sim)]
    sim_meal_results['Time'] = sim_meal_results['Time']*24.0*60.0 

    return sim_meal_results


# Start the app

st.title("Simulation weight change")
st.markdown("""Using the model for insulin resistance and weight change, you can here simulate the dynamics of different changes in energy intake based on custom anthropometrics. 

Below, you can specify how big change in energy intake you want to simulate and when/how big meals you want to simulate.

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
st.session_state['Ginit'] = 0.5
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
anthropometrics["height"] = st.number_input("Height (m):", 0.0, 2.5, st.session_state['height'],  key="height") 
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
diet_length = st.number_input("Diet length (years): ", 0.0, 100.0, 40.0, 0.1, key=f"diet_length")
diet = st.number_input("Change in kcal of diet (kcal): ", -1000.0, 1000.0, 400.0, 1.0, key=f"EIchange")
EIchange = [0.0] + [0.0] + [0.0] + [diet] + [0.0] 
t_long = [st.session_state['age']*365.0-10.0] + [st.session_state['age']*365.0] + [diet_start*365.0] + [(st.session_state['age']+diet_length)*365.0] 
ss_x = [0] + [0] + [1] + [1] + [0] 

stim_long = {
    "EIchange": {"t": t_long, "f": EIchange},
    "ss_x": {"t": t_long, "f": ss_x},
    }

t_start_sim = min(stim_long["ss_x"]["t"])+10.0
sim_long, inits = simulate(model, anthropometrics, stim_long, t_start_sim, -1)
sim_long['Time'] = sim_long['Time']/365.0

st.divider()
st.subheader("Meals")

meal_amount = []
meal_kcal = []
meal_times = []
meal_time = []

n_meals = st.slider("Number of (solid) meals:", 0, 5, 1)
sim_meal = list(range(n_meals))

for i in range(n_meals):
    st.markdown(f"**Meal {i+1}**")
    meal_time.append(st.number_input("Time of meal (age): ", start_time, start_time+diet_length, min(start_time+10*i, start_time+diet_length), key=f"meal_times{i}")*365.0)
    meal_kcal.append(st.number_input("Size of meal (kcal): ",0.0, 10000.0, 312.0, key=f"diet_kcals{i}"))
    t_before_meal = t_long[0:3] + [meal_time[i]] 
    stim_before_meal = {
    "EIchange": {"t": t_before_meal, "f": EIchange},
    "ss_x": {"t": t_before_meal, "f": ss_x},
        }

    t_start_sim = min(stim_before_meal["ss_x"]["t"])+10.0
    sim_before_meal, inits_meal = simulate(model, anthropometrics, stim_before_meal, t_start_sim, i)

    meal_times = [0.0] + [0.001] + [0.3]
    meal_amount = [0.0] + [0.0] + [meal_kcal[i]] + [0.0]
    meal = [0.0] + [0.0] + [1.0] + [0.0]
    ss_x_meal = [1.0] + [1.0] + [1.0] + [1.0]

    stim_meal = {
    "meal_amount": {"t": meal_times, "f": meal_amount},
    "meal": {"t": meal_times, "f": meal},
    "meal_time": {"t": meal_times, "f": meal},
    "ss_x": {"t": meal_times, "f": ss_x_meal},
        }

    sim_meal[i] = simulate_meal(model, anthropometrics, stim_meal, inits_meal, 0.0, i)
    st.divider()

if n_meals < 1.0:
    st.divider()

# Plotting weight change and meals
st.subheader("Plotting long term simulation of weight change")

feature_long = st.selectbox("Feature of the model to plot", model_features, key="long_plot")

l = (
    alt.Chart(sim_long).mark_point().encode(
    x = alt.X('Time').scale(zero=False).title('Time(age)'),
    y = alt.Y(feature_long).scale(zero=False)
))

st.altair_chart(l, use_container_width=True)

if n_meals > 0.0:
    st.divider()

    st.subheader("Plotting meal simulations")
    feature_meal = st.selectbox("Feature of the model to plot", model_features[5:], key="meal_plot")

    for i in range(n_meals):
        to_plot = pd.DataFrame(sim_meal[0]['Time'])
        column_names = ['Time']
        
        sim_feature = sim_meal[i][feature_meal]

        sim_feature.index = to_plot.index
        to_plot = pd.concat([to_plot,sim_feature], axis=1)
        meal_str = str(meal_kcal[i]) + ' kcal meal at age ' + str(meal_time[i]/365.0)
        column_names.append(meal_str)

        to_plot.columns = column_names
        to_plot = to_plot.reset_index(drop=True)
        to_plot = to_plot.set_index('Time')
        plot_data = to_plot.reset_index().melt('Time')

        m = (
        alt.Chart(plot_data).mark_line().encode(
            x=alt.X('Time').scale(zero=False).title('Time (minutes)'),
            y=alt.Y('value').scale(zero=False).title(feature_meal),
            color=alt.Color('variable', legend=alt.Legend(orient='bottom')).title("meal")
        ))

        st.altair_chart(m, use_container_width=True)
