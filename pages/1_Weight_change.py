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
import copy

# Install sund in a custom location
import subprocess
import sys

os.makedirs('./custom_package', exist_ok=True)

if "sund" not in os.listdir('./custom_package'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=./custom_package", 'sund<=3.0'])

sys.path.append('./custom_package')
import sund

# Setup the models
def setup_model(model_name):
    sund.install_model(f"./models/{model_name}.txt")
    model = sund.load_model(model_name)

    features = model.feature_names
    return model, features

model, model_features = setup_model('insres_model')


# Define functions needed

def flatten(list):
    return [item for sublist in list for item in sublist]


def simulate(m, anthropometrics, stim, t_start_sim, n):
    act = sund.Activity(time_unit = 'd')

    for key,val in stim.items():
        act.add_output(
            name = key, type="piecewise_constant", 
            t = val["t"], f = val["f"]
        ) 
    for key,val in anthropometrics.items():
        act.add_output(name = key, type="constant", f = val) 

    sim = sund.Simulation(models = m, activities = act, time_unit = 'd')

    # Getting initial values
    initial_conditions = copy.deepcopy(m.state_values)

    initial_conditions[1:5] = [anthropometrics[i] for i in ['Ginit','ECFinit','Finit','Linit']]
    sim.simulate(time_vector = np.linspace(min(stim["ss_x"]["t"]), max(stim["ss_x"]["t"]), 10000), state_values = initial_conditions)

    sim_results = pd.DataFrame(sim.feature_values,columns=sim.feature_names)
    sim_results.insert(0, 'Time', sim.time_vector)

    sim_diet_results = sim_results[(sim_results['Time']>=t_start_sim)]
    new_initial_conditions = sim.state_values
    return sim_diet_results, new_initial_conditions


def simulate_meal(m, anthropometrics, stim, inits, t_start_sim, n):
    act = sund.Activity(time_unit = 'd')

    for key,val in stim.items():
        act.add_output(
        name = key, type="piecewise_constant",
        t = val["t"], f = val["f"]
    ) 
    for key,val in anthropometrics.items():
        act.add_output(name = key, type="constant", f = val)

    sim = sund.Simulation(models = m, activities = act, time_unit = 'd')

    sim.simulate(time_vector = np.linspace(min(stim["ss_x"]["t"]), max(stim["ss_x"]["t"]), 10000), state_values = inits)

    sim_results = pd.DataFrame(sim.feature_values,columns=sim.feature_names)

    sim_results.insert(0, 'Time', sim.time_vector)
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

anthropometrics = {
    "weight": st.session_state['weight'], "ECFinit": st.session_state['ECFinit'], 
    "height": st.session_state['height'], "age": st.session_state['age'], 
    "Finit": st.session_state['Finit'], "Linit": st.session_state['Linit'],
    "Ginit": st.session_state['Ginit']
}  

anthropometrics["sex"] = st.selectbox("Sex:", ["Man", "Woman"], ["Man", "Woman"].index(st.session_state['sex']), key="sex")
anthropometrics["weight"] = st.number_input("Weight (kg):", 0.0, 1000.0, st.session_state['weight'], key="weight") # max, min 
anthropometrics["age"] = st.number_input("Age (years):", 0.0, 100.0, st.session_state['age'], key="age") # max, min 
anthropometrics["height"] = st.number_input("Height (m):", 0.0, 2.5, st.session_state['height'],  key="height") 
anthropometrics["ECFinit"] = st.session_state['ECFinit']

# Handle fat and lean mass inputs

fat_known = st.checkbox("Do you know your fat mass?")
if fat_known:
    fat_pct = st.number_input("Fat percentage (%):", 0.0, 100.0, 
                                (st.session_state['Finit']/st.session_state['weight'])*100, 
                                0.1, key="Finit_pct")
    st.session_state['Finit'] = (fat_pct / 100.0) * st.session_state['weight']



lean_known = st.checkbox("Do you know your lean mass?")
if fat_known: 
    max_lean_pct = 100.0 - fat_pct
else:
    max_lean_pct = 100.0
if lean_known:
    lean_pct = st.number_input("Lean percentage (%):", 0.0, max_lean_pct, 
                                (st.session_state['Linit']/st.session_state['weight'])*100, 
                                0.1, key="Linit_pct")
    st.session_state['Linit'] = (lean_pct / 100.0) * st.session_state['weight']

# Adjust based on what's known
if fat_known and lean_known:
    # Both known: scale Gly and ECF to match total weight
    # BW = F + L + (1 + 2.7)*Gly + ECF
    remaining = st.session_state['weight'] - st.session_state['Finit'] - st.session_state['Linit']
    original_other = (1.0 + 2.7)*st.session_state['Ginit'] + st.session_state['ECFinit']
    
    if original_other > 0:
        scale_factor = remaining / original_other
        st.session_state['Ginit'] = st.session_state['Ginit'] * scale_factor
        st.session_state['ECFinit'] = st.session_state['ECFinit'] * scale_factor
        st.info(f"""
                Fat mass: {st.session_state['Finit']:.1f} kg  
                Lean mass: {st.session_state['Linit']:.1f} kg  
                Estimated glycogen mass: {st.session_state['Ginit']:.2f} kg  
                Estimated extracellular fluid mass: {st.session_state['ECFinit']:.2f} kg
                """)
    
elif fat_known and not lean_known:
    # Only fat known: adjust lean mass
    st.session_state['Linit'] = st.session_state['weight'] - (st.session_state['Finit'] + 
                                                                (1.0 + 2.7)*st.session_state['Ginit'] + 
                                                                st.session_state['ECFinit'])
    st.info(f"""
            Fat mass: {st.session_state['Finit']:.1f} kg  
            Estimated lean mass: {st.session_state['Linit']:.1f} kg
            """)
    
elif lean_known and not fat_known:
    # Only lean known: adjust fat mass
    st.session_state['Finit'] = st.session_state['weight'] - (st.session_state['Linit'] + 
                                                                (1.0 + 2.7)*st.session_state['Ginit'] + 
                                                                st.session_state['ECFinit'])
    st.info(f"Lean mass: {st.session_state['Linit']:.1f} kg, estimated fat mass: {st.session_state['Finit']:.1f} kg")

anthropometrics["Finit"] = st.session_state['Finit']
anthropometrics["Linit"] = st.session_state['Linit']

# Map sex to numerical representation
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
EIchange = [0.0, 0.0, 0, diet, diet] 
t_long = [st.session_state['age']*365.0-28.0, st.session_state['age']*365, diet_start*365.0, (st.session_state['age']+diet_length)*365.0] 
ss_x = [0, 0, 0, 1, 0] 

stim_long = {
    "EIchange": {"t": t_long, "f": EIchange},
    "ss_x": {"t": t_long, "f": ss_x},
}

t_start_sim = min(stim_long["ss_x"]["t"])
sim_long, inits = simulate(model, anthropometrics, stim_long, t_start_sim, -1)
sim_long['Time'] = sim_long['Time']/365.0

st.divider()
st.subheader("Meals to simulate")

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

    t_start_sim = min(stim_before_meal["ss_x"]["t"])
    sim_before_meal, inits_meal = simulate(model, anthropometrics, stim_before_meal, t_start_sim, i)

    meal_times = [0.0] + [0.001] + [0.3]
    meal_amount = [0.0] + [0.0] + [meal_kcal[i]] + [0.0]
    ss_x_meal = [1.0] + [1.0] + [1.0] + [1.0]

    stim_meal = {
        "meal_amount": {"t": meal_times, "f": meal_amount},
        "ss_x": {"t": meal_times, "f": ss_x_meal},
    }

    sim_meal[i] = simulate_meal(model, anthropometrics, stim_meal, inits_meal, 0.0, i)
    st.divider()

if n_meals < 1.0:
    st.divider()

# Plotting weight change and meals
st.subheader("Long term simulation of weight change")

feature_long = st.selectbox("Feature of the model to plot", model_features, key="long_plot")

l = (
    alt.Chart(sim_long[sim_long['Time']>st.session_state['age']]).mark_line().encode(
    x = alt.X('Time').scale(zero=False).title('Time(age)'),
    y = alt.Y(feature_long).scale(zero=False)
))

st.altair_chart(l, width='stretch')

if n_meals > 0.0:
    st.divider()

    st.subheader("Meal simulations at different ages")
    feature_meal = st.selectbox("Feature of the model to plot", model_features[5:], key="meal_plot")

    for i in range(n_meals):
        st.markdown(f"#### Meal simulation at age {meal_time[i]/365.0} years ({meal_kcal[i]} kcal)")
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
            # color=alt.Color('variable', legend=alt.Legend(orient='bottom')).title("")
        ))

        st.altair_chart(m, width='stretch')
