import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import math 
import altair as alt

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

    # fs = []
    # for path, subdirs, files in os.walk('./results'):
    #    for name in files:
    #        if model_name in name.split('(')[0] and "ignore" not in path:
    #            fs.append(os.path.join(path, name))
    # fs.sort()
    #with open(fs[0],'r') as f:
    #    param_in = json.load(f)
    #    params = param_in['x']

    # model.parametervalues = params

    features = model.featurenames
    return model, features

model, model_features = setup_model('bloodpressure_model')

# Define functions needed

def flatten(list):
    return [item for sublist in list for item in sublist]

def simulate(m, stim, anthropometrics, initials): #, extra_time = 10):
    act = sund.Activity(timeunit = 'y')
    pwc = sund.PIECEWISE_CONSTANT # space saving only
    const = sund.CONSTANT # space saving only

    for key,val in stim.items():
        act.AddOutput(name = key, type=pwc, tvalues = val["t"], fvalues = val["f"]) 
    for key,val in anthropometrics.items():
        act.AddOutput(name = key, type=const, fvalues = val) 

    sim = sund.Simulation(models = m, activities = act, timeunit = 'y')

    sim.ResetStatesDerivatives()
    t_start = min(stim["drug_on"]["t"])

    sim.Simulate(timevector = np.linspace(t_start, max(stim["drug_on"]["t"]), 10000), statevalues = initials) # +extra_time
    
    sim_results = pd.DataFrame(sim.featuredata,columns=sim.featurenames)
    sim_results.insert(0, 'Time', sim.timevector)
    
    return sim_results

# Start the app

st.title("Simulation of blood pressure change")
st.markdown("""Your blood pressure changes as you age, and can be lowered using different blood pressure medications.

Below, you can specify for how long you want to simulate and if you want to take a blood pressure medication or not.

""")

if 'age' not in st.session_state:
    st.session_state['age'] = 40.0
if 'IC_SBP' not in st.session_state:
    st.session_state['IC_SBP'] = 150.0
if 'IC_DBP' not in st.session_state:
    st.session_state['IC_DBP'] = 80.0


anthropometrics = {"IC_SBP": st.session_state['IC_SBP'], "IC_DBP": st.session_state['IC_DBP']}

# Specifying blood pressure medication
st.subheader("Blood pressure")

#n_med = st.slider("Number of periods of blood pressure medication:", 1, 5, 1)

anthropometrics["IC_SBP"] = st.number_input("Systolic blood pressure at start (kg):", 40.0, 300.0, st.session_state.IC_SBP, 0.1, key="IC_SBP")
anthropometrics["IC_DBP"] = st.number_input("Diastolic blood pressure at start (kg):", 40.0, 200.0, st.session_state.IC_DBP, 0.1, key="IC_DBP")

start_time = st.session_state['age']
end_time = start_time + st.number_input("How long time do you want to simulate (years): ", 0.0, 200.0, 40.0, key=f"end_time")

initials = [st.session_state['IC_SBP'], st.session_state['IC_DBP']]

med_times = []
med_lengths = [] 
med_period = [] 
t_long = []
drug_on = [0] + [0] + [0] 

st.divider()

if 'age' not in st.session_state:
    st.session_state['age'] = 40.0

med_times = [start_time]

#for i in range(n_med):
st.markdown(f"**Blood pressure medication**")


take_BPmed = st.checkbox("Do you know want to add a blood pressure medication?")
if take_BPmed:
    med_times.append(st.number_input("Start of blood pressure medication (age): ", start_time, 100.0, start_time, key=f"BP_med"))
    med_times.append(med_times[1] + 2)
    np.disp(med_times)
    #extra_time = st.number_input("Additional time to simulate after medication(s) (years):", 0.0, 100.0, 0.0, 0.1)
    drug_on = [0] + [0] + [1] + [0] + [0] #[0] + [1, 0] * n_med

    #med_period.append(st.number_input("How long period of blood pressure medication (years): ", 0.0, 200.0, 40.0, key=f"t_long{i}"))
    #start_time += med_period[i]

med_times.append([end_time])
t_long = med_times # [time for t,l in zip(med_times, med_lengths) for time in (t,t+l)]
st.divider()

np.disp(drug_on)
np.disp(t_long)

# Setup stimulation to the model

stim = {
    "drug_on": {"t": t_long, "f": drug_on}
    }

# Plotting blood pressure 

sim = simulate(model, stim, anthropometrics, initials) #, extra_time=extra_time)

st.subheader("Plotting blood pressure over time")

feature = st.selectbox("Feature of the model to plot", model_features)
# st.line_chart(sim, x="Time", y=feature, use_container_width=True)

c = (
    alt.Chart(sim).mark_point().encode(
    x = alt.X('Time').scale(zero=False),
    y = alt.Y(feature).scale(zero=False)
))

st.altair_chart(c, use_container_width=True)

