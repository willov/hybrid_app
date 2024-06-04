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

    np.disp(initials)
    np.disp(type(initials))
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


anthropometrics = {"IC_SBP": st.session_state['IC_SBP'], "IC_DBP": st.session_state['IC_DBP'], "age": st.session_state['age']}

# Specifying blood pressure medication
st.subheader("Blood pressure")

#n_med = st.slider("Number of periods of blood pressure medication:", 1, 5, 1)

userSBP = st.number_input("Systolic blood pressure at start (kg):", 40.0, 300.0, st.session_state.IC_SBP, 0.1, key=f"IC_SBP")
userDBP = st.number_input("Diastolic blood pressure at start (kg):", 40.0, 200.0, st.session_state.IC_DBP, 0.1, key=f"IC_DBP")

start_time = st.number_input("When do you want to start the simulation (age)?:", 0.0, 200.0, st.session_state['age'], key=f"age")
end_time = start_time + st.number_input("How long time do you want to simulate (years): ", 0.0, 200.0, 40.0, key=f"end_time")

v = [[111.472772277228, 117.860744407774, 125.223689035570, 131.612577924459],
[112.666850018335, 119.611294462780, 124.055738907224, 133.362211221122],
[113.166483314998, 121.500733406674, 129.834066740007, 139.556288962229],
[114.221672167217, 124.084616795013, 133.390172350568, 142.557755775577],
[114.584250091676, 126.251833516685, 136.390722405574, 149.169416941694],
[117.724605793913, 128.419966996700, 139.669050238357, 153.558855885588],
[120.168683535020, 131.002933626696, 142.947378071140, 156.558489182251],
[121.361844517785, 133.585900256692, 144.697011367803, 157.475705903924],
[123.529977997800, 135.335533553355, 147.420700403374, 161.170700403373],
[124.170333700037, 135.837917125046, 148.476806013935, 167.644389438944],
[126.197744774477, 136.893105977264, 146.893105977264, 165.3671617161719]]

IC_DBPdata = [71.7975011786893,	75.8451202263084,	80.6667452459532, 83.4641678453560]
IC_SBPdata = v[0]
np.disp(v[0])
dataage = [30,
35,
40,
45,
50,
55,
60,
65,
70,
75,
80]

diff_time = [x-start_time for x in dataage]
[mindiff,chosenAgeIndex] = min([abs(x) for x in diff_time])
chosenAge = dataage(chosenAgeIndex)
dataSBP = v[chosenAgeIndex,:]
diff_SBP = (dataSBP-userSBP)
[mindiff,chosenColumn] = min([abs(x) for x in diff_SBP])

anthropometrics["IC_DBP"] = IC_DBPdata(chosenColumn)
anthropometrics["IC_SBP"] = IC_SBPdata(chosenColumn)

np.disp(IC_DBPdata(chosenColumn))

initials = [anthropometrics["IC_SBP"], anthropometrics["IC_DBP"]]

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
    med_times.append(med_times[1] + 5)
    #extra_time = st.number_input("Additional time to simulate after medication(s) (years):", 0.0, 100.0, 0.0, 0.1)
    drug_on = [0] + [0] + [1] + [0] + [0] #[0] + [1, 0] * n_med

    #med_period.append(st.number_input("How long period of blood pressure medication (years): ", 0.0, 200.0, 40.0, key=f"t_long{i}"))
    #start_time += med_period[i]

med_times.append(end_time)
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

