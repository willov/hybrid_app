import os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt


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

model, model_features = setup_model('bloodpressure_model')


# Define functions needed

def flatten(list):
    return [item for sublist in list for item in sublist]


def simulate(m, stim, anthropometrics, initials): #, extra_time = 10):
    act = sund.Activity(time_unit = 'y')

    for key,val in stim.items():
        act.add_output(
            name = key, type="piecewise_constant",
            t = val["t"], f = val["f"]
        ) 
    for key,val in anthropometrics.items():
        act.add_output(name = key, type="constant", f = val)

    sim = sund.Simulation(models = m, activities = act, time_unit = 'y')

    t_start = min(stim["drug_on"]["t"])

    sim.simulate(time_vector = np.linspace(t_start, max(stim["drug_on"]["t"]), 10000), state_values = initials) # +extra_time
    
    sim_results = pd.DataFrame(sim.feature_values,columns=sim.feature_names)
    sim_results.insert(0, 'Time', sim.time_vector)
    
    return sim_results


# Start the app

st.title("Simulation of blood pressure change")
st.markdown("""Your blood pressure changes as you age, and can be lowered using different blood pressure medications.

Below, you can specify for how long you want to simulate and if you want to take a blood pressure medication or not.

""")

if 'SBP0' not in st.session_state:
    st.session_state['SBP0'] = 131.613
if 'DBP0' not in st.session_state:
    st.session_state['DBP0'] = 83.4642

# Specifying blood pressure medication
st.subheader("Blood pressure")

SBP0 = st.number_input("Systolic blood pressure at start (mmHg)):", 40.0, 300.0, st.session_state.SBP0, 1.0, key=f"SBP0")
DBP0 = st.number_input("Diastolic blood pressure at start (mmHg):", 40.0, st.session_state.SBP0, st.session_state.DBP0, 1.0, key=f"DBP0")

start_time = st.number_input("When do you want to start the simulation (age)?:", 0.0, 200.0, 40.0, key=f"start_time")
end_time = start_time + st.number_input("How long time do you want to simulate (years): ", 0.0, 200.0, 40.0, key=f"end_time")

initials = [st.session_state['SBP0'], st.session_state['DBP0']]

med_times = []
med_lengths = [] 
med_period = [] 
t_long = []
drug_on = [0] + [0] + [0] 

st.divider()

med_times = [start_time]

#for i in range(n_med):
st.markdown(f"**Blood pressure medication**")


take_BPmed = st.checkbox("Do you know want to add a blood pressure medication?")
if take_BPmed:
    med_times.append(st.number_input("Start of blood pressure medication (age): ", start_time, 100.0, 45.0, key=f"BP_med"))
    med_times.append(med_times[1] + 2)
    drug_on = [0] + [0] + [1] + [0] + [0] 

med_times.append(end_time)
t_long = med_times 
st.divider()

#%% find correct bp group based on sbp and dbp value
v = np.array([
    [111.472772277228, 117.860744407774, 125.223689035570, 131.612577924459],
    [112.666850018335, 119.611294462780, 124.055738907224, 133.362211221122],
    [113.166483314998, 121.500733406674, 129.834066740007, 139.556288962229],
    [114.221672167217, 124.084616795013, 133.390172350568, 142.557755775577],
    [114.584250091676, 126.251833516685, 136.390722405574, 149.169416941694],
    [117.724605793913, 128.419966996700, 139.669050238357, 153.558855885588],
    [120.168683535020, 131.002933626696, 142.947378071140, 156.558489182251],
    [121.361844517785, 133.585900256692, 144.697011367803, 157.475705903924],
    [123.529977997800, 135.335533553355, 147.420700403374, 161.170700403373],
    [124.170333700037, 135.837917125046, 148.476806013935, 167.644389438944],
    [126.197744774477, 136.893105977264, 146.893105977264, 165.3671617161719]
])

IC_DBPdata = np.array([71.7975011786893, 75.8451202263084, 80.6667452459532, 83.4641678453560])
IC_SBPdata = v[0,:]
dataage = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])


mindiff, chosenAgeIndex = min((abs(age - start_time), idx) for idx, age in enumerate(dataage))
chosenAge = dataage[chosenAgeIndex]
dataSBP = v[chosenAgeIndex, :]
mindiff, chosenColumn = min((abs(sbp - SBP0), idx) for idx, sbp in enumerate(dataSBP))

IC_DBP = IC_DBPdata[chosenColumn]
IC_SBP = IC_SBPdata[chosenColumn]

anthropometrics = {"IC_SBP": IC_SBP, "IC_DBP": IC_DBP}

# Setup stimulation to the model

stim = {
    "drug_on": {"t": t_long, "f": drug_on}
}

# Plotting blood pressure 

sim = simulate(model, stim, anthropometrics, initials) 

st.subheader("Plotting blood pressure over time")

feature = st.selectbox("Feature of the model to plot", model_features)

c = (
    alt.Chart(sim).mark_point().encode(
    x = alt.X('Time').scale(zero=False).title('Time(age)'),
    y = alt.Y(feature).scale(zero=False)
))

st.altair_chart(c, width='stretch')

