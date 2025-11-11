import streamlit as st
import altair as alt

# Import shared utilities
from functions import setup_model, simulate_bp, extract_bp_from_table, setup_custom_packages

# Setup
setup_custom_packages()
model, model_features = setup_model('bloodpressure_model')


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
IC_SBP, IC_DBP = extract_bp_from_table(SBP0, DBP0, start_time)

anthropometrics = {"IC_SBP": IC_SBP, "IC_DBP": IC_DBP}

# Setup stimulation to the model

stim = {
    "drug_on": {"t": t_long, "f": drug_on}
}

# Plotting blood pressure 

sim = simulate_bp(model, stim, anthropometrics, initials) 

st.subheader("Plotting blood pressure over time")

feature = st.selectbox("Feature of the model to plot", model_features)

c = (
    alt.Chart(sim).mark_point().encode(
    x = alt.X('Time').scale(zero=False).title('Time(age)'),
    y = alt.Y(feature).scale(zero=False)
))

st.altair_chart(c, width='stretch')

