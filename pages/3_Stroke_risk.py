"""
Page 3: Continuous stroke risk simulation
Combines weight, blood pressure, and metabolic simulations with stroke risk prediction.
"""

import pandas as pd
import numpy as np
import streamlit as st
import math
import altair as alt

# Import shared utilities
from functions import (
    setup_custom_packages, setup_model, simulate_insres_weight, 
    simulate_bp, extract_bp_from_table,
    StrokeRiskEnsembleModel
)

# Setup
setup_custom_packages()

st.title("Continuous stroke risk simulation")
st.markdown("""
Simulate your 5-year stroke risk as it evolves over time based on changes in:
- Weight and metabolic factors (insulin resistance, blood glucose)
- Blood pressure levels
- Other clinical risk factors

This page integrates the physiological simulations from the previous pages with an ensemble machine learning model 
trained to predict stroke risk based on clinical characteristics.
""")

# Initialize session state for shared parameters
if 'sex' not in st.session_state:
    st.session_state['sex'] = 'Man'
if 'weight' not in st.session_state:
    st.session_state['weight'] = 67.6
if 'height' not in st.session_state:
    st.session_state['height'] = 1.85
if 'age' not in st.session_state:
    st.session_state['age'] = 40.0
if 'SBP0' not in st.session_state:
    st.session_state['SBP0'] = 131.613
if 'DBP0' not in st.session_state:
    st.session_state['DBP0'] = 83.4642

# Load models
st.spinner("Loading models...")
weight_model, weight_features = setup_model('insres_model')
bp_model, bp_features = setup_model('bloodpressure_model')
risk_model = StrokeRiskEnsembleModel()

# Define long-term features to exclude from meal plotting
long_term_features = [
    'Weight (kg)', 'BMI (kg/m^2)', 'Fat mass (kg)', 'Fat mass (%)',
    'Lean mass (kg)', 'Free fat mass (kg)', 'Diabetes'
]

st.divider()
st.subheader("Input parameters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    sex = st.selectbox("Sex:", ["Man", "Woman"], index=["Man", "Woman"].index(st.session_state['sex']), key="risk_sex")
    st.session_state['sex'] = sex

with col2:
    weight = st.number_input("Weight (kg):", 0.0, 1000.0, st.session_state['weight'], key="risk_weight")
    st.session_state['weight'] = weight

with col3:
    height = st.number_input("Height (m):", 0.0, 2.5, st.session_state['height'], key="risk_height")
    st.session_state['height'] = height

with col4:
    age = st.number_input("Age (years):", 0.0, 100.0, st.session_state['age'], key="risk_age")
    st.session_state['age'] = age

# Initialize body composition
Ginit = 0.5
ECFinit = 0.7 * 0.235 * weight

if st.session_state['sex'] == 'Woman':
    Finit = (weight / 100.0) * (
        0.14 * age + 39.96 * math.log(weight / (height ** 2)) - 102.01
    )
elif st.session_state['sex'] == 'Man':
    Finit = (weight / 100.0) * (
        0.14 * age + 37.31 * math.log(weight / (height ** 2)) - 103.95
    )

Linit = weight - (Finit + (1.0 + 2.7) * Ginit + ECFinit)

anthropometrics = {
    "weight": weight,
    "ECFinit": ECFinit,
    "height": height,
    "age": age,
    "Finit": Finit,
    "Linit": Linit,
    "Ginit": Ginit,
    "sex": float(sex.lower() in ["female", "woman", "women", "girl", "0", "lady"]) == 0
}

st.divider()
st.subheader("Blood pressure")

col1, col2 = st.columns(2)

with col1:
    SBP0 = st.number_input("Systolic BP (mmHg):", 40.0, 300.0, st.session_state['SBP0'], key="risk_SBP0")
    st.session_state['SBP0'] = SBP0

with col2:
    DBP0 = st.number_input("Diastolic BP (mmHg):", 40.0, 180.0, st.session_state['DBP0'], key="risk_DBP0")
    st.session_state['DBP0'] = DBP0


st.divider()
st.subheader("Simulation parameters")

col1, col2, col3 = st.columns(3)

with col1:
    start_time = st.number_input("Simulation start age (years):", age, 100.0, age, 1.0, key="risk_start_time")

with col2:
    end_time = st.number_input("Simulation end age (years):", start_time, 100.0, min(start_time + 20, 90.0), 1.0, key="risk_end_time")

with col3:
    diet_change = st.number_input("Diet change (kcal/day):", -1000.0, 1000.0, 100.0, 50.0, key="risk_diet_change")


st.divider()
st.subheader("Additional risk factors")

col1, col2, col3 = st.columns(3)

with col1:
    smoking = st.checkbox("Current smoker", value=False, key="risk_smoking")
    if smoking:
        cpd = st.number_input("Cigarettes per day (CPD):", 0, 100, 10, key="risk_cpd")
    else:
        cpd = 0

with col2:
    has_diabetes = st.checkbox("Type 2 diabetes", value=False, key="risk_diabetes")

    if has_diabetes:
        weight_model.state_values[weight_model.state_names.index('diabetes')] = 1.0

with col3:
    bp_medication = st.checkbox("On BP medication", value=False, key="risk_bp_med")
    if bp_medication:
        bp_med_start = st.number_input("Medication start age (years):", start_time, end_time, start_time + 2, 1.0, key="risk_bp_med_start")
        st.session_state['bp_med_start'] = bp_med_start

col1, col2, col3 = st.columns(3)

with col1:
    af_beforestroke = st.checkbox("Atrial Fibrillation", value=False, key="risk_af")

with col2:
    has_prior_stroke = st.checkbox("Prior stroke", value=False, key="risk_prior_stroke")

with col3:
    pass  # Placeholder for alignment

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

if st.button("Calculate Risk Trajectory", type="primary"):
    with st.spinner("Running simulations..."):
        # Create a common time vector for both simulations (in years, absolute time)
        time_common = np.linspace(start_time, end_time, 10000)
        time_common_days = time_common * 365.0  # Convert to days for weight simulation
        
        # Prepare weight simulation
        EIchange = [0.0, 0.0, 0.0, diet_change, diet_change]
        t_long = [
            start_time * 365.0 - 28.0,
            start_time * 365,
            start_time * 365.0,
            (end_time) * 365.0
        ]
        ss_x = [0, 0, 0, 1, 0]

        stim_long = {
            "EIchange": {"t": t_long, "f": EIchange},
            "ss_x": {"t": t_long, "f": ss_x},
        }

        t_start_sim = min(stim_long["ss_x"]["t"])

        # Run weight simulation with common time vector
        sim_weight, weight_inits = simulate_insres_weight(
            weight_model, anthropometrics, stim_long, t_start_sim, time_vector=time_common_days
        )
        sim_weight['Time'] = sim_weight['Time'] / 365.0

        # Prepare blood pressure simulation
        IC_SBP, IC_DBP = extract_bp_from_table(SBP0, DBP0, start_time)
        initials_bp = [SBP0, DBP0]
        anthropometrics_bp = {"IC_SBP": IC_SBP, "IC_DBP": IC_DBP}

        drug_on_times = [start_time, end_time]
        drug_on_values = [0, 0, 0]
        if bp_medication:
            med_start = st.session_state.get('bp_med_start', start_time + 2)
            med_end = med_start + 2  # Medication lasts 2 years
            drug_on_times = [start_time, med_start, med_end, end_time]
            drug_on_values = [0, 0, 1, 0, 0]  # SUND requires len(f) == len(t) + 1

        stim_bp = {
            "drug_on": {"t": drug_on_times, "f": drug_on_values}
        }

        # Run BP simulation with common time vector
        sim_bp = simulate_bp(bp_model, stim_bp, anthropometrics_bp, initials_bp, time_vector=time_common)


        # ====================================================================
        # COMBINE SIMULATIONS AND CALCULATE RISK
        # ====================================================================

        # Extract weight values directly (already on common time grid)
        weight_interp = {}
        for col in sim_weight.columns:
            if col != 'Time':
                weight_interp[col] = sim_weight[col].values
    
        # Extract BP values directly (already on common time grid)
        bp_interp = {}
        for col in sim_bp.columns:
            bp_interp[col] = sim_bp[col].values

        # Prepare features for ensemble model
        # The ensemble model expects these keys (based on model training):
        # SEX (1=male, 2=female), AGE, SBP, DBP, BMI, CPD, AF_beforestroke, DMRX
        sex_numeric = 2.0 if sex == "Woman" else 1.0
        
        risk_scenario = {
            'AGE': time_common,
            'SEX': np.full_like(time_common, sex_numeric),
            'SBP': bp_interp['Systolic blood pressure (mmHg)'],
            'DBP': bp_interp['Diastolic blood pressure (mmHg)'],
            'BMI': weight_interp['BMI (kg/m^2)'],
            'CPD': np.full_like(time_common, float(cpd)),
            'AF_beforestroke': np.full_like(time_common, float(af_beforestroke)),
            'DMRX': weight_interp['Diabetes'],  # Diabetes medication/status
        }

        # Calculate risk using ensemble model
        risk_prediction = risk_model.predict_ensemble(risk_scenario)
        risk_trajectory = risk_prediction['ensemble_absolute_risk'] * 100  # Convert to percentage

        # Create combined results DataFrame
        results_df = pd.DataFrame({
            'Age': time_common,
            'Sex': np.full_like(time_common, sex, dtype=object),
            'Height': np.full_like(time_common, height),
            'Weight (kg)': weight_interp.get('Weight (kg)', np.nan),
            'BMI': weight_interp.get('BMI (kg/m^2)', np.nan),
            'SBP (mmHg)': bp_interp.get('Systolic blood pressure (mmHg)', np.nan),
            'DBP (mmHg)': bp_interp.get('Diastolic blood pressure (mmHg)', np.nan),
            'CPD': np.full_like(time_common, cpd),
            'Diabetes': weight_interp.get('Diabetes', np.nan),
            'Atrial Fibrillation': np.full_like(time_common, af_beforestroke),
            'Prior Stroke': np.full_like(time_common, has_prior_stroke),
            'Stroke Risk (%)': risk_trajectory
        })

        st.session_state['risk_results'] = results_df
        st.session_state['sim_weight'] = sim_weight
        st.session_state['sim_bp'] = sim_bp

        st.success("✓ Simulation complete!")


# ============================================================================
# DISPLAY RESULTS
# ============================================================================

if 'risk_results' in st.session_state:
    results_df = st.session_state['risk_results']

    st.divider()
    st.subheader("Results Summary")

    col1, col2, col3, col4 = st.columns(4)

    max_risk = results_df['Stroke Risk (%)'].max()
    min_risk = results_df['Stroke Risk (%)'].min()
    risk_change = max_risk - min_risk

    with col1:
        st.metric("Max Risk", f"{max_risk:.1f}%")

    with col2:
        st.metric("Min Risk", f"{min_risk:.1f}%")

    with col3:
        st.metric("Risk Change", f"{risk_change:+.1f}%")

    with col4:
        final_risk = results_df['Stroke Risk (%)'].iloc[-1]
        st.metric("Final Risk", f"{final_risk:.1f}%")

    st.divider()
    st.subheader("Stroke Risk Trajectory")

    # Risk trajectory chart
    risk_chart = alt.Chart(results_df).mark_line(point=True).encode(
        x=alt.X('Age:Q').scale(zero=False).title('Age (years)'),
        y=alt.Y('Stroke Risk (%):Q').scale(zero=False),
        tooltip=['Age', 'Stroke Risk (%)']
    ).properties(
        height=400,
        title="5-Year Stroke Risk Over Time"
    )

    # Add risk level bands
    risk_bands = pd.DataFrame({
        'Risk Band': ['Low (<2%)', 'Moderate (2-5%)', 'High (5-10%)', 'Very High (>10%)'],
        'Lower': [0, 2, 5, 10],
        'Upper': [2, 5, 10, 100]
    })

    st.altair_chart(risk_chart, width="stretch")

    st.divider()
    st.subheader("Clinical Factor Trajectories")

    weight_chart = alt.Chart(results_df).mark_line().encode(
        x=alt.X('Age:Q').scale(zero=False),
        y=alt.Y('Weight (kg):Q').scale(zero=False),
        color=alt.value('#1f77b4')
    ).properties(height=300, title="Weight Over Time")

    bmi_chart = alt.Chart(results_df).mark_line().encode(
        x=alt.X('Age:Q').scale(zero=False),
        y=alt.Y('BMI:Q').scale(zero=False),
        color=alt.value('#ff7f0e')
    ).properties(height=300, title="BMI Over Time")

    st.altair_chart(weight_chart, width="stretch")
    st.altair_chart(bmi_chart, width="stretch")

    sbp_chart = alt.Chart(results_df).mark_line().encode(
        x=alt.X('Age:Q').scale(zero=False),
        y=alt.Y('SBP (mmHg):Q').scale(zero=False),
        color=alt.value('#2ca02c')
    ).properties(height=300, title="Systolic BP Over Time")

    dbp_chart = alt.Chart(results_df).mark_line().encode(
        x=alt.X('Age:Q').scale(zero=False),
        y=alt.Y('DBP (mmHg):Q').scale(zero=False),
        color=alt.value('#d62728')
    ).properties(height=300, title="Diastolic BP Over Time")

    st.altair_chart(sbp_chart, width="stretch")
    st.altair_chart(dbp_chart, width="stretch")

    # Make another chart for diabetes
    diabetes_chart = alt.Chart(results_df).mark_line().encode(
        x=alt.X('Age:Q').scale(zero=False),
        y=alt.Y('Diabetes:Q').scale(zero=False),
        color=alt.value('#9467bd')
    ).properties(height=300, title="Diabetes Status Over Time")

    st.altair_chart(diabetes_chart, width="stretch")