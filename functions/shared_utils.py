"""
Shared utility functions for the hybrid app pages.
Centralizes common functionality like model loading, simulation, and helper functions.
"""

import os
import subprocess
import sys
import numpy as np
import pandas as pd


def setup_custom_packages():
    """
    Ensures custom_package directory exists and installs sund if needed.
    Adds custom_package to sys.path for imports.
    """
    os.makedirs('./custom_package', exist_ok=True)
    
    if "sund" not in os.listdir('./custom_package'):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--target=./custom_package", 'sund<=3.0']
        )
    
    if './custom_package' not in sys.path:
        sys.path.append('./custom_package')

def setup_model(model_name):
    """
    Load a model by name from the models directory.
    
    Args:
        model_name: Name of the model file (without .txt extension)
    
    Returns:
        Tuple of (model, feature_names)
    """
    setup_custom_packages()
    import sund

    sund.install_model(f"./models/{model_name}.txt")
    model = sund.load_model(model_name)
    features = model.feature_names
    return model, features


def flatten(nested_list):
    """Flatten a nested list of lists."""
    return [item for sublist in nested_list for item in sublist]


def simulate_insres_weight(m, anthropometrics, stim, t_start_sim):
    """
    Simulate insulin resistance and weight change model (Page 1).
    
    Args:
        m: SUND model object
        anthropometrics: Dict with age, sex, height, weight, fat mass, lean mass, etc.
        stim: Dict with stimulus inputs (EIchange, meal_kcal, ss_x, etc.)
        t_start_sim: Start time for simulation
    
    Returns:
        Tuple of (sim_results DataFrame, new_initial_conditions)
    """
    import copy
    
    setup_custom_packages()
    import sund

    act = sund.Activity(time_unit='d')
    
    for key, val in stim.items():
        act.add_output(
            name=key, type="piecewise_constant",
            t=val["t"], f=val["f"]
        )
    
    for key, val in anthropometrics.items():
        act.add_output(name=key, type="constant", f=val)
    
    sim = sund.Simulation(models=m, activities=act, time_unit='d')
    
    # Getting initial values
    initial_conditions = copy.deepcopy(m.state_values)
    
    # Set initial conditions using state names and index logic
    state_mapping = {
        'Gly': anthropometrics['Ginit'],
        'ECF': anthropometrics['ECFinit'],
        'F': anthropometrics['Finit'],
        'L': anthropometrics['Linit']
    }
    
    for state_name, value in state_mapping.items():
        if state_name in m.state_names:
            idx = m.state_names.index(state_name)
            initial_conditions[idx] = value
    
    # simulate
    sim.simulate(
        time_vector=np.linspace(
            min(stim["ss_x"]["t"]), max(stim["ss_x"]["t"]), 10000
        ),
        state_values=initial_conditions
    )
    
    sim_results = pd.DataFrame(sim.feature_values, columns=sim.feature_names)
    sim_results.insert(0, 'Time', sim.time_vector)
    
    sim_diet_results = sim_results[(sim_results['Time'] >= t_start_sim)]
    new_initial_conditions = sim.state_values
    
    return sim_diet_results, new_initial_conditions


def simulate_meal(m, anthropometrics, stim, inits, t_start_sim):
    """
    Simulate a meal response for insulin resistance model (Page 1).
    
    Args:
        m: SUND model object
        anthropometrics: Dict with anthropometric parameters
        stim: Dict with stimulus inputs for meal
        inits: Initial conditions from previous simulation
        t_start_sim: Start time for simulation
    
    Returns:
        sim_meal_results DataFrame with time in minutes
    """
    setup_custom_packages()
    import sund

    act = sund.Activity(time_unit='d')
    
    for key, val in stim.items():
        act.add_output(
            name=key, type="piecewise_constant",
            t=val["t"], f=val["f"]
        )
    
    for key, val in anthropometrics.items():
        act.add_output(name=key, type="constant", f=val)
    
    sim = sund.Simulation(models=m, activities=act, time_unit='d')
    
    sim.simulate(
        time_vector=np.linspace(
            min(stim["ss_x"]["t"]), max(stim["ss_x"]["t"]), 10000
        ),
        state_values=inits
    )
    
    sim_results = pd.DataFrame(sim.feature_values, columns=sim.feature_names)
    sim_results.insert(0, 'Time', sim.time_vector)
    sim_meal_results = sim_results[(sim_results['Time'] >= t_start_sim)]
    sim_meal_results['Time'] = sim_meal_results['Time'] * 24.0 * 60.0
    
    return sim_meal_results


def simulate_bp(m, stim, anthropometrics, initials):
    """
    Simulate blood pressure change model (Page 2).
    
    Args:
        m: SUND model object
        stim: Dict with stimulus inputs (drug_on, etc.)
        anthropometrics: Dict with IC_SBP, IC_DBP
        initials: Initial state [SBP, DBP]
    
    Returns:
        sim_results DataFrame
    """
    setup_custom_packages()
    import sund

    act = sund.Activity(time_unit='y')
    
    for key, val in stim.items():
        act.add_output(
            name=key, type="piecewise_constant",
            t=val["t"], f=val["f"]
        )
    
    for key, val in anthropometrics.items():
        act.add_output(name=key, type="constant", f=val)
    
    sim = sund.Simulation(models=m, activities=act, time_unit='y')
    
    t_start = min(stim["drug_on"]["t"])
    
    sim.simulate(
        time_vector=np.linspace(t_start, max(stim["drug_on"]["t"]), 10000),
        state_values=initials
    )
    
    sim_results = pd.DataFrame(sim.feature_values, columns=sim.feature_names)
    sim_results.insert(0, 'Time', sim.time_vector)
    
    return sim_results


def extract_bp_from_table(SBP0, DBP0, start_time):
    """
    Extract blood pressure group information from lookup table.
    Used in Page 2 to determine initial conditions based on age and BP.
    
    Args:
        SBP0: Systolic blood pressure
        DBP0: Diastolic blood pressure
        start_time: Age at start of simulation
    
    Returns:
        Tuple of (IC_SBP, IC_DBP)
    """
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
    IC_SBPdata = v[0, :]
    dataage = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
    
    mindiff, chosenAgeIndex = min((abs(age - start_time), idx) for idx, age in enumerate(dataage))
    chosenAge = dataage[chosenAgeIndex]
    dataSBP = v[chosenAgeIndex, :]
    mindiff, chosenColumn = min((abs(sbp - SBP0), idx) for idx, sbp in enumerate(dataSBP))
    
    IC_DBP = IC_DBPdata[chosenColumn]
    IC_SBP = IC_SBPdata[chosenColumn]
    
    return IC_SBP, IC_DBP
