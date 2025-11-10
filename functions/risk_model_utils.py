"""
Utility functions for stroke risk model and data processing.
"""

import numpy as np
import pandas as pd


def prepare_risk_features(sim_results, sbp_col='SBP', dbp_col='DBP', 
                          bmi_col='BMI', weight_col='Weight (kg)', 
                          age_col='Age', smoking=False, diabetes_col=None):
    """
    Prepare features for stroke risk calculation from simulation results.
    
    Args:
        sim_results: DataFrame with simulation results
        sbp_col: Column name for systolic blood pressure
        dbp_col: Column name for diastolic blood pressure
        bmi_col: Column name for BMI (optional)
        weight_col: Column name for weight (optional, used to calculate BMI if needed)
        age_col: Column name for age
        smoking: Boolean or column name for smoking status
        diabetes_col: Column name for diabetes status (optional)
    
    Returns:
        Dict with feature arrays for risk calculation
    """
    features = {
        'age': sim_results[age_col].values,
        'sbp': sim_results[sbp_col].values if sbp_col in sim_results.columns else None,
        'dbp': sim_results[dbp_col].values if dbp_col in sim_results.columns else None,
    }
    
    # Add BMI if available
    if bmi_col in sim_results.columns:
        features['bmi'] = sim_results[bmi_col].values
    elif weight_col in sim_results.columns and 'height' in sim_results.columns:
        # Calculate BMI from weight and height
        height = sim_results['height'].values if 'height' in sim_results.columns else 1.75
        features['bmi'] = sim_results[weight_col].values / (height ** 2)
    
    # Add smoking status
    if isinstance(smoking, bool):
        features['smoking'] = np.ones(len(sim_results)) * smoking if smoking else np.zeros(len(sim_results))
    elif smoking in sim_results.columns:
        features['smoking'] = sim_results[smoking].values
    
    # Add diabetes status if available
    if diabetes_col and diabetes_col in sim_results.columns:
        features['diabetes'] = sim_results[diabetes_col].values
    
    # Remove None values
    features = {k: v for k, v in features.items() if v is not None}
    
    return features


def extract_risk_features_from_simulation(sim_results_dict, time_vector=None):
    """
    Extract risk model features from a complete simulation results dictionary.
    Handles both weight/metabolic and blood pressure simulations.
    
    Args:
        sim_results_dict: Dict with feature names as keys and arrays as values
        time_vector: Time points (optional, for alignment)
    
    Returns:
        Dict formatted for stroke risk calculation
    """
    features = {}
    
    # Map possible column names to standard feature names
    feature_mapping = {
        'age': ['Age', 'age', 'AGE'],
        'sbp': ['SBP', 'Systolic', 'SBP (mmHg)', 'sbp'],
        'dbp': ['DBP', 'Diastolic', 'DBP (mmHg)', 'dbp'],
        'bmi': ['BMI', 'bmi', 'BMI (kg/m^2)'],
        'weight': ['Weight', 'Weight (kg)', 'weight'],
        'diabetes': ['Diabetes', 'diabetes', 'DMRX', 'T2D'],
        'height': ['Height', 'height'],
    }
    
    # Try to find features by looking for column names
    for feature_name, possible_names in feature_mapping.items():
        for possible_name in possible_names:
            if possible_name in sim_results_dict:
                features[feature_name] = sim_results_dict[possible_name]
                break
    
    return features


def aggregate_simulations_for_risk(weight_sim_results, bp_sim_results, time_vector=None):
    """
    Combine weight/metabolic simulation and BP simulation results into 
    a single feature set for risk calculation.
    
    Args:
        weight_sim_results: DataFrame from weight simulation (Page 1)
        bp_sim_results: DataFrame from BP simulation (Page 2)
        time_vector: Optional common time vector for alignment
    
    Returns:
        Combined DataFrame with all risk features
    """
    # Handle case where time vectors might be different
    # For now, assume they're aligned or we take one as reference
    
    if time_vector is None and len(weight_sim_results) > 0:
        time_vector = weight_sim_results['Time'].values
    
    # Start with weight simulation data
    combined = weight_sim_results.copy()
    
    # Add BP data if available and aligned
    if len(bp_sim_results) > 0:
        bp_cols = [col for col in bp_sim_results.columns if col != 'Time']
        
        # Try to align on time
        if 'Time' in bp_sim_results.columns and 'Time' in combined.columns:
            # Merge on time (assuming same time vector)
            combined = pd.merge_asof(
                combined.sort_values('Time'),
                bp_sim_results[['Time'] + bp_cols].sort_values('Time'),
                on='Time',
                direction='nearest'
            )
        else:
            # Just concatenate if no time alignment possible
            for col in bp_cols:
                if col not in combined.columns:
                    combined[col] = bp_sim_results[col].values[:len(combined)]
    
    return combined


def format_risk_output(time_vector, risk_trajectory, time_unit='years'):
    """
    Format risk calculation output for visualization.
    
    Args:
        time_vector: Array of time points
        risk_trajectory: Array of risk values
        time_unit: Unit of time ('years', 'months', 'days')
    
    Returns:
        DataFrame with time and risk columns
    """
    output_df = pd.DataFrame({
        'Time': time_vector,
        'Stroke Risk (%)': risk_trajectory,
        'Time_Unit': time_unit
    })
    
    # Add risk categories for interpretation
    def get_risk_category(risk):
        if risk < 2:
            return 'Low'
        elif risk < 5:
            return 'Moderate'
        elif risk < 10:
            return 'High'
        else:
            return 'Very High'
    
    output_df['Risk Category'] = output_df['Stroke Risk (%)'].apply(get_risk_category)
    
    return output_df
