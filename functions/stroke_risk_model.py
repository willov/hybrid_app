import json
import numpy as np

class StrokeRiskEnsembleModel:
    """
    Python implementation of the R ensemble age model for stroke risk prediction.
    Replicates the functionality from ensemble_age_model.R
    """
    
    def __init__(self):
        """Initialize with the migration data containing all model coefficients and scaling parameters"""
        with open("./functions/python_migration_data_with_scaling.json", 'r') as f:
            self.models = json.load(f)

        # Absolute risk for control groups
        self.absolute_risk_control = {
            "Age Model <50": 0.002879261,
            "Age Model 50-59": 0.009622399,
            "Age Model 60-69": 0.02332245,
            "Age Model 70+": 0.04977592
        }

        # Age ranges for weighting
        self.age_ranges = {
            "Age Model <50": (25, 49),
            "Age Model 50-59": (50, 59),
            "Age Model 60-69": (60, 69),
            "Age Model 70+": (70, 96)
        }
    
    def _ensure_array(self, value):
        """Convert single values or lists to numpy arrays"""
        if np.isscalar(value):
            return np.array([value])
        return np.array(value)
    
    def _broadcast_to_length(self, value, target_length):
        """Broadcast a value to match target length"""
        value_array = self._ensure_array(value)
        if len(value_array) == 1:
            return np.repeat(value_array[0], target_length)
        elif len(value_array) == target_length:
            return value_array
        else:
            raise ValueError(f"Value length {len(value_array)} doesn't match target length {target_length}")
    
    def scale_variable(self, values, var_name: str, model_name: str):
        """
        Apply R's scale(x, center=TRUE, scale=TRUE) transformation to arrays.
        This is equivalent to: (values - mean) / std
        """
        model_key = list(self.models[model_name].keys())[0]
        model_data = self.models[model_name][model_key]
        
        values_array = self._ensure_array(values)
        
        if 'scaling_parameters' in model_data and var_name in model_data['scaling_parameters']:
            scaling = model_data['scaling_parameters'][var_name]
            return (values_array - scaling['mean']) / scaling['std']
        else:
            # Fallback: assume value is already scaled or use identity
            print(f"Warning: No scaling parameters found for {var_name} in {model_name}")
            return values_array
    
    def prepare_features_batch(self, patient_data: dict[str, int | float | list], model_name: str) -> dict[str, np.ndarray]:
        """
        Convert raw patient data to model features with proper scaling for batch processing.
        
        Expected patient_data format:
        {
            'SEX': int or list[int] (1=male, 2=female),
            'AGE': int/float or list[int/float],
            'BMI': float or list[float],
            'CPD': float or list[float] (cigarettes per day),
            'DBP': float or list[float] (diastolic blood pressure),
            'SBP': float or list[float] (systolic blood pressure),
            'DMRX': int or list[int] (diabetes flag),
            'AF_beforestroke': int or list[int] (atrial fibrillation flag)
        }
        """
        # Determine the batch size from the first array-like input
        batch_size = 1
        for key, value in patient_data.items():
            if not np.isscalar(value):
                batch_size = len(value)
                break
        
        features = {}
        
        # Handle categorical variables (R factor encoding)
        sex_array = self._broadcast_to_length(patient_data['SEX'], batch_size)
        features['factor(SEX)2'] = (sex_array == 2).astype(float)
        
        af_array = self._broadcast_to_length(patient_data['AF_beforestroke'], batch_size)
        features['factor(AF_beforestroke)1'] = af_array.astype(float)
        
        dmrx_array = self._broadcast_to_length(patient_data['DMRX'], batch_size)
        features['factor(DMRX)1'] = dmrx_array.astype(float)
        
        # Handle continuous variables with scaling
        sbp_array = self._broadcast_to_length(patient_data['SBP'], batch_size)
        features['scale(SBP, center = TRUE, scale = TRUE)'] = self.scale_variable(
            sbp_array, 'SBP', model_name
        )
        
        bmi_array = self._broadcast_to_length(patient_data['BMI'], batch_size)
        features['scale(BMI, center = TRUE, scale = TRUE)'] = self.scale_variable(
            bmi_array, 'BMI', model_name
        )
        
        dbp_array = self._broadcast_to_length(patient_data['DBP'], batch_size)
        features['scale(DBP, center = TRUE, scale = TRUE)'] = self.scale_variable(
            dbp_array, 'DBP', model_name
        )
        
        # CPD is not scaled in R model
        cpd_array = self._broadcast_to_length(patient_data['CPD'], batch_size)
        features['CPD'] = cpd_array.astype(float)
        
        return features, batch_size
    
    def predict_single_model_batch(self, patient_data: dict[str, int | float | list], model_name: str) -> dict[str, np.ndarray]:
        """
        Make prediction using a single age-specific logistic regression model for batch data.
        Returns log-odds, probability, and odds ratio arrays.
        """
        model_key = list(self.models[model_name].keys())[0]
        model = self.models[model_name][model_key]
        
        # Prepare features with proper scaling
        features, batch_size = self.prepare_features_batch(patient_data, model_name)
        
        # Calculate log-odds: β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
        log_odds = np.full(batch_size, model['intercept'])
        
        coeffs = model['all_coefficients']
        for feature_name, feature_values in features.items():
            if feature_name in coeffs:
                log_odds += coeffs[feature_name] * feature_values
        
        # Convert to probability using logistic function: 1 / (1 + e^(-log_odds))
        probability = 1.0 / (1.0 + np.exp(-log_odds))
        
        # Calculate odds ratio: e^(log_odds)
        odds_ratio = np.exp(log_odds)
        
        return {
            'log_odds': log_odds,
            'probability': probability,
            'odds_ratio': odds_ratio
        }
    
    def calculate_distance_weights_batch(self, ages) -> dict[str, np.ndarray]:
        """
        Calculate distance-based weights for ensemble for batch of ages.
        """
        ages_array = self._ensure_array(ages)
        
        d1 = np.sqrt((ages_array - 49)**2 + (ages_array - 25)**2)  # <50 model
        d2 = np.sqrt((ages_array - 59)**2 + (ages_array - 50)**2)  # 50-59 model
        d3 = np.sqrt((ages_array - 69)**2 + (ages_array - 60)**2)  # 60-69 model
        d4 = np.sqrt((ages_array - 96)**2 + (ages_array - 70)**2)  # 70+ model
        
        # Calculate weights (inverse distance weighting)
        total_denominator = d2*d3*d4 + d1*d3*d4 + d1*d2*d4 + d1*d2*d3
        
        weights = {
            "Age Model <50": (d2 * d3 * d4) / total_denominator,
            "Age Model 50-59": (d1 * d3 * d4) / total_denominator,
            "Age Model 60-69": (d1 * d2 * d4) / total_denominator,
            "Age Model 70+": (d1 * d2 * d3) / total_denominator
        }
        
        return weights
    
    def predict_ensemble(self, patient_data: dict[str, int | float | list]) -> dict[str, np.ndarray | dict]:
        """
        Make ensemble prediction combining all age-specific models for batch data.
        Replicates the exact logic from ensemble_age_model.R
        
        Returns numpy arrays for all predictions to match your simulation usage.
        """
        ages = patient_data['AGE']
        weights = self.calculate_distance_weights_batch(ages)
        
        # Determine batch size
        batch_size = len(self._ensure_array(ages))
        
        # Initialize weighted sums
        weighted_odds_ratio = np.zeros(batch_size)
        weighted_relative_risk = np.zeros(batch_size)
        weighted_absolute_risk = np.zeros(batch_size)
        
        # Get predictions from each model and combine with weights
        individual_predictions = {}
        
        for model_name in self.models.keys():
            # Get prediction from this age-specific model
            prediction = self.predict_single_model_batch(patient_data, model_name)
            individual_predictions[model_name] = prediction
            
            odds_ratio = prediction['odds_ratio']
            weight = weights[model_name]
            control_risk = self.absolute_risk_control[model_name]
            
            # Weighted odds ratio (from R script)
            weighted_odds_ratio += odds_ratio * weight
            
            # Convert odds ratio to relative risk using baseline control risk
            # Formula from R: OR / (1 - baseline_risk + (baseline_risk * OR))
            relative_risk = odds_ratio / (1 - control_risk + (control_risk * odds_ratio))
            weighted_relative_risk += relative_risk * weight
            
            # Convert relative risk to absolute risk
            # Formula from R: relative_risk * baseline_risk
            absolute_risk = relative_risk * control_risk
            weighted_absolute_risk += absolute_risk * weight
        
        return {
            'ensemble_odds_ratio': weighted_odds_ratio,
            'ensemble_relative_risk': weighted_relative_risk,
            'ensemble_absolute_risk': weighted_absolute_risk,
            'age_weights': weights,
            'individual_predictions': individual_predictions
        }
