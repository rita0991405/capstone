import pandas as pd
import numpy as np
import joblib
import json

# Korean metadata:
# HE_wt = weight (kg), HE_ht = height (cm), BMI = calculated
# EC_wht_23 = total work hours per week
# BE3_31 = walking days per week
# BE5_1  = strength training days per week
# BE3_72 = high-intensity work activity days per week
# BE3_82 = moderate-intensity work activity days per week
# BE3_76 = high-intensity leisure activity days per week
# BE3_86 = moderate-intensity leisure activity days per week
# BO3_01 = exercises for weight control: 0=no, 1=yes

# Thresholds for filtering irrelevant recommendations
HEALTHY_BMI           = 23.0   # BMI target (Asian population guideline)
MAX_ACTIVITY_DAYS     = 7
MAX_WORK_HOURS        = 40     # above this = overworked
MIN_ACTIVITY_TO_SKIP  = 5     # already doing >= 5 days → don't recommend more
MIN_WEIGHT_LOSS_EXCESS = 5    # must be >5kg above healthy weight to recommend loss
MIN_WORK_HOURS_EXCESS  = 10   # must work >10hrs above 40 to recommend reduction
MIN_RISK_REDUCTION     = 0.01 # ignore changes that reduce risk by less than
HABIT_THRESHOLD       = 3

# Load Japanese model assets globally
jp_models = joblib.load('jp_lr_models.pkl')
JP_MODEL = jp_models['full_model']
with open('jp_feature_columns.json', 'r') as f:
    JP_FEATURE_COLUMNS = json.load(f)['full_model']

# SCENARIO DEFINITIONS
MODIFIABLE_SCENARIOS = [
    # KOREAN FEATURES
    {'id': 'weight_loss_5kg', 'feature': 'HE_wt', 'delta': -5, 'source': 'kr', 'label': 'Lose 5kg', 'description': 'Reduce body weight by 5kg through diet and exercise.'},
    {'id': 'weight_loss_10kg', 'feature': 'HE_wt', 'delta': -10, 'source': 'kr', 'label': 'Lose 10kg', 'description': 'Reduce body weight by 10kg.'},
    {'id': 'more_walking_days', 'feature': 'BE3_31', 'delta': +2, 'source': 'kr', 'label': 'Walk 2 more days per week', 'description': 'Add 2 days of walking per week.'},
    {'id': 'add_strength_training', 'feature': 'BE5_1', 'delta': +2, 'source': 'kr', 'label': 'Add 2 days of strength training', 'description': 'Incorporate strength training 2 more days per week.'},
    {'id': 'reduce_work_hours', 'feature': 'EC_wht_23', 'delta': -10, 'source': 'kr', 'label': 'Work 10 fewer hours per week', 'description': 'Reduce total weekly working hours by 10.'},
    {'id': 'start_weight_control_exercise', 'feature': 'BO3_01', 'delta': +1, 'source': 'kr', 'label': 'Start exercising for weight control', 'description': 'Begin using exercise for weight management.'},
    
    # JAPANESE FEATURES
    {'id': 'quit_smoking', 'feature': 'smoking_index', 'target_val': 0, 'source': 'jp', 'label': 'Quit smoking', 'description': 'Eliminating tobacco use to improve metabolic health.'},
    {'id': 'improve_sleep_latency', 'feature': 'psqi_c2_sleep_latency', 'delta': -1, 'source': 'jp', 'label': 'Improve sleep onset', 'description': 'Reduce the time it takes to fall asleep.'},
    {'id': 'reduce_daytime_dysfunction', 'feature': 'psqi_c7_daytime_dysfunction', 'delta': -1, 'source': 'jp', 'label': 'Improve daytime alertness', 'description': 'Address habits causing daytime sleepiness.'}
]

def _is_applicable(scenario, feature, current_val, user_df):
    """
    Check whether this scenario makes practical sense for this specific user.
    """
    height_m = user_df['HE_ht'].values[0] / 100
    current_wt = user_df['HE_wt'].values[0]
    healthy_wt = HEALTHY_BMI * (height_m ** 2)
    excess_kg = current_wt - healthy_wt

    # ACTIVITY CHECKS
    # We define 3 days as the "habit threshold" 
    # If they do 3+ days of a specific activity, don't suggest adding more.
    HABIT_THRESHOLD = 3
    activity_features = [
        'BE3_31', 'BE5_1', 'BE3_72', 'BE3_82', 'BE3_76', 'BE3_86'
    ]
    
    if feature in activity_features:
        if current_val >= HABIT_THRESHOLD:
            return False

    # WEIGHT LOSS CHECKS
    if feature == 'HE_wt':
        if excess_kg < MIN_WEIGHT_LOSS_EXCESS:
            return False
        # Skip -10kg recommendation if they only have a small amount to lose
        if scenario['delta'] == -10 and excess_kg < 8:
            return False

    # WORK HOURS CHECKS
    if feature == 'EC_wht_23':
        if current_val <= MAX_WORK_HOURS + MIN_WORK_HOURS_EXCESS:
            return False

    # WEIGHT CONTROL EXERCISE (BO3_01)
    if feature == 'BO3_01':
        # Skip if already doing it OR if they are already at a healthy weight
        if current_val == 1 or current_wt <= healthy_wt:
            return False
        # Skip if they are already very active in other categories
        # (e.g., Strength Training or Vigorous Leisure >= 4 days)
        vigorous_days = user_df.get('BE3_76', pd.Series([0])).values[0]
        strength_days = user_df.get('BE5_1', pd.Series([0])).values[0]
        
        if vigorous_days >= 4 or strength_days >= 4:
            return False
        
    # Sleep/Smoking logic (JP)
    if scenario['source'] == 'jp':
        if 'target_val' in scenario and current_val <= scenario['target_val']: return False
        if current_val <= 0: return False

    return True


def _room_for_improvement(feature, current_val, user_df):
    """
    Returns 0-1: how much room does this user have to improve on this feature.
    """
    # Define shared physical benchmarks
    height_m = user_df['HE_ht'].values[0] / 100
    healthy_wt = HEALTHY_BMI * (height_m ** 2)

    # 1. Weight: Room is based on how much 'excess' they carry
    if feature == 'HE_wt':
        excess = max(0, current_val - healthy_wt)
        # Normalize: if they are 20kg over, room is 1.0 (capped)
        # You can adjust '20' based on what you consider a 'maximum' excess
        return min(1.0, excess / 20)

    # 2. Activity: Room is based on the gap to a full week (7 days)
    elif feature in ['BE3_31', 'BE5_1', 'BE3_72', 'BE3_82', 'BE3_76', 'BE3_86']:
        return (MAX_ACTIVITY_DAYS - current_val) / MAX_ACTIVITY_DAYS

    # 3. Work Hours: Room is based on hours worked over the 40hr baseline
    elif feature == 'EC_wht_23':
        excess_hours = max(0, current_val - MAX_WORK_HOURS)
        # Normalize: 20 hours of overtime is considered 'max' room (1.0)
        return min(1.0, excess_hours / 20)

    # 4. Weight Control Exercise: 
    elif feature == 'BO3_01':
        # If they already do it, no room to 'start'
        if current_val == 1:
            return 0.0
        # If they are already at a healthy weight, no room/need to improve this
        if user_df['HE_wt'].values[0] <= healthy_wt:
            return 0.0
        # Otherwise, give it a balanced weight (e.g., 0.5) 
        # so it doesn't always automatically win over weight loss.
        return 0.5

    return 0.5

def simulate_lifestyle_changes(user_df_kr, user_input_dict_jp, kr_pipeline, kr_baseline_risk):
    """
    Generate unified lifestyle recommendations by simulating changes across both 
    Korean and Japanese cohort models.
    Parameters:
    user_df_kr : pd.DataFrame
        A single-row DataFrame containing features formatted for the Korean model 
        (e.g., HE_ht, HE_wt, BE3_31).
    user_input_dict_jp : dict
        A dictionary containing raw user inputs for the Japanese model. Keys must 
        correspond to those in 'jp_feature_columns.json' (e.g., psqi_total_score, 
        smoking_index).
    kr_pipeline : sklearn.pipeline.Pipeline
        The fitted machine learning pipeline used for the Korean cohort risk prediction.
    kr_baseline_risk : float
        The initial risk score (0-1) calculated by the Korean model for the current user.

    Returns:
    list of dict
        A ranked list of recommendation objects. Each dict contains:
        - 'label' (str): The high-level action (e.g., "Lose 5kg").
        - 'description' (str): Detailed advice for the user.
        - 'risk_reduction' (float): Percentage point drop in risk.
        - 'priority_score' (float): The metric used for final ranking.
        - 'source' (str): Which cohort model generated the insight ("Activity (KR)" 
          or "Lifestyle (JP)").
    """
    recommendations = []

    # Prepare JP baseline if model exists
    jp_baseline_risk = None
    jp_df = None
    if JP_MODEL and JP_FEATURE_COLUMNS:
        jp_df = pd.DataFrame([user_input_dict_jp])[JP_FEATURE_COLUMNS]
        jp_baseline_risk = JP_MODEL.predict_proba(jp_df)[:, 1][0]

    for scenario in MODIFIABLE_SCENARIOS:
        source = scenario['source']
        feature = scenario['feature']

        # Skip if JP model is missing for a JP scenario
        if source == 'jp' and JP_MODEL is None: continue

        # Identify current value based on source
        if source == 'kr':
            if feature not in user_df_kr.columns: continue
            current_val = user_df_kr[feature].values[0]
        else:
            if feature not in jp_df.columns: continue
            current_val = jp_df[feature].values[0]

        if not _is_applicable(scenario, feature, current_val, user_df_kr): continue

        # Simulate change
        if source == 'kr':
            modified = user_df_kr.copy()
            new_val = max(0, current_val + scenario.get('delta', 0))
            if feature == 'BO3_01': new_val = 1
            modified[feature] = new_val
            if feature == 'HE_wt' and 'BMI' in modified.columns:
                modified['BMI'] = modified['HE_wt'] / (modified['HE_ht'] / 100) ** 2
            
            new_risk = kr_pipeline.predict_proba(modified)[:, 1][0]
            risk_reduction = kr_baseline_risk - new_risk
        else:
            modified = jp_df.copy()
            new_val = scenario.get('target_val', max(0, current_val + scenario.get('delta', 0)))
            modified[feature] = new_val
            new_risk = JP_MODEL.predict_proba(modified)[:, 1][0]
            risk_reduction = jp_baseline_risk - new_risk

        room = _room_for_improvement(feature, current_val, user_df_kr)
        priority_score = risk_reduction * room

        if risk_reduction > MIN_RISK_REDUCTION:
            room = _room_for_improvement(feature, current_val, user_df_kr)
            recommendations.append({
                'id':             scenario['id'],
                'label':          scenario['label'],
                'description':    scenario['description'],
                'current_value':  round(float(current_val), 2),
                'new_value':      round(float(new_val), 2),
                'new_risk':       round(float(new_risk * 100), 1),
                'risk_reduction': round(float(risk_reduction * 100), 2),
                'priority_score': round(float(priority_score * 100), 4),
                'message': (
                    f"If you {scenario['label'].lower()}, your predicted risk "
                    f"(a reduction of {risk_reduction*100:.1f} percentage points)."
                )
            })
    recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
    return recommendations
