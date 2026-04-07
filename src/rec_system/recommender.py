import pandas as pd
import numpy as np

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

# ── Thresholds for filtering irrelevant recommendations ───────────────────────
HEALTHY_BMI           = 23.0   # BMI target (Asian population guideline)
MAX_ACTIVITY_DAYS     = 7
MAX_WORK_HOURS        = 40     # above this = overworked
MIN_ACTIVITY_TO_SKIP  = 5     # already doing >= 5 days → don't recommend more
MIN_WEIGHT_LOSS_EXCESS = 5    # must be >5kg above healthy weight to recommend loss
MIN_WORK_HOURS_EXCESS  = 10   # must work >10hrs above 40 to recommend reduction
MIN_RISK_REDUCTION     = 0.01 # ignore changes that reduce risk by less than


def _is_applicable(scenario, feature, current_val, user_df):
    """
    Check whether this scenario makes practical sense for this specific user.
    """
    # --- 1. ALWAYS DEFINE SHARED VARIABLES AT THE TOP ---
    # This prevents the UnboundLocalError
    height_m = user_df['HE_ht'].values[0] / 100
    current_wt = user_df['HE_wt'].values[0]
    healthy_wt = HEALTHY_BMI * (height_m ** 2)
    excess_kg = current_wt - healthy_wt

    # --- 2. ACTIVITY CHECKS ---
    # We define 3 days as the "habit threshold" 
    # If they do 3+ days of a specific activity, don't suggest adding more.
    HABIT_THRESHOLD = 3
    activity_features = [
        'BE3_31', 'BE5_1', 'BE3_72', 'BE3_82', 'BE3_76', 'BE3_86'
    ]
    
    if feature in activity_features:
        if current_val >= HABIT_THRESHOLD:
            return False

    # --- 3. WEIGHT LOSS CHECKS ---
    if feature == 'HE_wt':
        if excess_kg < MIN_WEIGHT_LOSS_EXCESS:
            return False
        # Skip -10kg recommendation if they only have a small amount to lose
        if scenario['delta'] == -10 and excess_kg < 8:
            return False

    # --- 4. WORK HOURS CHECKS ---
    if feature == 'EC_wht_23':
        if current_val <= MAX_WORK_HOURS + MIN_WORK_HOURS_EXCESS:
            return False

    # --- 5. WEIGHT CONTROL EXERCISE (BO3_01) ---
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

MODIFIABLE_SCENARIOS = [
    {
        'id':          'weight_loss_5kg',
        'feature':     'HE_wt',
        'delta':       -5,
        'label':       'Lose 5kg',
        'description': 'Reduce your body weight by 5kg through diet and exercise.',
    },
    {
        'id':          'weight_loss_10kg',
        'feature':     'HE_wt',
        'delta':       -10,
        'label':       'Lose 10kg',
        'description': 'Reduce your body weight by 10kg.',
    },
    {
        'id':          'more_walking_days',
        'feature':     'BE3_31',
        'delta':       +2,
        'label':       'Walk 2 more days per week',
        'description': 'Add 2 days of walking per week for physical activity.',
    },
    {
        'id':          'add_strength_training',
        'feature':     'BE5_1',
        'delta':       +2,
        'label':       'Add 2 days of strength training per week',
        'description': 'Incorporate strength training 2 additional days per week.',
    },
    {
        'id':          'more_leisure_moderate',
        'feature':     'BE3_86',
        'delta':       +2,
        'label':       'Add 2 days of moderate leisure activity per week',
        'description': 'Engage in moderate-intensity leisure activities (e.g. cycling, swimming) 2 more days per week.',
    },
    {
        'id':          'more_leisure_vigorous',
        'feature':     'BE3_76',
        'delta':       +2,
        'label':       'Add 2 days of vigorous leisure activity per week',
        'description': 'Engage in high-intensity leisure activities (e.g. running, sports) 2 more days per week.',
    },
    {
        'id':          'reduce_work_hours',
        'feature':     'EC_wht_23',
        'delta':       -10,
        'label':       'Work 10 fewer hours per week',
        'description': 'Reduce your total weekly working hours by 10.',
    },
    {
        'id':          'start_weight_control_exercise',
        'feature':     'BO3_01',
        'delta':       +1,
        'label':       'Start exercising for weight control',
        'description': 'Begin using exercise as an intentional strategy for weight management.',
    },
]


def simulate_lifestyle_changes(user_df, pipeline, baseline_risk):
    """
    Generate personalized lifestyle recommendations.

    Filters out recommendations that don't apply to this user
    (already active, already healthy weight, not overworked etc.)
    then ranks remaining recommendations by personalized priority.

    Parameters
    ----------
    user_df       : pd.DataFrame — prepared user input (1 row)
    pipeline      : fitted sklearn Pipeline
    baseline_risk : float        — current risk score (0-1)

    Returns
    -------
    list of recommendation dicts, sorted by personalized priority
    """
    recommendations = []

    for scenario in MODIFIABLE_SCENARIOS:
        feature = scenario['feature']

        if feature not in user_df.columns:
            continue

        current_val = user_df[feature].values[0]

        # ── Filter: is this recommendation applicable for this user? ──────────
        if not _is_applicable(scenario, feature, current_val, user_df):
            continue

        # Calculate new value
        if feature in ['HE_wt', 'EC_wht_23']:
            new_val = max(0, current_val + scenario['delta'])
        else:
            new_val = float(np.clip(current_val + scenario['delta'], 0, 7))

        if new_val == current_val:
            continue

        # Simulate change
        modified = user_df.copy()
        modified[feature] = new_val
        if feature == 'HE_wt' and 'BMI' in modified.columns:
            modified['BMI'] = modified['HE_wt'] / (modified['HE_ht'] / 100) ** 2

        new_risk       = pipeline.predict_proba(modified)[:, 1][0]
        risk_reduction = baseline_risk - new_risk

        # Filter: skip if risk reduction is negligible
        if risk_reduction < MIN_RISK_REDUCTION:
            continue

        # Personalized priority
        room = _room_for_improvement(feature, current_val, user_df)
        priority_score = risk_reduction * room

        recommendations.append({
            'id':             scenario['id'],
            'label':          scenario['label'],
            'description':    scenario['description'],
            'current_value':  round(float(current_val), 2),
            'new_value':      round(float(new_val), 2),
            'baseline_risk':  round(float(baseline_risk * 100), 1),
            'new_risk':       round(float(new_risk * 100), 1),
            'risk_reduction': round(float(risk_reduction * 100), 2),
            'priority_score': round(float(priority_score * 100), 4),
            'message': (
                f"If you {scenario['label'].lower()}, your predicted risk "
                f"could drop from {baseline_risk*100:.1f}% to {new_risk*100:.1f}% "
                f"(a reduction of {risk_reduction*100:.1f} percentage points)."
            )
        })

    recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
    return recommendations