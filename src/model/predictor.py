import pandas as pd
import numpy as np
import pickle
import json

def load_model():
    with open('saved_model/lr_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open('saved_model/feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    return pipeline, feature_columns

def prepare_user_input(user_input: dict, feature_columns: list) -> pd.DataFrame:
    """
    Convert raw user input into the exact format the model expects.

    Parameters
    ----------
    user_input : dict
        Raw values from the user's form submission.
        Example:
        {
            'sex': 1,           # 0=female, 1=male
            'age': 45,          # years (19-60)
            'HE_wt': 80,        # weight in kg
            'HE_ht': 170,       # height in cm
            'EC_stt_1': 1,      # employment type:
                                #   0=unemployed, 1=salaried worker,
                                #   2=self-employed (no employees),
                                #   3=self-employed (with employees),
                                #   4=unpaid family worker
            'EC_occp': 3,       # occupation:
                                #   0=unemployed, 1=manager, 2=professional,
                                #   3=office worker, 4=service worker,
                                #   5=sales worker, 6=agricultural worker,
                                #   7=skilled worker, 8=assembly worker,
                                #   9=laborer, 10=soldier
            'EC_wht_23': 50,    # total work hours per week (0 if unemployed)
            'BE3_31': 5,        # walking days per week (0-7)
            'BE5_1': 2,         # strength training days per week (0-7)
            'BE3_72': 0,        # high-intensity work activity days per week (0-7)
            'BE3_82': 0,        # moderate-intensity work activity days per week (0-7)
            'BE3_76': 0,        # high-intensity leisure activity days per week (0-7)
            'BE3_86': 0,        # moderate-intensity leisure activity days per week (0-7)
            'BO3_01': 1,        # exercises for weight control: 0=no, 1=yes
        }

    Returns
    -------
    pd.DataFrame with exactly the columns the model was trained on
    """
    df = pd.DataFrame([user_input])

    # Calculate BMI from height and weight
    df['BMI'] = df['HE_wt'] / (df['HE_ht'] / 100) ** 2

    # One-hot encode occupation — must match training encoding exactly
    # drop_first=True means EC_occp_1 (manager) is the reference category
    df = pd.get_dummies(df, columns=['EC_occp'], drop_first=True)

    # Add any missing occupation columns with 0
    # e.g. if user is occupation 3, all other EC_occp_X columns must exist as 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to exactly match training order
    df = df[feature_columns]

    return df


def calculate_risk_score(user_input: dict) -> dict:
    """
    Main function: takes user input, returns risk score and interpretation.

    Returns
    -------
    dict with:
        - risk_score    : float (0-1 probability of prediabetes)
        - risk_percent  : str   (e.g. "73.2%")
        - risk_level    : str   ("Low", "Moderate", "High")
        - interpretation: str   (plain language explanation)
        - user_df       : pd.DataFrame (prepared input, passed to explainer)
        - pipeline      : fitted Pipeline (passed to explainer and recommender)
    """
    pipeline, feature_columns = load_model()
    user_df = prepare_user_input(user_input, feature_columns)

    # predict_proba[:, 1] = probability of class 1 (prediabetes)
    risk_score = pipeline.predict_proba(user_df)[:, 1][0]

    # Risk level thresholds
    # Note: these are relative to the Korean working adult population (age 19-60)
    # not absolute clinical thresholds
    baseline_prevalence = 6059 / 16137  # = 0.375 # prevalence in our dataset
    if risk_score < baseline_prevalence * 0.8:
        risk_level = "Low"
        interpretation = (
            "Your lifestyle profile is similar to people without prediabetes "
            "in our dataset. Keep maintaining your current habits."
        )
    elif risk_score < baseline_prevalence * 1.5:
        risk_level = "Moderate"
        interpretation = (
            "Your profile shows some risk factors associated with prediabetes. "
            "Consider the lifestyle changes below."
        )
    else:
        risk_level = "High"
        interpretation = (
            "Your profile is similar to people with prediabetes in our dataset. "
            "We strongly recommend consulting a healthcare provider and reviewing "
            "the lifestyle changes below."
        )

    return {
        'risk_score':     round(float(risk_score), 4),
        'risk_percent':   f"{risk_score * 100:.1f}%",
        'risk_level':     risk_level,
        'interpretation': interpretation,
        'user_df':        user_df,
        'pipeline':       pipeline,
    }

