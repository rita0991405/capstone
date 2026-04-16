from altair import value
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import json
import shap

from predictor import load_model, prepare_user_input, calculate_risk_score
from recommender import simulate_lifestyle_changes

st.config.set_option('theme.primaryColor', '#0165fc')

st.title('East Asians Type 2 Diabetes Risk Prediction')
st.showSidebarNavigation = False
st.caption('This page was designed to calculate Type 2 diabetes risk for East Asians. '
'The risk score is based on a machine learning model trained on data from Korea and Japan and is intended to provide recommendations for lifestyle changes that can reduce the risk of developing Type 2 diabetes. '
'The model is not a diagnostic tool and should not be used as a substitute for professional medical advice. '
'Any inputted data is not stored and is only used to calculate the risk score and recommendations for the current session. '
'Each variable is weighted differently based on its impact on diabetes risk, and the recommendations are prioritized based on the potential risk reduction they offer.')
st.divider()

# ============
# RISK MODEL
# ============

def get_bar_color(value):
    if value <= 0:    return '#cccccc'
    if value < 0.10:  return '#48A860'
    if value < 0.20:  return '#E78263'
    if value < 0.30:  return '#C0392B'
    return '#7B0D1E'

def score_to_color(s):
    s = s * 100
    if s <= 25:  return '#48A860'
    if s <= 50:  return '#E78263'
    if s <= 75:  return '#C0392B'
    return '#7B0D1E'

# ============
# USER INPUTS
# ============

st.subheader('Demographics')
age = st.slider('Age', min_value=18, max_value=60, value=35, step=1)
height = st.slider('Height (cm)', min_value=150, max_value=200, value=175, step=1)
weight = st.slider('Weight (kg)', min_value=40, max_value=150, value=75, step=1)

sex_label = st.radio('Sex', ['Male', 'Female'], horizontal=True)
sex = 1 if sex_label == 'Male' else 2

st.subheader('Work & General Activity')
occupation_options = {
    'Manager': 1, 'Professional / Technical': 2, 'Office / Clerical worker': 3,
    'Service worker': 4, 'Sales worker': 5, 'Agricultural / Fishing worker': 6,
    'Skilled / Craft worker': 7, 'Assembly / Machine operator': 8,
    'Laborer': 9, 'Military': 10, 'Not applicable or unemployed': 0,
}
occupation_label = st.selectbox('Occupation', list(occupation_options.keys()), index=2)
occupation = occupation_options[occupation_label]
work_hours = st.slider('Total work hours per week', min_value=0, max_value=120, value=40, step=1)

st.subheader('Lifestyle')
walking_days = st.slider('How many days do you walk per week?', 0, 7, 1)
strength_days = st.slider('How many days do you strength train per week?', 0, 7, 1)
high_leisure_days = st.slider('How many days do you engage in vigorous physical activity in your free time per week?', 0, 7, 1)
mod_leisure_days = st.slider('How many days do you engage in moderate physical activity in your free time per week?', 0, 7, 1)
high_work_days = st.slider('How many days do you engage in vigorous physical activity for work per week?', 0, 7, 1)
mod_work_days = st.slider('How many days do you engage in moderate physical activity for work per week?', 0, 7, 1)
exercise_for_weight_label = st.radio('Do you exercise for weight loss or maintenance?', ['No', 'Yes'], horizontal=True)
exercise_for_weight = 1 if exercise_for_weight_label == 'Yes' else 0

st.subheader('Sleep')
sleep_hours = st.slider('How many hours do you sleep per night?', min_value=2.0, max_value=15.0, value=7.0, step=0.5)
sleep_efficiency = st.slider('How frequently does it take you 30+ min. to fall asleep AND/OR how often do you wake up 2+ times throughout the night?',
                                 min_value=0, max_value=100, value=30, step=10, format='%d%%')

cigs = 0
smoking_years = 0
smoking_index = 0
st.subheader('Smoking')
is_smoker = st.radio("Do you currently smoke?", ["No", "Yes"])

if is_smoker == "Yes":
    cigs = st.slider("How many cigarettes do you smoke per day?", min_value=1, max_value=20, value=10, step=1)
    smoking_years = st.slider("How many years have you been smoking?", min_value=0, max_value=30, value=5, step=5)
    smoking_index = max(1, cigs * smoking_years)

st.divider()
calculate_button = st.button('Calculate Risk', use_container_width=True, type='primary')

if calculate_button:

    user_input_kr = {
        'age': age,
        'sex': sex,
        'HE_ht': height,
        'HE_wt': weight,
        'EC_stt_1': 1,
        'EC_occp': occupation,
        'EC_wht_23': work_hours,
        'BE3_31': walking_days,
        'BE5_1': strength_days,
        'BE3_72': high_work_days,
        'BE3_82': mod_work_days,
        'BE3_76': high_leisure_days,
        'BE3_86': mod_leisure_days,
        'BO3_01': exercise_for_weight,
    }

    user_input_jp = {
        'age': age,
        'sex': sex,
        'height': height,
        'weight': weight,
        'bmi': weight / (height/100)**2,
        'cigarettes_per_day': cigs,
        'smoking_years': smoking_years,
        'smoking_index': smoking_index,
        'alcohol': 0,
        'hypertension': 0,
        'hyperlipidemia': 0,
        'psqi_c1_sleep_quality': 1,
        'psqi_c2_sleep_latency': 1 if sleep_efficiency < 30 else 2,
        'psqi_c3_sleep_duration': 1 if 6 <= sleep_hours <= 8 else 2,
        'psqi_c4_sleep_efficiency': 1,
        'psqi_c5_sleep_disturbance': 1,
        'psqi_c6_sleep_medication': 0,
        'psqi_c7_daytime_dysfunction': 1 if sleep_hours < 5 else 0,
        'psqi_total_score': 5,
        'sleep_duration_hrs': sleep_hours,
        'sleep_efficiency_pct': 100 - sleep_efficiency,
        'sleep_latency_min': 15 if sleep_efficiency < 30 else 45
    }

    result = calculate_risk_score(user_input_kr)
    current_risk_score = result['risk_score']
    user_df_kr = result['user_df']
    pipeline_kr = result['pipeline']

    recommendations = simulate_lifestyle_changes(
        user_df_kr,
        user_input_jp,
        pipeline_kr,
        current_risk_score
    )

    current_risk_level = result['risk_level']
    risk_percent = result['risk_percent']
    interpretation = result['interpretation']

    reduced_risk_score = recommendations[0]['new_risk'] / 100 if recommendations else None
    improvable_factors = [(r['label'], r['risk_reduction']) for r in recommendations]

    # ============
    #  VISUALS
    # ============

    st.divider()

    # --- Visual 1: Risk Score ---
    st.subheader('Risk Score')
    score = round(current_risk_score * 100, 1)

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=score,
        number={'font': {'size': 52}},
        gauge={
            'axis': {'range': [0, 100], 'tickvals': [0, 25, 50, 75, 100]},
            'bar': {'color': '#32333f', 'thickness': 0.02},
            'steps': [
                {'range': [0,  25],  'color': '#48A860'},
                {'range': [25, 50],  'color': '#E78263'},
                {'range': [50, 75],  'color': '#C0392B'},
                {'range': [75, 100], 'color': '#7B0D1E'},
            ],
            'threshold': {
                'line': {'color': '#32333f', 'width': 1},
                'thickness': 0.5,
                'value': score,
            },
            'bgcolor': 'white',
            'borderwidth': 0,
        },
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<br><br><br>', unsafe_allow_html=True)

    # --- Visual 2: Biggest Influencing Factors ---
    st.subheader('Biggest Factors Influencing Risk')

    pipeline_obj, feature_columns = load_model()
    scaler = pipeline_obj.named_steps['scaler']
    model  = pipeline_obj.named_steps['model']

    user_df_aligned = user_df_kr[feature_columns].fillna(0)
    user_scaled = scaler.transform(user_df_aligned)
    background = np.tile(user_scaled, (50, 1))

    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(user_scaled)
    vals = shap_values[0]

    FEATURE_LABELS = {
        'age':        'Age',
        'sex':        'Sex',
        'HE_ht':      'Height',
        'HE_wt':      'Weight',
        'EC_stt_1':   'Smoking Status',
        'EC_occp':    'Occupation',
        'EC_occp_0':  'Occupation: Unemployed',
        'EC_occp_1':  'Occupation: Manager',
        'EC_occp_2':  'Occupation: Professional/Technical',
        'EC_occp_3':  'Occupation: Office/Clerical',
        'EC_occp_4':  'Occupation: Service worker',
        'EC_occp_5':  'Occupation: Sales worker',
        'EC_occp_6':  'Occupation: Agricultural/Fishing',
        'EC_occp_7':  'Occupation: Skilled/Craft worker',
        'EC_occp_8':  'Occupation: Assembly/Machine operator',
        'EC_occp_9':  'Occupation: Laborer',
        'EC_occp_10': 'Occupation: Soldier',
        'EC_wht_23':  'Weekly Work Hours',
        'BE3_31':     'Walking Days/Week',
        'BE5_1':      'Strength Training Days/Week',
        'BE3_72':     'Vigorous Work Activity Days/Week',
        'BE3_82':     'Moderate Work Activity Days/Week',
        'BE3_76':     'Vigorous Leisure Activity Days/Week',
        'BE3_86':     'Moderate Leisure Activity Days/Week',
        'BO3_01':     'Exercises for Weight Management',
    }

    shap_series = (pd.Series(dict(zip(feature_columns, vals)))
        .sort_values(key=abs, ascending=False)
        .head(10)
        .dropna()
    )

    total_abs = shap_series.abs().sum()
    pct_series = (shap_series.abs() / total_abs * 100).sort_values(ascending=True)

    factor_names  = [FEATURE_LABELS.get(f, f) for f in pct_series.index]
    factor_values = list(pct_series.values)
    bar_colors    = [get_bar_color(v / 100) for v in factor_values]
    bar_labels    = [f'{v:.1f}%' for v in factor_values]

    fig = go.Figure(go.Bar(
        x=factor_values,
        y=factor_names,
        orientation='h',
        marker_color=bar_colors,
        text=bar_labels,
        textposition='outside',
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=30, b=10, l=10, r=50),
        xaxis_title='',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', gridwidth=1),
        showlegend=False,
        hovermode=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<br>', unsafe_allow_html=True)

    # --- Visual 3: Risk Reduction ---
    st.subheader('Risk Reduction by Change')

    if not improvable_factors:
        st.success('Nothing to improve!')
    else:
        wf_labels = ['Current Score'] + [f for f, _ in improvable_factors] + ['Projected Score']
        wf_values = [current_risk_score] + [-v/100 for _, v in improvable_factors] + [reduced_risk_score]
        wf_types = ['absolute'] + ['relative'] * len(improvable_factors) + ['total']
        wf_text = [risk_percent] + [f'-{v:.1f}' for _, v in improvable_factors] + [f'{reduced_risk_score * 100:.1f}%' if reduced_risk_score is not None else '—']

        fig = go.Figure(go.Waterfall(
            measure=wf_types,
            x=wf_labels,
            y=wf_values,
            text=wf_text,
            textposition='outside',
            decreasing={'marker': {'color': '#48A860'}},
            totals={'marker': {'color': '#0165fc'}},
            connector={'visible': False},
        ))

        fig.update_layout(
            height=280,
            margin=dict(t=30, b=10, l=20, r=20),
            yaxis_title='Risk Score',
            showlegend=False,
            hovermode=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.divider()

    # --- Key Recommendation ---
    st.subheader('Key Recommendation')

    if not recommendations:
        st.success('Everything looks good. Keep it up!')
    else:
        for idx, rec in enumerate(recommendations[:1]):
            priority = rec.get('priority', 'normal')
            st.markdown(f"""
            <div class='rec-card {'critical' if priority=='critical' else 'high' if priority=='high' else ''}'>
            <div class='rec-title' style='font-size:1.3rem; font-weight:500;'>{rec.get('title', rec['label'])}</div>
            <div style='display:flex; gap:8px; align-items:baseline;'>
                <span class='rec-detail'>{rec.get('detail', rec['description'])}</span>
                <span class='rec-reduction'>This reduces your risk score by {rec['risk_reduction']:.1f}%</span>
            </div>
            </div>
            <br>
            """, unsafe_allow_html=True)