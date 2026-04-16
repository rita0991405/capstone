import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt

# Read in data
df = pd.read_csv('S1File.csv')

# Map column names for translation
column_mapping = {
    '登録番号': 'registration_number',
    '施設': 'facility',
    '患者識別番号': 'patient_id',
    'DM Type\n1:1型\n2:2型\n3:その他\n4:IGT\n5:正常耐糖能': 'dm_type',
    '年齢': 'age',
    '性別\n1：男\n2：女': 'sex',
    '推定罹病期間\n(年)': 'illness_duration_years',
    '身長': 'height',
    '体重': 'weight',
    'BMI': 'bmi',
    '喫煙本数': 'cigarettes_per_day',
    '喫煙\u3000年': 'smoking_years',
    '喫煙指数': 'smoking_index',
    '１:過去\n2:今': 'smoking_status',
    '飲酒\u3000\n0:なし  \n1：有り': 'alcohol',
    '神経症\u3000\n0:なし  \n1：有り': 'neuropathy',
    '網膜症\n0:NDR \n１：SDR\n2:PPDR\n3:PDR': 'retinopathy',
    '腎症\u3000\u3000\u3000\n0：～1期 \n１：2期  \n 2：3期 \u3000\n 3：4期\u3000\n 4：5期': 'nephropathy',
    '足病変\u3000\n0:なし\n1：有り': 'foot_lesion',
    '高血圧\n0:なし\n1：有り': 'hypertension',
    '高脂血症\u3000\u3000\u30000:なし \n1：有り': 'hyperlipidemia',
    '脳梗塞\n0:なし \n1：有り': 'cerebral_infarction',
    '心血管\n0:なし \n1：有り': 'cardiovascular_disease',
    '高尿酸血症': 'hyperuricemia',
    '精神疾患': 'psychiatric_disorder',
    'TP（０）': 'total_protein',
    'Alb（０）': 'albumin',
    'TG(0)': 'triglycerides',
    'TC(0)': 'total_cholesterol',
    'HDL(0）': 'hdl_cholesterol',
    'LDL(0)': 'ldl_cholesterol',
    'LDL(F式）(0)': 'ldl_cholesterol_formula',
    'nonHDL(0)': 'non_hdl_cholesterol',
    'HbA1c（0） (NGSP)': 'hba1c',
    'GA': 'glycated_albumin',
    'AST（0）': 'ast',
    'ALT（0）': 'alt',
    'gGTP（0）': 'ggtp',
    'BUN（０）': 'bun',
    'Cr（0）': 'creatinine',
    'eGFR（0）': 'egfr',
    'UA（０）': 'uric_acid',
    'Na（０）': 'sodium',
    'K（０）': 'potassium',
    'Cl（０）': 'chloride',
    'Ca（０）': 'calcium',
    'P（０）': 'phosphorus',
    '随時血糖': 'random_blood_glucose',
    'ACR\n(mg/gcre)': 'acr',
    'PAID\n総計': 'paid_total',
    'PAID\n得点': 'paid_score',
    'PSQI\nC1': 'psqi_c1_sleep_quality',
    'PSQI\nC2': 'psqi_c2_sleep_latency',
    'PSQI\nC3': 'psqi_c3_sleep_duration',
    'PSQI\nC4': 'psqi_c4_sleep_efficiency',
    'PSQI\nC5': 'psqi_c5_sleep_disturbance',
    'PSQI\nC6': 'psqi_c6_sleep_medication',
    'PSQI\nC7': 'psqi_c7_daytime_dysfunction',
    'PSQI\n得点': 'psqi_total_score',
    '睡眠時間': 'sleep_duration_hrs',
    '睡眠効率（％）': 'sleep_efficiency_pct',
    '入眠時間（分）': 'sleep_latency_min',
    'DTRQOL\n1\n社会活動/日常活動への負担': 'dtrqol_social_activity',
    'DTRQOL\n2\n治療への不安と不満': 'dtrqol_treatment_anxiety',
    'DTRQOL\n3\n低血糖': 'dtrqol_hypoglycemia',
    'DTRQOL\n4\n治療満足度': 'dtrqol_treatment_satisfaction',
    'DTRQOL\n総スコア': 'dtrqol_total_score',
    'CPR食前': 'cpr_preprandial',
    'CPR食後': 'cpr_postprandial',
    'IRI食前': 'iri_preprandial',
    'IRI食後': 'iri_postprandial',
    'BS食前': 'bs_preprandial',
    'BS食後': 'bs_postprandial',
    'Unnamed: 73': 'unnamed_col',
    'CPRindex': 'cpr_index',
    'HOMA-IR': 'homa_ir'
}

df = df.rename(columns=column_mapping)
df = df[df['dm_type'].isin([2, 5])].copy()
print("After filter:", df['dm_type'].value_counts())  # should show only 4 and 5
print("Total rows:", len(df))

# Drop leakage and ID columns
leakage_cols = [
    'registration_number', 'facility', 'patient_id', 
    'hba1c', #remove the stronggest indicator
    'illness_duration_years',                          # post-diagnosis
    'neuropathy', 'retinopathy', 'nephropathy',        # diabetes complications
    'foot_lesion', 'cerebral_infarction', 'cardiovascular_disease',
    'glycated_albumin', 'random_blood_glucose',  # defines the label
    'bs_preprandial', 'bs_postprandial',               # blood sugar measurements
    'homa_ir', 'cpr_index',                            # insulin resistance
    'unnamed_col'                                      # 100% missing
]
df = df.drop(columns=leakage_cols)

# Keep pre-diagnosis features
keep_cols = [
    'age', 'sex', 'height', 'weight', 'bmi', 
    'cigarettes_per_day', 'smoking_years', 'smoking_index', 'smoking_status',
    'alcohol', 'hypertension', 'hyperlipidemia',
    'psqi_c1_sleep_quality', 'psqi_c2_sleep_latency', 'psqi_c3_sleep_duration',
    'psqi_c4_sleep_efficiency', 'psqi_c5_sleep_disturbance',
    'psqi_c6_sleep_medication', 'psqi_c7_daytime_dysfunction',
    'psqi_total_score', 'sleep_duration_hrs', 'sleep_efficiency_pct',
    'sleep_latency_min'
]

X = df[keep_cols].copy()
y = (df['dm_type'] == 2).astype(int)  # 1=T2D, 0=Normal
print(y.value_counts())

# Drop columns with >40% missing, impute the rest
X = X.loc[:, X.isnull().mean() < 0.4]
X = X.fillna(X.median(numeric_only=True))

# Pipeline + cross-validation (10-fold given small sample)
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(
    lr_pipeline, X, y, cv=cv,
    scoring=['f1_macro', 'roc_auc', 'recall_macro'],
    return_train_score=True
)

print(f"\nROC-AUC : {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}")
print(f"F1      : {scores['test_f1_macro'].mean():.4f} ± {scores['test_f1_macro'].std():.4f}")
print(f"Recall  : {scores['test_recall_macro'].mean():.4f} ± {scores['test_recall_macro'].std():.4f}")
print(f"Train ROC-AUC: {scores['train_roc_auc'].mean():.4f}")

# Add SHAP visualization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit pipeline
lr_pipeline.fit(X_train, y_train)

# Extract scaled arrays for SHAP
scaler = lr_pipeline.named_steps['scaler']
X_train_np = scaler.transform(X_train).astype(float)
X_test_np = scaler.transform(X_test).astype(float)

# SHAP
explainer = shap.LinearExplainer(lr_pipeline.named_steps['model'], X_train_np)
shap_values = explainer.shap_values(X_test_np)
shap.summary_plot(shap_values, X_test_np, feature_names=X.columns.tolist(), show = False)
plt.title("SHAP beeswarm - S1 file full model")
plt.show()

### Sleep-only model ####
sleep_cols = [
    'psqi_c1_sleep_quality', 'psqi_c2_sleep_latency',
    'psqi_c3_sleep_duration', 'psqi_c4_sleep_efficiency',
    'psqi_c5_sleep_disturbance', 'psqi_c6_sleep_medication',
    'psqi_c7_daytime_dysfunction', 'psqi_total_score',
    'sleep_duration_hrs', 'sleep_efficiency_pct', 'sleep_latency_min'
]

X_sleep = df[sleep_cols].fillna(df[sleep_cols].median())

scores_sleep = cross_validate(
    lr_pipeline, X_sleep, y, cv=cv,
    scoring=['f1_macro', 'roc_auc', 'recall_macro'],
    return_train_score=True
)

print("\n Sleep-Only Model ")
print(f"ROC-AUC : {scores_sleep['test_roc_auc'].mean():.4f} ± {scores_sleep['test_roc_auc'].std():.4f}")
print(f"F1      : {scores_sleep['test_f1_macro'].mean():.4f} ± {scores_sleep['test_f1_macro'].std():.4f}")
print(f"Recall  : {scores_sleep['test_recall_macro'].mean():.4f} ± {scores_sleep['test_recall_macro'].std():.4f}")
print(f"Train ROC-AUC: {scores_sleep['train_roc_auc'].mean():.4f}")

## Demographics-only model###

demo_only_cols = [
    'age', 'sex', 'height', 'weight', 'bmi',
    'alcohol', 'hypertension', 'hyperlipidemia',
    'hyperuricemia', 'smoking_index'
]

X_demo = df[demo_only_cols].fillna(df[demo_only_cols].median())

scores_demo = cross_validate(
    lr_pipeline, X_demo, y, cv=cv,
    scoring=['f1_macro', 'roc_auc', 'recall_macro'],
    return_train_score=True
)

print("\n Demographics-Only Model ")
print(f"ROC-AUC : {scores_demo['test_roc_auc'].mean():.4f} ± {scores_demo['test_roc_auc'].std():.4f}")
print(f"F1      : {scores_demo['test_f1_macro'].mean():.4f} ± {scores_demo['test_f1_macro'].std():.4f}")
print(f"Recall  : {scores_demo['test_recall_macro'].mean():.4f} ± {scores_demo['test_recall_macro'].std():.4f}")
print(f"Train ROC-AUC: {scores_demo['train_roc_auc'].mean():.4f}")

print("\n Comparison ")
print(f"Full model        : {scores['test_roc_auc'].mean():.4f}")
print(f"Demographics only : {scores_demo['test_roc_auc'].mean():.4f}")
print(f"Sleep only        : {scores_sleep['test_roc_auc'].mean():.4f}")
print(f"Sleep added value : {scores['test_roc_auc'].mean() - scores_demo['test_roc_auc'].mean():.4f}")

#SHAP
scaler_shap = StandardScaler()
X_train_np  = scaler_shap.fit_transform(X_train.to_numpy().astype(float))
X_test_np   = scaler_shap.transform(X_test.to_numpy().astype(float))

lr_shap_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_shap_model.fit(X_train_np, y_train)

explainer   = shap.LinearExplainer(lr_shap_model, X_train_np)
shap_values = explainer.shap_values(X_test_np)

# Get PSQI correlation
import pandas as pd
from scipy import stats

df_jp = pd.read_excel('S1File.xlsx', sheet_name='T2DM(PSQI、A1cすべて○3294人)')

df_jp.columns = [
    'age', 'gender', 'duration', 'bmi', 'alcohol',
    'neuropathy', 'retinopathy', 'nephropathy',
    'hypertension', 'hyperlipidemia', 'macroangiopathy',
    'psqi_c1', 'psqi_c2', 'psqi_c3', 'psqi_c4', 'hba1c',
    'psqi_c5', 'psqi_c6', 'psqi_c7', 'psqi_total',
    'sleep_duration', 'sleep_efficiency', 'sleep_latency', 'smoker'
]

target = 'hba1c'

psqi_cols = [
    'psqi_c1', 'psqi_c2', 'psqi_c3', 'psqi_c4',
    'psqi_c5', 'psqi_c6', 'psqi_c7', 'psqi_total',
    'sleep_duration', 'sleep_efficiency', 'sleep_latency'
]

print(f"{'Feature':<25} {'n':<7} {'r':>7}  {'p':<12} Sig")
print("-" * 58)
for col in psqi_cols:
    df_clean = df_jp[[col, target]].dropna()  # ← both columns
    r, p = stats.pearsonr(df_clean[col], df_clean[target])  # ← two arrays
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    print(f"{col:<25} {len(df_clean):<7} {r:>+.4f}  {p:<12.6f} {sig}")


sleep_cols = ['sleep_duration', 'sleep_efficiency', 'sleep_latency']

print(f"{'Stat':<12}", end='')
for col in sleep_cols:
    print(f"{col:>20}", end='')
print()

for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
    print(f"{stat:<12}", end='')
    for col in sleep_cols:
        val = df_jp[col].describe()[stat]
        print(f"{val:>20.4f}", end='')
    print()

print()
print("Correlations with HbA1c:")
print(f"{'Feature':<25} {'n':<7} {'r':>7}  {'p':<12} Sig")
print("-" * 58)
for col in sleep_cols:
    df_clean = df_jp[[col, target]].dropna()  # ← both columns
    r, p = stats.pearsonr(df_clean[col], df_clean[target])  # ← two arrays
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    print(f"{col:<25} {len(df_clean):<7} {r:>+.4f}  {p:<12.6f} {sig}")
