import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import shap

# Load data
df = pd.read_csv('Prediabetes.csv')

# Drop constant column (everyone's working)
df = df.drop(columns=['EC1_1']) # employment status

# Add BMI column
df['BMI'] = df['HE_wt'] / (df['HE_ht'] / 100) ** 2

# Define features and target
X = df.drop(columns=['diabetes'])
y = df['diabetes']

# One-hot encode occupation (nominal)
X = pd.get_dummies(X, columns=['EC_occp'], drop_first=True)

# updated: Save feature column names
# must use the exact same columns when predicting for new users
feature_columns = X.columns.tolist()
with open('saved_model/feature_columns.json', 'w') as f:
    json.dump(feature_columns, f)
print("Feature columns saved:", feature_columns)

# Carve out a holdout set FIRST before any modeling
# This is never touched until final evaluation
X_temp, X_holdout, y_temp, y_holdout = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Use Pipelines so scaling is fit per fold, not globally
# This prevents scaler leakage inside cross-validation
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(
        scale_pos_weight=len(y_temp[y_temp==0]) / len(y_temp[y_temp==1]),
        random_state=42,
        eval_metric='logloss'
    ))
])

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42))
])

# Cross-validation instead of single split
# Mean ± std across folds is more honest than a single 80/20 result
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print(" Cross-Validation Results ")
for name, pipeline in [('Logistic Regression', lr_pipeline), ('XGBoost', xgb_pipeline), ('Random Forest', rf_pipeline)]:
    scores = cross_validate(
        pipeline, X_temp, y_temp, cv=cv,
        scoring=['f1_macro', 'roc_auc', 'recall_macro', 'precision_macro'],
        return_train_score=True
    )
    print(f"\n  {name} ")
    print(f"ROC-AUC : {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}")
    print(f"F1      : {scores['test_f1_macro'].mean():.4f} ± {scores['test_f1_macro'].std():.4f}")
    print(f"Recall  : {scores['test_recall_macro'].mean():.4f} ± {scores['test_recall_macro'].std():.4f}")
    # If train score >> test score, model is overfitting
    print(f"Train ROC-AUC: {scores['train_roc_auc'].mean():.4f} (vs test — check for overfitting)")

# Final evaluation on holdout — run this ONCE at the end
print("\n Final Holdout Evaluation (run once) ")
# Refit best model (LR based on previous results) on full X_temp
lr_pipeline.fit(X_temp, y_temp)
preds = lr_pipeline.predict(X_holdout)
probs = lr_pipeline.predict_proba(X_holdout)[:, 1]
print(classification_report(y_holdout, preds))
print(f"ROC-AUC: {roc_auc_score(y_holdout, probs):.4f}")

# SHAP fitted on training data only
# Refit scaler on X_temp manually just for SHAP (pipeline doesn't expose internals easily)
scaler = StandardScaler()
X_temp_np = scaler.fit_transform(X_temp.to_numpy().astype(float))
X_holdout_np = scaler.transform(X_holdout.to_numpy().astype(float))

lr_final = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_final.fit(X_temp_np, y_temp)

lr_explainer = shap.LinearExplainer(lr_final, X_temp_np)
lr_shap = lr_explainer.shap_values(X_holdout_np)
shap.summary_plot(lr_shap, X_holdout_np, feature_names=X_temp.columns.tolist())

#### Conclusion ########
###Logistic Regression: best model, no overfitting
###XGBoost: severely overfitting
###Random Forest: completely overfitting
### final holdout ROC-AUC of 0.7485 almost exactly matches the CV mean of 0.7481 