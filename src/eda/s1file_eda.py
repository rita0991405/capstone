import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load data
df = pd.read_csv('S1File.csv')

col_map = {
    '登録番号': 'registration_id',
    '施設': 'facility',
    '患者識別番号': 'patient_id',
    'DM Type\n1:1型\n2:2型\n3:その他\n4:IGT\n5:正常耐糖能': 'dm_type',
    '年齢': 'age',
    '性別\n1：男\n2：女': 'sex',
    '推定罹病期間\n(年)': 'illness_duration_yrs',
    '身長': 'height_cm',
    '体重': 'weight_kg',
    'BMI': 'bmi',
    '喫煙本数': 'cigarettes_per_day',
    '喫煙　年': 'smoking_years',
    '喫煙指数': 'smoking_index',
    '１:過去\n2:今': 'smoking_status',
    '飲酒　\n0:なし  \n1：有り': 'alcohol',
    '神経症　\n0:なし  \n1：有り': 'neuropathy',
    '網膜症\n0:NDR \n１：SDR\n2:PPDR\n3:PDR': 'retinopathy',
    '腎症　　　\n0：～1期 \n１：2期  \n 2：3期 　\n 3：4期　\n 4：5期': 'nephropathy',
    '足病変　\n0:なし\n1：有り': 'foot_lesion',
    '高血圧\n0:なし\n1：有り': 'hypertension',
    '高脂血症　　　0:なし \n1：有り': 'hyperlipidemia',
    '脳梗塞\n0:なし \n1：有り': 'cerebral_infarction',
    '心血管\n0:なし \n1：有り': 'cardiovascular',
    '高尿酸血症': 'hyperuricemia',
    '精神疾患': 'psychiatric_disorder',
    'HbA1c（0） (NGSP)': 'hba1c',
    'PSQI\nC1': 'psqi_c1_quality',
    'PSQI\nC2': 'psqi_c2_latency',
    'PSQI\nC3': 'psqi_c3_duration',
    'PSQI\nC4': 'psqi_c4_efficiency',
    'PSQI\nC5': 'psqi_c5_disturbances',
    'PSQI\nC6': 'psqi_c6_medication',
    'PSQI\nC7': 'psqi_c7_daytime',
    'PSQI\n得点': 'psqi_total',
    '睡眠時間': 'sleep_duration_hrs',
    '睡眠効率（％）': 'sleep_efficiency_pct',
    '入眠時間（分）': 'sleep_onset_latency_min',
}
df = df.rename(columns=col_map)

# Derived labels
dm_map = {1.0: 'Type 1', 2.0: 'Type 2', 3.0: 'Other', 4.0: 'IGT', 5.0: 'Normal'}
df['dm_label'] = df['dm_type'].map(dm_map)
df['sex_label'] = df['sex'].map({1.0: 'Male', 2.0: 'Female'})
df['alcohol_label'] = df['alcohol'].map({0.0: 'Non-drinker', 1.0: 'Drinker'})
df['smoking_label'] = df['smoking_status'].map({1.0: 'Past smoker', 2.0: 'Current smoker'})

# Focus subset: Type 2 + Normal + IGT (most relevant for diabetes risk)
df_focus = df[df['dm_type'].isin([2.0, 4.0, 5.0])].copy()
df_focus['risk_label'] = df_focus['dm_type'].map({2.0: 'Type 2 DM', 4.0: 'IGT', 5.0: 'Normal'})

palette_dm   = {'Type 2 DM': '#E05C5C', 'IGT': '#F5A623', 'Normal': '#4A90D9'}
palette_alc  = {'Non-drinker': '#4A90D9', 'Drinker': '#E05C5C'}
palette_smk  = {'Past smoker': '#4A90D9', 'Current smoker': '#E05C5C'}

TITLE_FS = 17
LABEL_FS = 14
TICK_FS  = 13
LEG_FS   = 13

os.makedirs('eda_plots_s1', exist_ok=True)

# Plot 1: DM Type distribution
fig, ax = plt.subplots(figsize=(9, 6), facecolor='#F8F9FA')
order = ['Normal', 'IGT', 'Type 1', 'Type 2', 'Other']
counts = df['dm_label'].value_counts().reindex(order)
colors = ['#4A90D9', '#F5A623', '#9B59B6', '#E05C5C', '#95A5A6']
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:,}', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_title('Diabetes Type Distribution (SOREKA, Japan)', fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_ylabel('Count', fontsize=LABEL_FS)
ax.set_xlabel('DM Type', fontsize=LABEL_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots_s1/01_dm_type_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 1")

# Plot 2: Age distribution by DM group
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
for label, grp in df_focus.groupby('risk_label'):
    ax.hist(grp['age'].dropna(), bins=25, alpha=0.6, label=label,
            color=palette_dm[label], edgecolor='white')
ax.set_title('Age Distribution by Diabetes Status', fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_xlabel('Age (years)', fontsize=LABEL_FS)
ax.set_ylabel('Count', fontsize=LABEL_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.legend(fontsize=LEG_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots_s1/02_age_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 2")

# Plot 3: BMI distribution by DM group
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F8F9FA')
bmi_clean = df_focus[df_focus['bmi'] < 60]   # remove extreme outlier (331)
for label, grp in bmi_clean.groupby('risk_label'):
    ax.hist(grp['bmi'].dropna(), bins=30, alpha=0.6, label=label,
            color=palette_dm[label], edgecolor='white')
ax.axvline(25, color='gray', linestyle='--', linewidth=1.5, label='Overweight (25)')
ax.axvline(30, color='black', linestyle=':', linewidth=1.5, label='Obese (30)')
ax.set_title('BMI Distribution by Diabetes Status\n(BMI > 60 excluded as outlier)',
             fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_xlabel('BMI (kg/m²)', fontsize=LABEL_FS)
ax.set_ylabel('Count', fontsize=LABEL_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.legend(fontsize=LEG_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots_s1/03_bmi_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 3")

# Plot 4: Sleep duration by DM group
fig, ax = plt.subplots(figsize=(9, 6), facecolor='#F8F9FA')
sns.boxplot(data=df_focus.dropna(subset=['sleep_duration_hrs']),
            x='risk_label', y='sleep_duration_hrs',
            hue='risk_label', palette=palette_dm, ax=ax,
            order=['Normal', 'IGT', 'Type 2 DM'],
            linewidth=1.8, width=0.5, legend=False)
ax.axhline(7, color='gray', linestyle='--', linewidth=1.5, label='Recommended minimum (7h)')
ax.set_title('Sleep Duration by Diabetes Status', fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_xlabel('')
ax.set_ylabel('Sleep Duration (hours)', fontsize=LABEL_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.legend(fontsize=LEG_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots_s1/04_sleep_duration_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 4")

# Plot 5: Sleep efficiency by DM group
fig, ax = plt.subplots(figsize=(9, 6), facecolor='#F8F9FA')
eff_clean = df_focus[df_focus['sleep_efficiency_pct'] <= 100]
sns.boxplot(data=eff_clean.dropna(subset=['sleep_efficiency_pct']),
            x='risk_label', y='sleep_efficiency_pct',
            hue='risk_label', palette=palette_dm, ax=ax,
            order=['Normal', 'IGT', 'Type 2 DM'],
            linewidth=1.8, width=0.5, legend=False)
ax.axhline(85, color='gray', linestyle='--', linewidth=1.5, label='Clinical threshold (85%)')
ax.set_title('Sleep Efficiency by Diabetes Status\n(Values > 100% excluded)',
             fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_xlabel('')
ax.set_ylabel('Sleep Efficiency (%)', fontsize=LABEL_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.legend(fontsize=LEG_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots_s1/05_sleep_efficiency_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 5")

# Plot 6: Sleep onset latency by DM group
fig, ax = plt.subplots(figsize=(9, 6), facecolor='#F8F9FA')
lat_clean = df_focus[df_focus['sleep_onset_latency_min'] <= 120]
sns.boxplot(data=lat_clean.dropna(subset=['sleep_onset_latency_min']),
            x='risk_label', y='sleep_onset_latency_min',
            hue='risk_label', palette=palette_dm, ax=ax,
            order=['Normal', 'IGT', 'Type 2 DM'],
            linewidth=1.8, width=0.5, legend=False)
ax.axhline(30, color='gray', linestyle='--', linewidth=1.5, label='Clinical threshold (30 min)')
ax.set_title('Sleep Onset Latency by Diabetes Status\n(Values > 120 min excluded)',
             fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_xlabel('')
ax.set_ylabel('Sleep Onset Latency (minutes)', fontsize=LABEL_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.legend(fontsize=LEG_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots_s1/06_sleep_onset_latency_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 6")

# Plot 7: Smoking status by DM group
fig, ax = plt.subplots(figsize=(9, 6), facecolor='#F8F9FA')
smk = df_focus.dropna(subset=['smoking_status'])
smk_grp = smk.groupby(['risk_label', 'smoking_label']).size().unstack(fill_value=0)
smk_pct = smk_grp.div(smk_grp.sum(axis=1), axis=0) * 100
smk_pct.reindex(['Normal', 'IGT', 'Type 2 DM']).plot(
    kind='bar', ax=ax, color=[palette_smk['Past smoker'], palette_smk['Current smoker']],
    edgecolor='white', linewidth=1.2, width=0.55)
ax.set_title('Smoking Status by Diabetes Group', fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_ylabel('Percentage (%)', fontsize=LABEL_FS)
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=TICK_FS)
ax.tick_params(axis='y', labelsize=TICK_FS)
ax.legend(fontsize=LEG_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots_s1/07_smoking_status.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 7")

# Plot 8: Alcohol by DM group
fig, ax = plt.subplots(figsize=(9, 6), facecolor='#F8F9FA')
alc = df_focus.dropna(subset=['alcohol'])
alc_grp = alc.groupby(['risk_label', 'alcohol_label']).size().unstack(fill_value=0)
alc_pct = alc_grp.div(alc_grp.sum(axis=1), axis=0) * 100
alc_pct.reindex(['Normal', 'IGT', 'Type 2 DM']).plot(
    kind='bar', ax=ax, color=[palette_alc['Non-drinker'], palette_alc['Drinker']],
    edgecolor='white', linewidth=1.2, width=0.55)
ax.set_title('Alcohol Consumption by Diabetes Group', fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_ylabel('Percentage (%)', fontsize=LABEL_FS)
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=TICK_FS)
ax.tick_params(axis='y', labelsize=TICK_FS)
ax.legend(fontsize=LEG_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots_s1/08_alcohol_consumption.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 8")

# Plot 9: PSQI total score by DM group
fig, ax = plt.subplots(figsize=(9, 6), facecolor='#F8F9FA')
sns.boxplot(data=df_focus.dropna(subset=['psqi_total']),
            x='risk_label', y='psqi_total',
            hue='risk_label', palette=palette_dm, ax=ax,
            order=['Normal', 'IGT', 'Type 2 DM'],
            linewidth=1.8, width=0.5, legend=False)
ax.axhline(5, color='gray', linestyle='--', linewidth=1.5,
           label='Poor sleep threshold (PSQI > 5)')
ax.set_title('PSQI Total Score by Diabetes Status\n(Higher = Worse Sleep Quality)',
             fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_xlabel('')
ax.set_ylabel('PSQI Total Score', fontsize=LABEL_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.legend(fontsize=LEG_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots_s1/09_psqi_total.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 9")

# Plot 10: Correlation heatmap (key features)
fig, ax = plt.subplots(figsize=(12, 10), facecolor='#F8F9FA')
corr_cols = ['age', 'bmi', 'sleep_duration_hrs', 'sleep_efficiency_pct',
             'sleep_onset_latency_min', 'psqi_total', 'hba1c', 'dm_type']
corr_labels = ['Age', 'BMI', 'Sleep Duration', 'Sleep Efficiency',
               'Sleep Onset Latency', 'PSQI Total', 'HbA1c', 'DM Type']
corr_df = df_focus[corr_cols].copy()
corr_df.columns = corr_labels
corr = corr_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, linewidths=0.6, annot_kws={'size': 13},
            cbar_kws={'label': 'Pearson r', 'shrink': 0.7})
ax.set_title('Feature Correlation Matrix (Type 2 DM, IGT & Normal)',
             fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.tick_params(axis='x', labelsize=TICK_FS, rotation=35)
ax.tick_params(axis='y', labelsize=TICK_FS, rotation=0)
plt.tight_layout()
plt.savefig('eda_plots_s1/10_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 10")

print("\nAll 10 plots saved to eda_plots_s1/")
