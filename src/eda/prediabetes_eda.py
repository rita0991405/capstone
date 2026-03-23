import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load data
df = pd.read_csv('Prediabetes.csv')

# Data info

print(df.info())
print(df.describe())

# Label maps
df['sex_label']      = df['sex'].map({1: 'Male', 0: 'Female'})
df['diabetes_label'] = df['diabetes'].map({0: 'Normal', 1: 'Prediabetes'})
df['EC_occp_label']  = df['EC_occp'].map({
    0: 'Unemployed', 1: 'Manager', 2: 'Professional', 3: 'Office Worker',
    4: 'Service', 5: 'Sales', 6: 'Agricultural', 7: 'Skilled',
    8: 'Assembly', 9: 'Laborer', 10: 'Soldier'
})
df['BO3_01_label'] = df['BO3_01'].map({
    1: 'Exercises for Weight Control',
    0: 'Does Not Exercise for Weight Control'
})
df['BMI'] = df['HE_wt'] / (df['HE_ht'] / 100) ** 2

palette   = {'Normal': '#4A90D9', 'Prediabetes': '#E05C5C'}
TITLE_FS  = 18
LABEL_FS  = 14
TICK_FS   = 13
LEGEND_FS = 13

os.makedirs('eda_plots', exist_ok=True)

# Plot 1: Target distribution
fig, ax = plt.subplots(figsize=(8, 6), facecolor='#F8F9FA')
counts = df['diabetes_label'].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[palette['Normal'], palette['Prediabetes']],
              edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150,
            f'{val:,}\n({val / len(df) * 100:.1f}%)',
            ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.set_title('Target Distribution', fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_ylabel('Count', fontsize=LABEL_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.set_ylim(0, 14000)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots/01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 1")

# Plot 2: Age distribution
fig, ax = plt.subplots(figsize=(9, 6), facecolor='#F8F9FA')
for label, grp in df.groupby('diabetes_label'):
    ax.hist(grp['age'], bins=20, alpha=0.65, label=label,
            color=palette[label], edgecolor='white')
ax.set_title('Age Distribution by Diabetes Status', fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_xlabel('Age (years)', fontsize=LABEL_FS)
ax.set_ylabel('Count', fontsize=LABEL_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.legend(fontsize=LEGEND_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots/02_age_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 2")

# Plot 3: Sex vs. Diabetes
fig, ax = plt.subplots(figsize=(8, 6), facecolor='#F8F9FA')
sex_diab = df.groupby(['sex_label', 'diabetes_label']).size().unstack()
sex_pct  = sex_diab.div(sex_diab.sum(axis=1), axis=0) * 100
sex_pct.plot(kind='bar', ax=ax,
             color=[palette['Normal'], palette['Prediabetes']],
             edgecolor='white', linewidth=1.2, width=0.5)
ax.set_title('Prediabetes Rate by Sex', fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_ylabel('Percentage (%)', fontsize=LABEL_FS)
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=TICK_FS)
ax.tick_params(axis='y', labelsize=TICK_FS)
ax.legend(fontsize=LEGEND_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots/03_sex_vs_diabetes.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 3")

# Plot 4: BMI distribution
fig, ax = plt.subplots(figsize=(9, 6), facecolor='#F8F9FA')
for label, grp in df.groupby('diabetes_label'):
    ax.hist(grp['BMI'], bins=30, alpha=0.65, label=label,
            color=palette[label], edgecolor='white')
ax.axvline(25, color='gray', linestyle='--', linewidth=1.5,
           label='Overweight threshold (25)')
ax.set_title('BMI Distribution by Diabetes Status', fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_xlabel('BMI (kg/m²)', fontsize=LABEL_FS)
ax.set_ylabel('Count', fontsize=LABEL_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.legend(fontsize=LEGEND_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots/04_bmi_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 4")

# Plot 5: Work hours boxplot
fig, ax = plt.subplots(figsize=(8, 6), facecolor='#F8F9FA')
employed = df[df['EC_wht_23'] > 0]
sns.boxplot(data=employed, x='diabetes_label', y='EC_wht_23',
            hue='diabetes_label', palette=palette, ax=ax,
            linewidth=1.8, width=0.45, order=['Normal', 'Prediabetes'],
            legend=False)
ax.set_title('Weekly Work Hours (Employed Only)', fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_xlabel('')
ax.set_ylabel('Hours / Week', fontsize=LABEL_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots/05_work_hours_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 5")

# Plot 6: Weight-control exercise
fig, ax = plt.subplots(figsize=(9, 6), facecolor='#F8F9FA')
bo_diab = df.groupby(['BO3_01_label', 'diabetes_label']).size().unstack()
bo_pct  = bo_diab.div(bo_diab.sum(axis=1), axis=0) * 100
bo_pct.plot(kind='bar', ax=ax,
            color=[palette['Normal'], palette['Prediabetes']],
            edgecolor='white', linewidth=1.2, width=0.5)
ax.set_title('Weight-Control Exercise vs Diabetes Status',
             fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_ylabel('Percentage (%)', fontsize=LABEL_FS)
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha='right', fontsize=TICK_FS)
ax.tick_params(axis='y', labelsize=TICK_FS)
ax.legend(fontsize=LEGEND_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots/06_weight_control_exercise.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 6")

# ── Plot 7: Physical activity heatmap ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5), facecolor='#F8F9FA')
activity_cols = ['BE3_31', 'BE5_1', 'BE3_72', 'BE3_82', 'BE3_76', 'BE3_86']
act_labels    = ['Walking (days/wk)', 'Strength Training', 'High-Int Work',
                 'Mod-Int Work', 'High-Int Leisure', 'Mod-Int Leisure']
act_means = df.groupby('diabetes_label')[activity_cols].mean()
act_means.columns = act_labels
sns.heatmap(act_means, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax,
            linewidths=0.8, cbar_kws={'label': 'Mean Days/Week'},
            annot_kws={'size': 14})
ax.set_title('Mean Physical Activity (Days/Week) by Diabetes Status',
             fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_ylabel('')
ax.tick_params(axis='x', labelsize=TICK_FS, rotation=20)
ax.tick_params(axis='y', labelsize=TICK_FS, rotation=0)
plt.tight_layout()
plt.savefig('eda_plots/07_activity_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 7")

# ── Plot 8: Prediabetes rate by occupation ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7), facecolor='#F8F9FA')
occ_rate = df.groupby('EC_occp_label')['diabetes'].mean().sort_values(ascending=True) * 100
colors   = ['#E05C5C' if v > df['diabetes'].mean() * 100 else '#4A90D9'
            for v in occ_rate.values]
ax.barh(occ_rate.index, occ_rate.values, color=colors, edgecolor='white',
        linewidth=1.2, height=0.65)
ax.axvline(df['diabetes'].mean() * 100, color='gray', linestyle='--',
           linewidth=1.5, label=f'Overall avg ({df["diabetes"].mean()*100:.1f}%)')
ax.set_title('Prediabetes Rate by Occupation', fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.set_xlabel('Prediabetes Rate (%)', fontsize=LABEL_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.legend(fontsize=LEGEND_FS)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('eda_plots/08_occupation_prediabetes_rate.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 8")

# ── Plot 9: Correlation heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10), facecolor='#F8F9FA')
corr_cols   = ['age', 'BMI', 'EC_wht_23', 'BE3_31', 'BE5_1',
               'BE3_72', 'BE3_82', 'BE3_76', 'BE3_86', 'BO3_01', 'diabetes']
corr_labels = ['Age', 'BMI', 'Work Hrs/Wk', 'Walking', 'Strength',
               'High-Int Work', 'Mod-Int Work', 'High-Int Leisure',
               'Mod-Int Leisure', 'Weight-Ctrl Exercise', 'Diabetes']
corr = df[corr_cols].corr()
corr.index   = corr_labels
corr.columns = corr_labels
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, linewidths=0.6, annot_kws={'size': 13},
            cbar_kws={'label': 'Pearson r', 'shrink': 0.7})
ax.set_title('Feature Correlation Matrix', fontsize=TITLE_FS, fontweight='bold', pad=12)
ax.tick_params(axis='x', labelsize=TICK_FS, rotation=35)
ax.tick_params(axis='y', labelsize=TICK_FS, rotation=0)
plt.tight_layout()
plt.savefig('eda_plots/09_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot 9")

print("\nAll 9 plots saved to eda_plots/")
