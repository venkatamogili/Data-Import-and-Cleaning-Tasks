import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Input preference
candidates = ['NSMES1988_analysis_df.csv','NSMES1988updated.csv','NSMES1988new.csv','NSMES1988.csv']
path = next((p for p in candidates if os.path.exists(p)), None)
if path is None:
    raise FileNotFoundError('No source CSV found')

print('Using source file:', path)
df = pd.read_csv(path)

# Create working df
df_work = df.copy()

# Ensure categorical types for key columns
for c in ['health','region','gender','married','employed','insurance','medicaid']:
    if c in df_work.columns:
        df_work[c] = df_work[c].astype('category')

# Create output directory for plots
out_dir = 'plots'
os.makedirs(out_dir, exist_ok=True)

# Week 3 plots: categorical data - Health and Region
# Pivot counts (health x region)
pivot_count = pd.crosstab(df_work['health'], df_work['region'])
plt.figure(figsize=(8,5))
sns.heatmap(pivot_count, annot=True, fmt='d', cmap='Blues')
plt.title('Counts: Health by Region')
plt.ylabel('Health')
plt.xlabel('Region')
plt.tight_layout()
fn = os.path.join(out_dir,'heatmap_health_region_counts.png')
plt.savefig(fn)
plt.close()

# Stacked bar chart of health distribution by region (proportion)
prop = pivot_count.div(pivot_count.sum(axis=0), axis=1)
ax = prop.T.plot(kind='bar', stacked=True, figsize=(9,5), colormap='tab20')
ax.set_ylabel('Proportion')
ax.set_title('Proportion of Health Status by Region')
plt.legend(title='Health', bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
fn2 = os.path.join(out_dir,'stacked_health_by_region_prop.png')
plt.savefig(fn2)
plt.close()

# Week 4 plots: analysis and correlation
# Numeric columns selection
numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
# Correlation heatmap
corr = df_work[numeric_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation matrix (numeric variables)')
plt.tight_layout()
fn3 = os.path.join(out_dir,'correlation_heatmap.png')
plt.savefig(fn3)
plt.close()

# Distribution plots for Age and Income
if 'age' in df_work.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df_work['age'].dropna(), kde=True, bins=20)
    plt.title('Age distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'age_distribution.png'))
    plt.close()

if 'income' in df_work.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df_work['income'].dropna(), kde=True, bins=30)
    plt.title('Income distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'income_distribution.png'))
    plt.close()

# Boxplot: Visits by Health
if 'visits' in df_work.columns and 'health' in df_work.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(x='health', y='visits', data=df_work)
    plt.title('Visits by Health Status')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'visits_by_health_boxplot.png'))
    plt.close()

# Save pivot tables to CSV for review
pivot_count.to_csv('pivot_health_region_counts.csv')
prop.to_csv('pivot_health_region_prop.csv')

# Generate detailed report
report_lines = []
report_lines.append('Plotting library choice: Matplotlib + Seaborn')
report_lines.append('\nReasons for choice:')
report_lines.append('- Matplotlib is the standard Python plotting library with strong flexibility.')
report_lines.append('- Seaborn is built on Matplotlib and provides high-level statistical plotting with attractive defaults (heatmaps, boxplots, distribution plots).')
report_lines.append('- Together they enable easy creation of heatmaps, distribution plots, and correlation visuals required for this analysis.')

report_lines.append('\nData source: ' + path)
report_lines.append(f'Rows: {len(df_work)}, Columns: {len(df_work.columns)}')

report_lines.append('\nWeek 3: Categorical analysis (Health and Region)')
report_lines.append('- Heatmap `heatmap_health_region_counts.png` shows counts of observations for each Health × Region cell.')
report_lines.append('- Stacked bar `stacked_health_by_region_prop.png` shows proportion of each health status within each region.')
report_lines.append('\nObservations:')
# Add some computed observations
report_lines.append(f"- Most common health status across regions: {df_work['health'].value_counts().idxmax()}")
health_counts = df_work['health'].value_counts()
for h, cnt in health_counts.items():
    report_lines.append(f"  - {h}: {cnt} records")

# Week 4: Analysis and correlation
report_lines.append('\nWeek 4: Numerical analysis and correlation')
report_lines.append('- Correlation matrix saved to `correlation_heatmap.png` and numeric correlation table below:')
report_lines.append('\nTop correlations (absolute value sorted):')
corr_pairs = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
# pick top non-1 correlations
top_pairs = corr_pairs[corr_pairs < 0.999].dropna().sort_values(ascending=False).head(8)
for (a,b),v in top_pairs.items():
    report_lines.append(f"- {a} vs {b}: corr = {v:.2f}")

report_lines.append('\nAdditional plots: age_distribution.png, income_distribution.png, visits_by_health_boxplot.png')

report_lines.append('\nSummary observations:')
# Example observations using pivots created earlier
# Mean visits by health
if 'visits' in df_work.columns:
    mean_visits_by_health = df_work.groupby('health')['visits'].mean().to_dict()
    for k,v in mean_visits_by_health.items():
        report_lines.append(f"- Mean visits for {k}: {v:.2f}")

# Income by health
if 'income' in df_work.columns:
    mean_income_by_health = df_work.groupby('health')['income'].mean().to_dict()
    for k,v in mean_income_by_health.items():
        report_lines.append(f"- Mean income for {k}: {v:.2f}")

# Age by health
if 'age' in df_work.columns:
    mean_age_by_health = df_work.groupby('health')['age'].mean().to_dict()
    for k,v in mean_age_by_health.items():
        report_lines.append(f"- Mean age for {k}: {v:.2f}")

report_path = 'plots_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print('Plots saved in', out_dir)
print('Report saved to', report_path)
