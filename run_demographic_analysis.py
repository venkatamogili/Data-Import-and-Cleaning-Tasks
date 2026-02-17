import os
import pandas as pd
import numpy as np

# Prefer analysis df saved earlier
candidates = ['NSMES1988_analysis_df.csv', 'NSMES1988updated.csv', 'NSMES1988new.csv', 'NSMES1988.csv']
path = next((p for p in candidates if os.path.exists(p)), None)
if path is None:
    raise FileNotFoundError('No source CSV found')

print('Using source:', path)
df = pd.read_csv(path)

# Ensure gender, health exist
for req in ['gender','health']:
    if req not in df.columns:
        raise KeyError(f"Required column '{req}' not found in {path}")

# Create age groups (5-year bins)
min_age = int(np.floor(df['age'].min() / 5.0) * 5)
max_age = int(np.ceil(df['age'].max() / 5.0) * 5)
bins = list(range(min_age, max_age + 5, 5))
labels = [f"{b}-{b+4}" for b in bins[:-1]]
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Age and Gender distribution table
age_gender = pd.crosstab(df['age_group'], df['gender']).sort_index()
age_gender.to_csv('age_gender_distribution.csv')
print('\nAge x Gender distribution saved to age_gender_distribution.csv')
print(age_gender)

# Health status by Gender (counts and column %)
health_gender_counts = pd.crosstab(df['health'], df['gender'])
health_gender_counts.to_csv('health_by_gender_counts.csv')

# Column percentages (distribution within each gender)
health_gender_colpct = health_gender_counts.div(health_gender_counts.sum(axis=0), axis=1).round(4)
health_gender_colpct.to_csv('health_by_gender_colpct.csv')

print('\nHealth x Gender counts saved to health_by_gender_counts.csv')
print(health_gender_counts)
print('\nHealth x Gender column % (proportion by gender) saved to health_by_gender_colpct.csv')
print(health_gender_colpct)

# Additional distribution tables for requested factors by gender
factors = ['married','employed','insurance','medicaid','school']
for f in factors:
    if f in df.columns:
        tbl = pd.crosstab(df[f], df['gender'])
        out = f"{f}_by_gender.csv"
        tbl.to_csv(out)
        print(f"\nSaved {out}")
        print(tbl)
    else:
        print(f"\nColumn {f} not found; skipping {f} distribution")

# Save a short summary to text
with open('demographic_analysis_summary.txt','w',encoding='utf-8') as fh:
    fh.write(f"Source file: {path}\n")
    fh.write(f"Rows: {len(df)}\n")
    fh.write('\nAge x Gender distribution (counts):\n')
    fh.write(age_gender.to_csv(index=True))
    fh.write('\nHealth x Gender counts:\n')
    fh.write(health_gender_counts.to_csv(index=True))
    fh.write('\nHealth x Gender column %:\n')
    fh.write(health_gender_colpct.to_csv(index=True))

print('\nSummary written to demographic_analysis_summary.txt')
