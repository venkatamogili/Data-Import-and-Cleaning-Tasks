import os
import pandas as pd
import numpy as np

path = 'NSMES1988updated.csv'
if not os.path.exists(path):
    raise FileNotFoundError(f'{path} not found')

df_work = pd.read_csv(path)
print('Loaded', path, '-> shape:', df_work.shape)
print('\nData types:')
print(df_work.dtypes)

cat_candidates = df_work.select_dtypes(include=['object','category']).columns.tolist()
low_card = [col for col in df_work.columns if df_work[col].nunique(dropna=True) <= 50 and col not in cat_candidates]
categorical = sorted(set(cat_candidates + low_card))
print('\nCategorical candidate columns:')
print(categorical)

for c in ['health','region']:
    if c in df_work.columns:
        df_work[c] = df_work[c].astype('category')
    else:
        print(f'Warning: {c} not found')

print('\nCategorical dtypes after casting:')
print(df_work.dtypes[df_work.dtypes == 'category'])

print('\nComputing pivots...')
pivot_count = pd.pivot_table(df_work, index='health', columns='region', values='visits', aggfunc='count', fill_value=0)
pivot_mean_visits = pd.pivot_table(df_work, index='health', columns='region', values='visits', aggfunc='mean')
pivot_mean_income = pd.pivot_table(df_work, index='health', columns='region', values='income', aggfunc='mean')
pivot_mean_age = pd.pivot_table(df_work, index='health', columns='region', values='age', aggfunc='mean')

print('\nPivot counts (visits):')
print(pivot_count)
print('\nPivot mean visits:')
print(pivot_mean_visits)
print('\nPivot mean income:')
print(pivot_mean_income)
print('\nPivot mean age:')
print(pivot_mean_age)

out = 'NSMES1988_analysis_df.csv'
df_work.to_csv(out, index=False)
print('\nSaved analysis df to', out)
