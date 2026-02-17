import sys
import os
import pandas as pd
import numpy as np

INPUT_CSV = "NSMES1988.csv"
OUTPUT_CSV = "NSMES1988updated.csv"
REPORT_TXT = "NSMES1988_report.txt"

def find_column(df, name):
    lname = name.lower()
    for col in df.columns:
        if col.lower().strip() == lname:
            return col
    return None

def basic_stats(series):
    return {
        'count': int(series.count()),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std(ddof=1)),
        'min': float(series.min()),
        '25%': float(series.quantile(0.25)),
        '50%': float(series.quantile(0.5)),
        '75%': float(series.quantile(0.75)),
        'max': float(series.max())
    }

def compare_stats(basic, describe):
    diffs = {}
    for k, v in basic.items():
        if k in describe:
            desc_v = float(describe[k])
            if isinstance(v, (int, float)):
                equal = np.isclose(v, desc_v, rtol=1e-6, atol=1e-9)
            else:
                equal = v == desc_v
            diffs[k] = {'basic': v, 'describe': desc_v, 'equal': bool(equal)}
    return diffs

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Input CSV '{INPUT_CSV}' not found in current directory.")
        sys.exit(2)

    df = pd.read_csv(INPUT_CSV)

    age_col = find_column(df, 'age')
    inc_col = find_column(df, 'income')

    if age_col is None or inc_col is None:
        print("Could not find 'age' or 'income' columns (case-insensitive match). Available columns:")
        print(list(df.columns))
        sys.exit(3)

    # Copy dataframe to avoid modifying original in memory
    df_updated = df.copy()

    # Multiply as requested
    df_updated[age_col] = pd.to_numeric(df_updated[age_col], errors='coerce') * 10
    df_updated[inc_col] = pd.to_numeric(df_updated[inc_col], errors='coerce') * 10000

    # Save updated CSV
    df_updated.to_csv(OUTPUT_CSV, index=False)

    # Compute statistics
    stats_age = basic_stats(df_updated[age_col].dropna())
    stats_inc = basic_stats(df_updated[inc_col].dropna())

    # Describe
    desc = df_updated[[age_col, inc_col]].describe().to_dict()

    comp_age = compare_stats(stats_age, {
        'count': desc[age_col]['count'], 'mean': desc[age_col]['mean'],
        'std': desc[age_col]['std'], 'min': desc[age_col]['min'],
        '25%': desc[age_col]['25%'], '50%': desc[age_col]['50%'], '75%': desc[age_col]['75%'], 'max': desc[age_col]['max']
    })
    comp_inc = compare_stats(stats_inc, {
        'count': desc[inc_col]['count'], 'mean': desc[inc_col]['mean'],
        'std': desc[inc_col]['std'], 'min': desc[inc_col]['min'],
        '25%': desc[inc_col]['25%'], '50%': desc[inc_col]['50%'], '75%': desc[inc_col]['75%'], 'max': desc[inc_col]['max']
    })

    # Write brief report
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write('Brief Statistical Report for NSMES1988updated.csv\n')
        f.write('--------------------------------------------\n\n')
        f.write(f"Transformed columns: '{age_col}' multiplied by 10; '{inc_col}' multiplied by 10000.\n\n")

        f.write('Basic statistics (computed explicitly):\n')
        f.write('\nAge statistics:\n')
        for k, v in stats_age.items():
            f.write(f" - {k}: {v}\n")
        f.write('\nIncome statistics:\n')
        for k, v in stats_inc.items():
            f.write(f" - {k}: {v}\n")

        f.write('\nDescribe() output summary comparison:\n')
        f.write('\nAge comparisons:\n')
        for k, d in comp_age.items():
            f.write(f" - {k}: basic={d['basic']}, describe={d['describe']}, equal={d['equal']}\n")

        f.write('\nIncome comparisons:\n')
        for k, d in comp_inc.items():
            f.write(f" - {k}: basic={d['basic']}, describe={d['describe']}, equal={d['equal']}\n")

        f.write('\nNotes:\n')
        f.write(' - The `describe()` summary should match the explicit basic statistics for numeric measures (count, mean, std, min, 25%, 50%, 75%, max).\n')
        f.write(' - Differences may arise due to NaN handling or floating-point rounding; comparisons use a tolerance for equality.\n')

    # Also print a concise summary to stdout
    print('Updated CSV saved to', OUTPUT_CSV)
    print('Brief report saved to', REPORT_TXT)

if __name__ == '__main__':
    main()
