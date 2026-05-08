import pandas as pd
import numpy as np

df = pd.read_csv('data/data.csv')  # adjust path

print("=" * 60)
print("AUDIT 1: SHAPE AND BASIC INFO")
print("=" * 60)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print()
print(df.dtypes)

print("\n" + "=" * 60)
print("AUDIT 2: NULL VALUE CHECK")
print("=" * 60)
null_counts = df.isnull().sum()
null_pct = (null_counts / len(df) * 100).round(2)
null_report = pd.DataFrame({'null_count': null_counts, 'null_pct': null_pct})
print(null_report[null_report['null_count'] > 0])
if null_counts.sum() == 0:
    print("No null values found.")

print("\n" + "=" * 60)
print("AUDIT 3: CLASS DISTRIBUTION (OUTPUT)")
print("=" * 60)
counts = df['loan_status'].value_counts()
pct = df['loan_status'].value_counts(normalize=True) * 100
print(pd.DataFrame({'count': counts, 'pct': pct.round(1)}))

print("\n" + "=" * 60)
print("AUDIT 4: UNIQUE VALUES FOR CATEGORICAL/DISCRETE COLUMNS")
print("=" * 60)
suspect_cols = ['education', 'self_employed', 'no_of_dependents', 'loan_term']
for col in suspect_cols:
    vals = sorted(df[col].dropna().unique())
    print(f"\n{col}: {len(vals)} unique values")
    print(f"  Values: {vals}")

print("\n" + "=" * 60)
print("AUDIT 5: DESCRIPTIVE STATS FOR CONTINUOUS COLUMNS")
print("=" * 60)
continuous_cols = [
    'income_annum', 'loan_amount', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value'
]
print(df[continuous_cols].describe().round(2).T[
    ['min', 'max', 'mean', 'std', '25%', '50%', '75%']
])

print("\n" + "=" * 60)
print("AUDIT 6: NEGATIVE VALUES CHECK")
print("=" * 60)
for col in continuous_cols:
    neg_count = (df[col] < 0).sum()
    if neg_count > 0:
        print(f"  WARNING: {col} has {neg_count} negative values")
    else:
        print(f"  {col}: OK (no negatives)")

print("\n" + "=" * 60)
print("AUDIT 7: DUPLICATE ROWS")
print("=" * 60)
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")

print("\n" + "=" * 60)
print("AUDIT 8: loan_id INTEGRITY CHECK")
print("=" * 60)
print(f"Unique loan_ids: {df['loan_id'].nunique()} out of {len(df)} rows")
if df['loan_id'].nunique() == len(df):
    print("  All loan_ids are unique. Safe to drop as index.")
else:
    print("  WARNING: Duplicate loan_ids detected.")