import pandas as pd
import numpy as np

print("=" * 60)
print("DATA PREPROCESSING — LOAN APPROVAL PREDICTION")
print("=" * 60)

# =====================================================================
# STEP 1: LOAD DATA
# =====================================================================
print("\n1. Loading data...")
df = pd.read_csv('data/data.csv') 
df = df.drop(columns=['loan_id'])       # index only, no predictive value
print(f"   Shape: {df.shape}")

# =====================================================================
# STEP 2: CLEAN RAW STRING VALUES
# Leading spaces detected in audit on education and self_employed
# =====================================================================
print("\n2. Cleaning raw string values...")
df['education']     = df['education'].str.strip()
df['self_employed'] = df['self_employed'].str.strip()
df['loan_status'] = df['loan_status'].str.strip()


print(f"   education unique    : {df['education'].unique()}")
print(f"   self_employed unique: {df['self_employed'].unique()}")
print(f"   loan_status unique  : {df['loan_status'].unique()}")

# =====================================================================
# STEP 3: FIX ANOMALOUS VALUES
# residential_assets_value has 28 negative entries (min = -100,000)
# might be data error or debt notation. equated to 0.
# =====================================================================
print("\n3. Fixing anomalous values...")
neg_before = (df['residential_assets_value'] < 0).sum()
df['residential_assets_value'] = df['residential_assets_value'].clip(lower=0)
neg_after  = (df['residential_assets_value'] < 0).sum()
print(f"   residential_assets_value: {neg_before} negatives clipped to 0 "
      f"(remaining: {neg_after})")

# =====================================================================
# STEP 4: ENCODE OUTPUT VARIABLE
# Binary class output
#   Approved  → 0.9  (target: high activation)
#   Rejected  → 0.1  (target: low activation)
# logistic sigmoid cannot produce exactly 0 or 1 — training with hard targets creates an impossible objective and drives weights toward infinity.
# =====================================================================
print("\n4. Encoding output variable...")
df['loan_status'] = df['loan_status'].map({'Approved': 0.9, 'Rejected': 0.1})
print(f"   Approved (0.9): {(df['loan_status'] == 0.9).sum()}")
print(f"   Rejected (0.1): {(df['loan_status'] == 0.1).sum()}")

# =====================================================================
# STEP 5: ENCODE BINARY CLASS INPUTS
# Per Pabico: use 0.9 for presence, 0.1 for absence.
# Avoids zeroing out weights during forward and backward passes.
# =====================================================================
print("\n5. Encoding binary class inputs (0.1 / 0.9)...")
df['education']     = df['education'].map({'Graduate': 0.9,
                                           'Not Graduate': 0.1})
df['self_employed'] = df['self_employed'].map({'Yes': 0.9,
                                               'No': 0.1})
print("   education     → Graduate=0.9, Not Graduate=0.1")
print("   self_employed → Yes=0.9, No=0.1")

# =====================================================================
# STEP 6: TRAIN / TEST SPLIT — BEFORE SCALING
# Must split BEFORE computing min/max.
# If we scale first, the test set leaks into the scaling
# =====================================================================
print("\n6. Splitting into train (80%) and test (20%) sets...")
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx   = int(len(df_shuffled) * 0.8)
train_df    = df_shuffled.iloc[:split_idx].copy()
test_df     = df_shuffled.iloc[split_idx:].copy()
print(f"   Train: {len(train_df)} rows | Test: {len(test_df)} rows")

# =====================================================================
# STEP 7: SCALE CONTINUOUS INPUTS TO [0.1, 0.9]
# Formula
#   Tar = T_min + [(Val - V_min) / (V_max - V_min)] * (T_max - T_min)
#   where T_min = 0.1, T_max = 0.9
#
# Min and max are derived from TRAINING SET ONLY.
# The same parameters are applied to the test set.
# Test values outside the training range are allowed — clipping
# would distort genuine outliers.
# =====================================================================
print("\n7. Scaling continuous inputs to [0.1, 0.9] (fit on train only)...")

T_MIN = 0.1
T_MAX = 0.9

continuous_cols = [
    'no_of_dependents',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value'
]

scaling_params = {}  # store for paper documentation and inverse transform

for col in continuous_cols:
    v_min = train_df[col].min()
    v_max = train_df[col].max()
    scaling_params[col] = {'v_min': v_min, 'v_max': v_max}

    if v_max != v_min:
        train_df[col] = T_MIN + ((train_df[col] - v_min) /
                                 (v_max - v_min)) * (T_MAX - T_MIN)
        test_df[col]  = T_MIN + ((test_df[col]  - v_min) /
                                 (v_max - v_min)) * (T_MAX - T_MIN)
    else:
        # Edge case: constant column — assign midpoint
        train_df[col] = 0.5
        test_df[col]  = 0.5

    print(f"   {col:<30} V_min={v_min:>12.0f}  V_max={v_max:>12.0f}")

# =====================================================================
# STEP 8: SEPARATE INPUTS AND OUTPUT
# =====================================================================
print("\n8. Separating inputs (X) and output (Y)...")

feature_cols = [
    'no_of_dependents', 'education', 'self_employed',
    'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value'
]

X_train = train_df[feature_cols]
Y_train = train_df[['loan_status']]
X_test  = test_df[feature_cols]
Y_test  = test_df[['loan_status']]

# =====================================================================
# STEP 9: DATA CHECK
# =====================================================================

print(f"   X_train shape : {X_train.shape}   (should be ~3415 x 11)")
print(f"   X_test  shape : {X_test.shape}   (should be ~854  x 11)")
print(f"\n   X_train value range (should all be near [0.1, 0.9]):")
print(f"   min = {X_train.min().min():.4f}   max = {X_train.max().max():.4f}")
print(f"\n   X_test value range (may slightly exceed bounds for outliers):")
print(f"   min = {X_test.min().min():.4f}   max = {X_test.max().max():.4f}")

print(f"\n   Y_train distribution:")
print(f"   Approved (0.9): {(Y_train['loan_status'] == 0.9).sum()}")
print(f"   Rejected (0.1): {(Y_train['loan_status'] == 0.1).sum()}")

# =====================================================================
# STEP 10: SAVE TO CSV
# =====================================================================
print("\n10. Saving to CSV...")
X_train.to_csv('X_train.csv', index=False)
Y_train.to_csv('Y_train.csv', index=False)
X_test.to_csv('X_test.csv',   index=False)
Y_test.to_csv('Y_test.csv',   index=False)

print("\n--- Preprocessing complete ---")
print(f"Input nodes  : {len(feature_cols)}")
print(f"Output nodes : 1")
print(f"Network I/O  : {len(feature_cols)} → ? hidden → 1")