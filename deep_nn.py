# This file implements a deep feedforward neural network with two hidden layers,
# trained using the Delta-Bar-Delta with Momentum optimization algorithm.
# CABRERA, JAN CARLO L.
# REFERENCED PSEUDOCODES FROM: backprop.bas and pseudocode-ann.pl

import pandas as pd
import numpy as np

print("=" * 60)
print(" PHASE 2B — DEEP NETWORK")
print(" Architecture : 11 → 16 → 8 → 1")
print(" Optimizer    : Full-Batch DBD with Momentum")
print("=" * 60)

# =====================================================================
# SECTION 1: CONFIGURATION
# =====================================================================
J1    = 16       # Hidden layer 1 — expand: more nodes than inputs,  lets the network find rich combinations of all 11 inputs
J2    = 8        # Hidden layer 2 — compress: distill what layer 1 found
EPOCHS      = 10000
RANDOM_SEED = 42

# BACKPROPAGATION PARAMETERS
KAPPA = 0.01
PHI   = 0.50
THETA = 0.70
MU    = 0.90

EVAL_EVERY    = 200
OVERFIT_LIMIT = 5.0

# =====================================================================
# SECTION 2: LOAD DATA
# =====================================================================
X_train = pd.read_csv('X_train.csv').values          # (3415, 11)
Y_train = pd.read_csv('Y_train.csv').values.reshape(-1, 1)
X_test  = pd.read_csv('X_test.csv').values
Y_test  = pd.read_csv('Y_test.csv').values.reshape(-1, 1)

N, I = X_train.shape
K    = 1

print(f"\nTraining    : {N} examples")
print(f"Architecture: {I} → {J1} → {J2} → {K}")
print(f"DBD params  : κ={KAPPA}  φ={PHI}  θ={THETA}  μ={MU}")
print(f"Epochs      : {EPOCHS}\n")

# =====================================================================
# SECTION 3: WEIGHT INITIALIZATION
#
# Three weight matrices for a two-hidden-layer network:
#
#   A1  connects Input      → Hidden Layer 1
#       shape: (I+1, J1)    the +1 accounts for the bias node
#
#   A2  connects Hidden L1  → Hidden Layer 2
#       shape: (J1+1, J2)
#
#   B   connects Hidden L2  → Output
#       shape: (J2+1, K)
#
# Hidden weights: uniform(-0.1, 0.1) 
# Output weights: alternating +1/-1
# DBD memory (e, f, c): one set per weight matrix, per layer
# =====================================================================
np.random.seed(RANDOM_SEED)

# 
A1 = np.random.uniform(-0.1, 0.1, (I  + 1, J1))
A2 = np.random.uniform(-0.1, 0.1, (J1 + 1, J2))

B = np.array([[0.0 if j == 0 else (1.0 if j % 2 != 0 else -1.0) 
               for _ in range(K)] 
              for j in range(J2 + 1)])

# DBD memory — Layer 1 hidden weights
eA1 = np.full_like(A1, KAPPA)
fA1 = np.zeros_like(A1)
cA1 = np.zeros_like(A1)

# DBD memory — Layer 2 hidden weights
eA2 = np.full_like(A2, KAPPA)
fA2 = np.zeros_like(A2)
cA2 = np.zeros_like(A2)

# DBD memory — Output weights
eB  = np.full_like(B,  KAPPA)
fB  = np.zeros_like(B)
cB  = np.zeros_like(B)

total_weights = A1.size + A2.size + B.size
print("Weight initialization:")
print(f"  A1 (input → H1)  : uniform(-0.1, 0.1)  shape {A1.shape}")
print(f"  A2 (H1    → H2)  : uniform(-0.1, 0.1)  shape {A2.shape}")
print(f"  B  (H2    → out) : alternating ±1       shape {B.shape}")
print(f"  Total weights    : {total_weights}\n")

# =====================================================================
# SECTION 4: CORE FUNCTIONS
# =====================================================================
def logistic(uv):
    """
    Logistic sigmoid activation function.
    Clips input to [-88, 88] to prevent overflow in exp(),
    """
    return 1.0 / (1.0 + np.exp(-np.clip(uv, -88, 88)))


def dbd_update(W, dW, f, e, c):
    """
    Delta-Bar-Delta with Momentum.
    """
    agree = (dW * f) > 0.0
    e     = np.where(agree, e + KAPPA, e * PHI)
    f     = (1.0 - THETA) * dW + THETA * f
    c     = (1.0 - MU) * (-e * dW) + MU * c
    W     = W + c
    return W, f, e, c

# =====================================================================
# SECTION 5: PREPEND BIAS COLUMNS
#
# bias is handled as weight index 0:
#   u = a(0, j)          ← start with bias weight
#   u = u + a(i,j)*x(i)  ← add weighted inputs
# =====================================================================
ones_train = np.ones((N, 1))
ones_test  = np.ones((X_test.shape[0], 1))

X_train_b = np.hstack([ones_train, X_train])   # (N, I+1)
X_test_b  = np.hstack([ones_test,  X_test])    # (Nt, I+1)

# =====================================================================
# SECTION 6: TRAINING LOOP
#
#   For each epoch:
#     1. Forward pass  — compute activations through all layers
#     2. Backward pass — accumulate derivatives through all layers
#     3. Weight update — one DBD step per weight matrix
#
# Deep network forward pass (two hidden layers):
#
#   U1 = X_b  · A1        weighted sum at hidden layer 1
#   Y1 = σ(U1)            activation of hidden layer 1
#
#   U2 = Y1_b · A2        weighted sum at hidden layer 2
#   Y2 = σ(U2)            activation of hidden layer 2
#
#   V  = Y2_b · B         weighted sum at output
#   Z  = σ(V)             network output
#
# Deep network backward pass:
#
#   P  = (Z - T) · Z · (1-Z)             output delta       (N, K)
#   dB = Y2_b.T · P / N                  output weight grad (J2+1, K)
#
#   Q2 = (P · B[1:].T) · Y2 · (1-Y2)    layer 2 delta      (N, J2)
#   dA2= Y1_b.T · Q2 / N                layer 2 weight grad (J1+1, J2)
#
#   Q1 = (Q2 · A2[1:].T) · Y1 · (1-Y1)  layer 1 delta      (N, J1)
#   dA1= X_b.T · Q1 / N                 layer 1 weight grad (I+1, J1)
#
# The [1:] slicing on B and A2 skips the bias row when propagating
# error backward — the bias has no connection to the previous layer.
# =====================================================================
learning_curve = []

print("Training...\n")
print(f"{'Epoch':>7}  {'Train MSE':>12}  {'Test MSE':>12}")
print("-" * 38)

for epoch in range(EPOCHS):

    # ------------------------------------------------------------------
    # FORWARD PASS
    # ------------------------------------------------------------------

    # Input → Hidden Layer 1
    U1   = np.dot(X_train_b, A1)               # (N, J1)
    Y1   = logistic(U1)                         # (N, J1)
    Y1_b = np.hstack([ones_train, Y1])          # (N, J1+1) — add bias

    # Hidden Layer 1 → Hidden Layer 2
    U2   = np.dot(Y1_b, A2)                     # (N, J2)
    Y2   = logistic(U2)                         # (N, J2)
    Y2_b = np.hstack([ones_train, Y2])          # (N, J2+1) — add bias

    # Hidden Layer 2 → Output
    V    = np.dot(Y2_b, B)                      # (N, K)
    Z    = logistic(V)                          # (N, K)

    # ------------------------------------------------------------------
    # BACKWARD PASS
    # ------------------------------------------------------------------

    # --- Output layer ---
    # P: how much the output node's pre-activation contributed to error
    P   = (Z - Y_train) * Z * (1.0 - Z)        # (N, K)
    dB  = np.dot(Y2_b.T, P) / N                # (J2+1, K)

    # --- Hidden Layer 2 ---
    # Propagate P backward through output weights (skip bias row 0)
    # Multiply by local derivative Y2*(1-Y2) — the sigmoid's own slope
    Q2  = np.dot(P, B[1:].T) * Y2 * (1.0 - Y2)   # (N, J2)
    dA2 = np.dot(Y1_b.T, Q2) / N                  # (J1+1, J2)

    # --- Hidden Layer 1 ---
    # Propagate Q2 backward through Layer 2 weights (skip bias row 0)
    # Multiply by local derivative Y1*(1-Y1)
    Q1  = np.dot(Q2, A2[1:].T) * Y1 * (1.0 - Y1)  # (N, J1)
    dA1 = np.dot(X_train_b.T, Q1) / N              # (I+1, J1)

    # ------------------------------------------------------------------
    # WEIGHT UPDATE — Delta-Bar-Delta with Momentum
    # Each weight matrix updated independently with its own DBD state
    # ------------------------------------------------------------------
    B,  fB,  eB,  cB  = dbd_update(B,  dB,  fB,  eB,  cB)
    A2, fA2, eA2, cA2 = dbd_update(A2, dA2, fA2, eA2, cA2)
    A1, fA1, eA1, cA1 = dbd_update(A1, dA1, fA1, eA1, cA1)

    # ------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------
    if epoch % EVAL_EVERY == 0:
        train_mse = float(np.mean((Z - Y_train) ** 2))

        # Test forward pass — no weight update
        U1_t  = np.dot(X_test_b, A1)
        Y1_t  = logistic(U1_t)
        Y1_tb = np.hstack([ones_test, Y1_t])
        U2_t  = np.dot(Y1_tb, A2)
        Y2_t  = logistic(U2_t)
        Y2_tb = np.hstack([ones_test, Y2_t])
        Z_t   = logistic(np.dot(Y2_tb, B))
        test_mse = float(np.mean((Z_t - Y_test) ** 2))

        learning_curve.append({
            'epoch'    : epoch,
            'train_mse': round(train_mse, 6),
            'test_mse' : round(test_mse,  6)
        })
        print(f"{epoch:>7}  {train_mse:>12.6f}  {test_mse:>12.6f}")

print("\nTraining complete.\n")

# =====================================================================
# SECTION 7: FINAL EVALUATION
# =====================================================================

# Final test forward pass
U1_t  = np.dot(X_test_b, A1)
Y1_t  = logistic(U1_t)
Y1_tb = np.hstack([ones_test, Y1_t])
U2_t  = np.dot(Y1_tb, A2)
Y2_t  = logistic(U2_t)
Y2_tb = np.hstack([ones_test, Y2_t])
Z_t   = logistic(np.dot(Y2_tb, B))

approved_mask = (Y_test == 0.9).flatten()
rejected_mask = (Y_test == 0.1).flatten()

print("--- Score Distribution ---")
print(f"  Avg output → Approved (target 0.9) : "
      f"{np.mean(Z_t[approved_mask]):.4f}")
print(f"  Avg output → Rejected (target 0.1) : "
      f"{np.mean(Z_t[rejected_mask]):.4f}")
print(f"  Separation gap                     : "
      f"{np.mean(Z_t[approved_mask]) - np.mean(Z_t[rejected_mask]):.4f}\n")

print("--- Threshold Tuning ---")
print(f"{'Thresh':>8} {'Acc%':>8} {'TP':>6} {'TN':>6} "
      f"{'FP':>6} {'FN':>6} {'Prec':>8} {'Rec':>8} {'F1':>8}")
print("-" * 72)

threshold_results = []
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

for thresh in thresholds:
    preds = np.where(Z_t >= thresh, 0.9, 0.1)
    TP = int(np.sum((preds == 0.9) & (Y_test == 0.9)))
    TN = int(np.sum((preds == 0.1) & (Y_test == 0.1)))
    FP = int(np.sum((preds == 0.9) & (Y_test == 0.1)))
    FN = int(np.sum((preds == 0.1) & (Y_test == 0.9)))

    acc  = (TP + TN) / len(Y_test) * 100
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)
            if (prec + rec) > 0 else 0.0)

    threshold_results.append({
        'threshold': thresh, 'accuracy': round(acc, 2),
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'precision': round(prec, 4),
        'recall'   : round(rec,  4),
        'f1'       : round(f1,   4)
    })
    print(f"{thresh:>8.2f} {acc:>7.2f}% {TP:>6} {TN:>6} "
          f"{FP:>6} {FN:>6} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")

# =====================================================================
# SECTION 8: OVERFITTING DIAGNOSIS
# =====================================================================

# Training set forward pass with final weights
U1_tr  = np.dot(X_train_b, A1)
Y1_tr  = logistic(U1_tr)
Y1_trb = np.hstack([ones_train, Y1_tr])
U2_tr  = np.dot(Y1_trb, A2)
Y2_tr  = logistic(U2_tr)
Y2_trb = np.hstack([ones_train, Y2_tr])
Z_tr   = logistic(np.dot(Y2_trb, B))

print("\n--- Overfitting Diagnosis ---")
print(f"{'Thresh':>8} {'Train%':>9} {'Test%':>9} "
      f"{'Gap%':>8} {'Verdict':>12}")
print("-" * 52)

gaps = []
for r in threshold_results:
    thresh      = r['threshold']
    train_preds = np.where(Z_tr >= thresh, 0.9, 0.1)
    train_acc   = np.sum(train_preds == Y_train) / N * 100
    test_acc    = r['accuracy']
    gap         = train_acc - test_acc
    verdict     = "OVERFIT" if gap > OVERFIT_LIMIT else "OK"
    gaps.append(gap)
    print(f"{thresh:>8.2f} {train_acc:>8.2f}% {test_acc:>8.2f}% "
          f"{gap:>7.2f}% {verdict:>12}")

avg_gap = np.mean(gaps)
print(f"\n  Average gap : {avg_gap:.2f}%")
print(f"  Diagnosis   : "
      f"{'OVERFITTING' if avg_gap > OVERFIT_LIMIT else 'ACCEPTABLE'}")

# =====================================================================
# SECTION 9: SAVE
# =====================================================================
pd.DataFrame(learning_curve).to_csv('lc_phase2b.csv',      index=False)
pd.DataFrame(threshold_results).to_csv('results_phase2b.csv', index=False)
print("\nSaved: lc_phase2b.csv  |  results_phase2b.csv")