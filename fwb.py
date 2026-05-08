import pandas as pd
import numpy as np

print("=" * 60)
print(" PHASE 2A — BASE NETWORK")
print(" Architecture : 11 → 8 → 1")
print(" Optimizer    : Full-Batch DBD with Momentum")
print(" Reference    : Pabico, backprop.bas / CMSC 191")
print("=" * 60)

# =====================================================================
# SECTION 1: CONFIGURATION
# All tunable hyperparameters are declared here.
# Do not change values anywhere else in the script.
# =====================================================================
NUM_HIDDEN    = 16        # 2/3 rule: (2/3 × 11) + 1 ≈ 8
EPOCHS        = 5000
RANDOM_SEED   = 42

# DBD hyperparameters — directly from professor's BASIC
KAPPA = 0.01   # Additive increase when gradient agrees with history
PHI   = 0.50   # Multiplicative decrease when gradient flips
THETA = 0.70   # Smoothing factor for gradient trace f
MU    = 0.90   # Momentum coefficient

EVAL_EVERY    = 100      # Print and log every N epochs
OVERFIT_LIMIT = 5.0      # % gap that triggers overfit warning

# =====================================================================
# SECTION 2: LOAD DATA
# =====================================================================
X_train = pd.read_csv('X_train.csv').values
Y_train = pd.read_csv('Y_train.csv').values.reshape(-1, 1)
X_test  = pd.read_csv('X_test.csv').values
Y_test  = pd.read_csv('Y_test.csv').values.reshape(-1, 1)

N, I  = X_train.shape   # N examples, I inputs
K     = 1               # output nodes
J     = NUM_HIDDEN

print(f"\nTraining   : {N} examples")
print(f"Architecture: {I} → {J} → {K}")
print(f"DBD params  : κ={KAPPA}  φ={PHI}  θ={THETA}  μ={MU}")
print(f"Epochs      : {EPOCHS}\n")

# =====================================================================
# SECTION 3: WEIGHT INITIALIZATION
# Faithful to backprop.bas:
#   Hidden weights  a(i,j) = 0.2*(RND - 0.5) → uniform(-0.1, 0.1)
#   Output weights  b(j,k) = +1 if j even, -1 if j odd
#   Learning rates  e      = kappa (initial per-weight learning rate)
#   All other DBD memory   = 0
# =====================================================================
np.random.seed(RANDOM_SEED)

# Hidden layer weights: (I inputs + 1 bias) × J hidden nodes
# Row 0 = bias weights, rows 1..I = input weights
A     = np.random.uniform(-0.1, 0.1, (I + 1, J))

# Output layer weights: (J hidden + 1 bias) × K outputs
# Alternating ±1 per professor's initWeights subroutine
B     = np.array([[1.0 if j % 2 == 0 else -1.0
                   for _ in range(K)]
                  for j in range(J + 1)])   # shape (J+1, K)

# DBD memory: per-weight adaptive learning rate, gradient trace, momentum
eA = np.full_like(A, KAPPA)
fA = np.zeros_like(A)
cA = np.zeros_like(A)

eB = np.full_like(B, KAPPA)
fB = np.zeros_like(B)
cB = np.zeros_like(B)

print("Weight initialization:")
print(f"  A (hidden) : uniform(-0.1, 0.1)  shape {A.shape}")
print(f"  B (output) : alternating ±1      shape {B.shape}")
print(f"  e (all)    : initialized to kappa = {KAPPA}\n")

# =====================================================================
# SECTION 4: CORE FUNCTIONS
# =====================================================================
def logistic(uv):
    """Sigmoid. Clipped to prevent overflow, matching professor's BASIC."""
    return 1.0 / (1.0 + np.exp(-np.clip(uv, -88, 88)))

def dbd_update(W, dW, f, e, c):
    """
    Delta-Bar-Delta with Momentum.
    Translated directly from professor's change() subroutine in BASIC.

    If the current gradient d agrees with the running average f
    (same sign), increase the per-weight learning rate additively.
    If they disagree, decrease it multiplicatively.
    This allows each weight to accelerate or brake independently.
    """
    agree = (dW * f) > 0.0
    e     = np.where(agree, e + KAPPA, e * PHI)
    f     = (1.0 - THETA) * dW + THETA * f
    c     = (1.0 - MU) * (-e * dW) + MU * c
    W     = W + c
    return W, f, e, c

# =====================================================================
# SECTION 5: TRAINING LOOP
#
# Structure mirrors backprop.bas exactly:
#   DO (epoch)
#       FOR each example n        ← traverse all examples
#           forward(n)            ← compute y and z
#           back(n)               ← accumulate dA and dB
#       NEXT n
#       changeWeights             ← one DBD update after full pass
#   LOOP
#
# We use NumPy matrix ops which produce the same accumulated
# derivatives as the per-example loops in BASIC, but faster.
# =====================================================================
learning_curve = []

# Prepend bias column (1s) to inputs once — avoids repeat allocation
X_train_b = np.hstack([np.ones((N, 1)), X_train])       # (N, I+1)
X_test_b  = np.hstack([np.ones((X_test.shape[0], 1)),
                        X_test])                          # (Nt, I+1)

print("Training...\n")
print(f"{'Epoch':>6}  {'Train MSE':>12}  {'Test MSE':>12}")
print("-" * 36)

for epoch in range(EPOCHS):

    # ------------------------------------------------------------------
    # FORWARD PASS (full batch)
    # u = bias + weighted inputs  →  y = logistic(u)
    # v = bias + weighted hiddens →  z = logistic(v)
    # ------------------------------------------------------------------
    U = np.dot(X_train_b, A)                    # (N, J)
    Y = logistic(U)                             # (N, J)

    Y_b = np.hstack([np.ones((N, 1)), Y])       # (N, J+1)  prepend bias
    V   = np.dot(Y_b, B)                        # (N, K)
    Z   = logistic(V)                           # (N, K)

    # ------------------------------------------------------------------
    # BACKWARD PASS (full batch accumulation)
    # p = dE/dv  for output layer  — equation from back: in BASIC
    # q = dE/du  for hidden layer  — propagated error
    # dB and dA are the accumulated weight derivatives
    # Divided by N to produce the mean gradient (batch averaging)
    # ------------------------------------------------------------------
    P   = (Z - Y_train) * Z * (1.0 - Z)        # (N, K)

    dB  = np.dot(Y_b.T, P) / N                 # (J+1, K)

    # Propagate error back through output weights (excluding bias row 0)
    Q   = np.dot(P, B[1:, :].T) * Y * (1.0 - Y)  # (N, J)

    dA  = np.dot(X_train_b.T, Q) / N           # (I+1, J)

    # ------------------------------------------------------------------
    # WEIGHT UPDATE — Delta-Bar-Delta with Momentum
    # ------------------------------------------------------------------
    A,  fA, eA, cA = dbd_update(A,  dA, fA, eA, cA)
    B,  fB, eB, cB = dbd_update(B,  dB, fB, eB, cB)

    # ------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------
    if epoch % EVAL_EVERY == 0:
        train_mse = float(np.mean((Z - Y_train) ** 2))

        U_t = np.dot(X_test_b, A)
        Y_t = logistic(U_t)
        Y_tb = np.hstack([np.ones((X_test_b.shape[0], 1)), Y_t])
        Z_t = logistic(np.dot(Y_tb, B))
        test_mse = float(np.mean((Z_t - Y_test) ** 2))

        learning_curve.append({
            'epoch'    : epoch,
            'train_mse': round(train_mse, 6),
            'test_mse' : round(test_mse,  6)
        })
        print(f"{epoch:>6}  {train_mse:>12.6f}  {test_mse:>12.6f}")

print("\nTraining complete.\n")

# =====================================================================
# SECTION 6: FINAL TEST EVALUATION + THRESHOLD TUNING
# =====================================================================
U_t  = np.dot(X_test_b, A)
Y_t  = logistic(U_t)
Y_tb = np.hstack([np.ones((Y_t.shape[0], 1)), Y_t])
Z_t  = logistic(np.dot(Y_tb, B))

# Score distribution — sanity check
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
        'precision': round(prec, 4), 'recall': round(rec, 4),
        'f1': round(f1, 4)
    })
    print(f"{thresh:>8.2f} {acc:>7.2f}% {TP:>6} {TN:>6} "
          f"{FP:>6} {FN:>6} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")

# =====================================================================
# SECTION 7: OVERFITTING DIAGNOSIS
# =====================================================================
X_train_b_full = np.hstack([np.ones((N, 1)), X_train])
U_tr = np.dot(X_train_b_full, A)
Y_tr = logistic(U_tr)
Y_trb = np.hstack([np.ones((Y_tr.shape[0], 1)), Y_tr])
Z_tr = logistic(np.dot(Y_trb, B))

print("\n--- Overfitting Diagnosis ---")
print(f"{'Thresh':>8} {'Train%':>9} {'Test%':>9} {'Gap%':>8} {'Verdict':>12}")
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
# SECTION 8: SAVE ALL OUTPUTS
# =====================================================================
pd.DataFrame(learning_curve).to_csv('lc_phase2a.csv', index=False)
pd.DataFrame(threshold_results).to_csv('results_phase2a.csv', index=False)
print("\nSaved: lc_phase2a.csv  |  results_phase2a.csv")