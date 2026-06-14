# ================================================================
# Harder Learning Baseline (MoE NN, GBM, mNAM)
#  - Labels: nonlinear state×geometry + geometry×geometry terms
#  - Noise: heteroscedastic (depends on geometry)
#  - Split: distribution-shifted train/test
#  - Classification: fixed GHz bins (physics-based)
# ================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor
import warnings; warnings.filterwarnings("ignore")
import xgboost as xgb


# ----------------------------- seeds -----------------------------
torch.manual_seed(0); np.random.seed(0)

# -------------------- config flags --------------------
USE_NONLINEAR_INTERACTIONS = True
HETEROSCEDASTIC_NOISE      = True
SHIFTED_TEST               = True
FIXED_BINS                 = True

# -------------------- geometry sampler --------------------
def sample_geometry(n, rng=np.random.default_rng(2)):
    return pd.DataFrame({
        "bottom_gap_mm":           rng.uniform(0.06, 0.14, n),
        "bottom_r_out_mm":         rng.uniform(0.23, 0.27, n),
        "bottom_r_in_mm":          rng.uniform(0.14, 0.18, n),
        "substrate_thickness_mm":  rng.uniform(0.22, 0.28, n),
        "epsilon_r":               rng.uniform(3.55, 3.77, n),
        "top_L2_mm":               rng.uniform(0.06, 0.12, n),
        "top_L3_mm":               rng.uniform(0.04, 0.10, n),
    })

# ------------------------ diode tokens ------------------------
def add_tokens_tm_only(Xdf, d_top_on, d_bot_on, dataset_id=0, n_datasets=1):
    Xo = Xdf.copy()
    Xo["d_top_on"]  = int(bool(d_top_on))
    Xo["d_bot_on"]  = int(bool(d_bot_on))
    Xo["d_both_on"] = Xo["d_top_on"] * Xo["d_bot_on"]
    for k in range(n_datasets):
        Xo[f"ds_{k}"] = 1 if k == dataset_id else 0
    return Xo

# -------------------------- build dataset --------------------------
rng = np.random.default_rng(3)
N_PER = 800

datasets = [
    (0, 0, 0, "TM: all off (156 GHz)"),
    (1, 1, 0, "TM: top on (148 GHz)"),
    (2, 0, 1, "TM: bottom on (148 GHz)"),
    (3, 1, 1, "TM: both on (142 GHz)"),
]

X_blocks = []
for dsid, top, bot, _ in datasets:
    Xg = sample_geometry(N_PER, rng)
    Xd = add_tokens_tm_only(Xg, top, bot, dataset_id=dsid, n_datasets=len(datasets))
    X_blocks.append(Xd)

X_all = pd.concat(X_blocks, ignore_index=True)

# ------------------------ Label generation ------------------------
BASE_TM = 156.0
_GEOM_REFS = dict(
    bottom_gap_mm=0.085, bottom_r_out_mm=0.25, bottom_r_in_mm=0.16,
    substrate_thickness_mm=0.254, epsilon_r=3.66, top_L2_mm=0.085, top_L3_mm=0.060
)

def _geom_linear_np(df):
    return ( 80.0*(df["bottom_gap_mm"].values - _GEOM_REFS["bottom_gap_mm"])
           - 90.0*(df["bottom_r_out_mm"].values - _GEOM_REFS["bottom_r_out_mm"])
           + 60.0*(df["bottom_r_in_mm"].values  - _GEOM_REFS["bottom_r_in_mm"])
           - 40.0*(df["substrate_thickness_mm"].values - _GEOM_REFS["substrate_thickness_mm"])
           -  3.0*(df["epsilon_r"].values       - _GEOM_REFS["epsilon_r"])
           - 50.0*(df["top_L2_mm"].values       - _GEOM_REFS["top_L2_mm"])
           + 40.0*(df["top_L3_mm"].values       - _GEOM_REFS["top_L3_mm"]) )

def f0_phys_full_np(df):
    top  = df["d_top_on"].astype(float).values
    bot  = df["d_bot_on"].astype(float).values
    both = top * bot
    anchor = BASE_TM - 7.0*top - 6.0*bot + 1.5*both
    lin = _geom_linear_np(df)
    if not USE_NONLINEAR_INTERACTIONS:
        return anchor + lin

    # nonlinear + interactions
    gap = df["bottom_gap_mm"].values
    L2  = df["top_L2_mm"].values
    L3  = df["top_L3_mm"].values
    r_o = df["bottom_r_out_mm"].values
    r_i = df["bottom_r_in_mm"].values
    er  = df["epsilon_r"].values
    h   = df["substrate_thickness_mm"].values

    inter = (
        - 6.0*top * (L2/(gap+0.05))
        - 4.0*bot * (L3/(gap+0.05))
        + 3.0*both * (r_o - r_i)
        - 2.0*both * (er - 3.66)
        + 2.5*np.sin(8.0*(r_o - 0.25))
        - 2.0*np.tanh(20.0*(gap - 0.085))
        + 1.5*(h - 0.254)*(L2 + L3)
    )
    return anchor + lin + inter

def build_labels(X_train_df, X_df, rng=np.random.default_rng(0)):
    f_full = f0_phys_full_np(X_df)
    # heteroscedastic noise
    if HETEROSCEDASTIC_NOISE:
        gap = X_df["bottom_gap_mm"].values
        r_o = X_df["bottom_r_out_mm"].values
        sigma = 0.10 + 0.18/(gap+0.02) + 0.12*np.clip(0.26 - r_o, 0, None)
    else:
        sigma = 0.12
    eps = rng.normal(0.0, sigma, size=len(X_df))
    return f_full + eps

# ------------------------------ split ------------------------------
if SHIFTED_TEST:
    test_mask = (X_all["epsilon_r"] > 3.70) | ((X_all["bottom_gap_mm"] < 0.075) & (X_all["substrate_thickness_mm"] > 0.26))
    X_test = X_all[test_mask].reset_index(drop=True)
    X_train = X_all[~test_mask].reset_index(drop=True)
else:
    X_train, X_test = train_test_split(X_all, test_size=0.25, random_state=0, shuffle=True)

y_train = build_labels(X_train, X_train, rng=np.random.default_rng(10))
y_test  = build_labels(X_train, X_test,  rng=np.random.default_rng(11))

# ----------------------- helpers -----------------------
COLS = list(X_train.columns); idx = {c:i for i,c in enumerate(COLS)}
# Physics-informed monotonicity signs for each feature:
# +1 means prediction should increase with the feature,
# -1 means prediction should decrease, 0 = unconstrained.
MONO_SIGNS = {
    "bottom_gap_mm": +1,
    "bottom_r_out_mm": -1,
    "bottom_r_in_mm": +1,
    "substrate_thickness_mm": -1,
    "epsilon_r": -1,
    "top_L2_mm": -1,
    "top_L3_mm": +1,
}

# Ensure every remaining column has a default (no constraint)
for c in COLS:
    if c not in MONO_SIGNS:
        MONO_SIGNS[c] = 0

geom_feats  = ["bottom_gap_mm","bottom_r_out_mm","bottom_r_in_mm",
               "substrate_thickness_mm","epsilon_r","top_L2_mm","top_L3_mm"]

def state_code_from_df(df):
    top = df["d_top_on"].astype(int).values
    bot = df["d_bot_on"].astype(int).values
    return top + 2*bot  # 0=off,1=top,2=bot,3=both

# Per-state TRAIN stats
state_codes_train = state_code_from_df(X_train)
state_stats = {}
state_medians = {}
for s in [0,1,2,3]:
    mask = (state_codes_train == s)
    yt = np.asarray(y_train)[mask]
    if mask.any():
        state_stats[s]  = {"mean": yt.mean(), "std": yt.std() if yt.std()>1e-9 else 1.0}
        state_medians[s]= float(np.median(yt))
    else:
        state_stats[s]  = {"mean": float(np.mean(y_train)), "std": float(np.std(y_train))+1e-9}
        state_medians[s]= float(np.median(y_train))

def baseline_from_df(df):
    sc = state_code_from_df(df)
    return np.array([state_medians[int(s)] for s in sc], dtype=float)

# ----------------------- Standardization -----------------------
mu_np  = X_train.mean().values
std_np = X_train.std().replace(0, 1e-9).values
def std_np_fn(df): return (df.values - mu_np)/std_np

mu_X  = torch.tensor(mu_np, dtype=torch.float32)
std_X = torch.tensor(std_np, dtype=torch.float32)
def to_std(x_raw): return (x_raw - mu_X.to(x_raw.device)) / std_X.to(x_raw.device)

# ------------------------- Datasets -------------------------
class RegrMoEDS(Dataset):
    def __init__(self, Xdf, y):
        self.x_raw = torch.tensor(Xdf.values, dtype=torch.float32)
        self.y     = torch.tensor(np.asarray(y), dtype=torch.float32)
        self.state = torch.tensor(state_code_from_df(Xdf), dtype=torch.long)
        # normalized target per state
        yn = []
        for i, s in enumerate(self.state.numpy()):
            m, sd = state_stats[int(s)]["mean"], state_stats[int(s)]["std"]
            yn.append( (float(self.y[i]) - m) / sd )
        self.y_norm = torch.tensor(yn, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        xr = self.x_raw[i]
        return to_std(xr), xr, self.y[i], self.y_norm[i], self.state[i]

ds_full_train = RegrMoEDS(X_train, y_train)
ds_test       = RegrMoEDS(X_test,  y_test)
tr_idx, va_idx = train_test_split(np.arange(len(ds_full_train)), test_size=0.2, random_state=42)
ds_tr, ds_va  = Subset(ds_full_train, tr_idx), Subset(ds_full_train, va_idx)
dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
dl_va = DataLoader(ds_va, batch_size=128, shuffle=False)
dl_te = DataLoader(ds_test, batch_size=128, shuffle=False)

# -------------------- ADD: ID test set (extra eval only; training unchanged) --------------------
# Draw an ID test split from the same distribution as X_train (does not alter training set)
X_id_train_tmp, X_test_ID = train_test_split(X_train, test_size=0.25, random_state=0, shuffle=True)
y_true_ID = build_labels(X_train, X_test_ID, rng=np.random.default_rng(12))
ds_id  = RegrMoEDS(X_test_ID, y_true_ID)
dl_id  = DataLoader(ds_id, batch_size=128, shuffle=False)

# ----------------------- Classification bins -----------------------
if FIXED_BINS:
    q_edges = np.array([140.0, 146.0, 150.0, 154.0, 160.0])
else:
    q_edges = np.quantile(y_train, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(float)
bin_labels = ["B1","B2","B3","B4"]

def to_codes(vals):
    bins = pd.cut(vals, bins=q_edges, labels=bin_labels, include_lowest=True, right=True)
    return pd.Categorical(bins, categories=bin_labels).codes

# ================================================================
# (rest of pipeline: Model A/B/C definitions, OOD gating, training loops,
# evaluation metrics, confusion matrices, and plots)
# ================================================================
# ------------------------- OOD (Mahalanobis) -------------------------
Xtr_std_np = (X_train.values - X_train.mean().values) / np.where(X_train.std().values==0, 1e-9, X_train.std().values)
mu_std_np  = Xtr_std_np.mean(axis=0, keepdims=True)
cov_np     = np.cov(Xtr_std_np, rowvar=False); p = cov_np.shape[0]
gamma=0.05; cov_shr = (1-gamma)*cov_np + gamma*np.eye(p)
inv_cov    = np.linalg.inv(cov_shr + 1e-6*np.eye(p))
d2_train   = np.sum((Xtr_std_np - mu_std_np) @ inv_cov * (Xtr_std_np - mu_std_np), axis=1)
TAU_D2     = float(np.percentile(d2_train, 95.0))
MU_STD     = torch.tensor(mu_std_np.squeeze(), dtype=torch.float32)
INV_COV    = torch.tensor(inv_cov, dtype=torch.float32)
def mahalanobis2(x_std):
    z = x_std - MU_STD.to(x_std.device)
    return torch.sum(z @ INV_COV.to(x_std.device) * z, dim=1)

# -------------------------- Model A: MoE NN (per-state heads) --------------------------
class StateMoE(nn.Module):
    def __init__(self, D, H=128, p=0.10):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(D, H), nn.ReLU(), nn.Dropout(p),
            nn.Linear(H, H), nn.ReLU(), nn.Dropout(p)
        )
        self.heads = nn.ModuleList([nn.Linear(H,1) for _ in range(4)])  # 0..3
    def forward(self, x_std, state_idx):
        h = self.trunk(x_std)
        # predict all heads then gather by state
        all_pred = torch.cat([head(h) for head in self.heads], dim=1)  # [B,4]
        sel = all_pred.gather(1, state_idx.view(-1,1)).squeeze(1)      # normalized y_hat
        return sel, all_pred  # return also for optional regularization

huber = nn.SmoothL1Loss()
LAMBDA_MONO, LAMBDA_LC = 0.02, 0.003

def lc_cv(f0_pred, Xb, eps=1e-9):
    r_out = Xb[:, idx["bottom_r_out_mm"]]
    r_in  = Xb[:, idx["bottom_r_in_mm"]]
    er    = Xb[:, idx["epsilon_r"]]
    gap   = Xb[:, idx["bottom_gap_mm"]]
    L2    = Xb[:, idx["top_L2_mm"]]
    L3    = Xb[:, idx["top_L3_mm"]]
    L_eff = (r_out + r_in)/2.0
    C_eff = er / (gap + 0.5*L2 + 0.5*L3 + eps)
    k = f0_pred * torch.sqrt(torch.clamp(L_eff*C_eff, min=1e-12))
    return torch.var(k)/(torch.mean(k)**2 + 1e-12)

def mono_penalty(x_std, modelA):
    x = x_std.clone().detach().requires_grad_(True)
    # use state-agnostic gradient by averaging heads
    h = modelA.trunk(x)
    preds_all = torch.cat([head(h) for head in modelA.heads], dim=1)  # [B,4]
    f = preds_all.mean(dim=1)  # normalized space, but gradients w.r.t x are fine
    g = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
    pen = 0.0
    for c, sgn in MONO_SIGNS.items():
        j = idx[c]
        pen = pen + torch.relu(-sgn * g[:, j]).mean()
    return pen

def unnormalize_y(norm_pred, state_idx_np):
    y_hat = np.empty_like(norm_pred, dtype=float)
    for i, s in enumerate(state_idx_np.astype(int)):
        m, sd = state_stats[s]["mean"], state_stats[s]["std"]
        y_hat[i] = norm_pred[i]*sd + m
    return y_hat

# ------------------------------ Train Model A ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mu_X, std_X, MU_STD, INV_COV = [t.to(device) for t in (mu_X, std_X, MU_STD, INV_COV)]
modelA = StateMoE(D=len(COLS)).to(device)
optA = optim.Adam(modelA.parameters(), lr=1e-3, weight_decay=1e-4)
schedA = optim.lr_scheduler.ReduceLROnPlateau(optA, mode='min', factor=0.5, patience=8)

def eval_val_bias_A():
    modelA.eval()
    losses=[]; preds=[]; truths=[]; states=[]
    with torch.no_grad():
        for x_std,x_raw,y,y_norm,s in dl_va:
            x_std,x_raw,y,y_norm,s = x_std.to(device), x_raw.to(device), y.to(device), y_norm.to(device), s.to(device)
            yhat_norm,_ = modelA(x_std, s)
            # use unnormalized predictions for bias calc
            y_hat = unnormalize_y(yhat_norm.detach().cpu().numpy(), s.detach().cpu().numpy())
            losses.append(huber(torch.tensor(y_hat), y.cpu()).item()*len(y))
            preds.append(y_hat); truths.append(y.cpu().numpy()); states.append(s.cpu().numpy())
    preds = np.concatenate(preds); truths=np.concatenate(truths)
    bias = np.median(truths - preds)
    return sum(losses)/max(len(ds_va),1), float(bias)

best=float("inf"); best_state=None
EPOCHS=120
for ep in range(1, EPOCHS+1):
    modelA.train()
    for x_std,x_raw,y,y_norm,s in dl_tr:
        x_std,x_raw,y,y_norm,s = x_std.to(device), x_raw.to(device), y.to(device), y_norm.to(device), s.to(device)
        yhat_norm,_ = modelA(x_std, s)
        # compute unnormalized y_hat for penalties
        y_hat = torch.empty_like(yhat_norm)
        for ss in range(4):
            mask = (s==ss)
            if mask.any():
                m = state_stats[ss]["mean"]; sd = state_stats[ss]["std"]
                y_hat[mask] = yhat_norm[mask]*sd + m
        data = huber(yhat_norm, y_norm)   # loss in normalized space
        lc   = lc_cv(y_hat, x_raw)
        mono = mono_penalty(x_std, modelA)
        loss = data + LAMBDA_LC*lc + LAMBDA_MONO*mono
        optA.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(modelA.parameters(), 1.0)
        optA.step()
    vloss, vbias = eval_val_bias_A()
    schedA.step(vloss)
    if vloss < best - 1e-4:
        best = vloss
        best_state = {"model": {k:v.detach().cpu().clone() for k,v in modelA.state_dict().items()},
                      "bias":  vbias}
    if ep % 10 == 0:
        print(f"[A ep {ep:3d}] val_loss={vloss:.4f}  bias={vbias:+.3f} GHz")

VAL_BIAS_A = 0.0
if best_state:
    modelA.load_state_dict(best_state["model"])
    VAL_BIAS_A = best_state["bias"]
print(f"[Model A VAL bias] b = {VAL_BIAS_A:+.3f} GHz")

# ---------------------- Model B: XGBoost with monotone constraints ----------------------
Xtr_df = X_train.iloc[tr_idx].reset_index(drop=True)
Xva_df = X_train.iloc[va_idx].reset_index(drop=True)
ytr = np.asarray(y_train)[tr_idx]
yva = np.asarray(y_train)[va_idx]
Xte_df = X_test.reset_index(drop=True)

# monotone constraints map (+1 means increasing feature -> prediction increases; -1 decreasing)
mono_map = {
    "bottom_gap_mm": +1,  "bottom_r_out_mm": -1, "bottom_r_in_mm": +1,
    "substrate_thickness_mm": -1, "epsilon_r": -1, "top_L2_mm": -1, "top_L3_mm": +1
}
# XGBoost expects a string like "(+1, -1, ...)" in feature order
mono_vec = [mono_map.get(c, 0) for c in COLS]

def fit_xgb_with_constraints():
    dtrain = xgb.DMatrix(std_np_fn(Xtr_df), label=ytr, feature_names=COLS)
    dvalid = xgb.DMatrix(std_np_fn(Xva_df), label=yva, feature_names=COLS)

    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        eta=0.05,
        max_depth=8,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        monotone_constraints="(" + ",".join(str(v) for v in mono_vec) + ")"
    )

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    def predict_fn(df):
        return booster.predict(xgb.DMatrix(std_np_fn(df), feature_names=COLS),
                               iteration_range=(0, booster.best_iteration + 1))
    return predict_fn, booster

# train XGBoost once; keep name for plots/tables
gbm_predict, xgb_booster = fit_xgb_with_constraints()
gbm_name = "xgboost"

# VAL bias for B (median bias correction, consistent with A/C)
y_va_hat_B = gbm_predict(Xva_df)
VAL_BIAS_B = float(np.median(yva - y_va_hat_B))

# ---------------------- Learned OOD Gate (fit on VAL) ----------------------
# Gate blends model prediction with per-state median baseline
def fit_learned_gate_on_val(model_preds, Xval_df, yval, d2_val):
    base_val = baseline_from_df(Xval_df)
    a_grid   = np.linspace(0.2, 5.0, 16)     # slope
    b_grid   = np.linspace(-5.0, 5.0, 21)    # bias
    tau_grid = np.linspace(np.percentile(d2_train, 80), np.percentile(d2_train, 99.5), 16)
    best = (np.inf, 1.0, 0.0, float(TAU_D2))
    for a in a_grid:
        for b in b_grid:
            for tau in tau_grid:
                w = 1.0 / (1.0 + np.exp(-(d2_val - tau)*a + b))
                y_blend = (1 - w)*model_preds + w*base_val
                rmse = np.sqrt(np.mean((y_blend - yval)**2))
                if rmse < best[0]:
                    best = (rmse, a, b, tau)
    _, a_best, b_best, tau_best = best
    return a_best, b_best, tau_best

# collect VAL features to fit the gate for B
def collect_val_for_gate_B(predict_fn):
    xstds=[]; yval=[]
    with torch.no_grad():
        for x_std, x_raw, y, y_norm, s in dl_va:
            xstds.append(x_std.cpu().numpy())
            yval.append(y.cpu().numpy())
    xstd  = np.concatenate(xstds)
    yval  = np.concatenate(yval)
    d2_val = np.sum((xstd - MU_STD.cpu().numpy()) @ inv_cov * (xstd - MU_STD.cpu().numpy()), axis=1)
    Xva_df_local = Xva_df.reset_index(drop=True)
    B_pred = predict_fn(Xva_df_local) + VAL_BIAS_B
    return B_pred, Xva_df_local, yval, d2_val

B_val_pred, Xva_df_local, yval_array, d2_val = collect_val_for_gate_B(gbm_predict)
aB, bB, tB = fit_learned_gate_on_val(B_val_pred, Xva_df_local, yval_array, d2_val)
print(f"[Gate B] a={aB:.3f} b={bB:.3f} tau={tB:.3f}")

# ------------------------------ Evaluate Model B on TEST ------------------------------

def collect_val_for_gate_A(modelA):
    modelA.eval()
    predsA, yval, xstds = [], [], []
    with torch.no_grad():
        for x_std, x_raw, y, y_norm, s in dl_va:
            x_std, s = x_std.to(device), s.to(device)
            yhat_norm, _ = modelA(x_std, s)
            y_hat = unnormalize_y(yhat_norm.cpu().numpy(), s.cpu().numpy())
            predsA.append(y_hat); yval.append(y.cpu().numpy()); xstds.append(x_std.cpu().numpy())
    A_pred = np.concatenate(predsA) + VAL_BIAS_A
    yval = np.concatenate(yval)
    xstd = np.concatenate(xstds)
    d2_val = np.sum((xstd - MU_STD.cpu().numpy()) @ inv_cov * (xstd - MU_STD.cpu().numpy()), axis=1)
    return A_pred, yval, d2_val

A_val_pred, yval_A, d2_val_A = collect_val_for_gate_A(modelA)
aA, bA, tA = fit_learned_gate_on_val(A_val_pred, Xva_df.reset_index(drop=True), yval_A, d2_val_A)


# ------------------------------ Evaluate on TEST ------------------------------
def metrics_reg(name, p, t):
    mae = np.mean(np.abs(p-t))
    rmse= np.sqrt(np.mean((p-t)**2))
    r2  = 1 - np.sum((p-t)**2)/np.sum((t - t.mean())**2)
    print(f"{name}: MAE={mae:.3f} GHz  RMSE={rmse:.3f} GHz  R2={r2:.3f}")
    return mae, rmse, r2

# Model A raw predictions on TEST
modelA.eval()
preds_A=[]; d2_A=[]; states_A=[]; raw_rows=[]
with torch.no_grad():
    for x_std,x_raw,y,y_norm,s in dl_te:
        x_std,x_raw,y,y_norm,s = x_std.to(device), x_raw.to(device), y.to(device), y_norm.to(device), s.to(device)
        yhat_norm,_ = modelA(x_std, s)
        y_hat = unnormalize_y(yhat_norm.cpu().numpy(), s.cpu().numpy())
        preds_A.append(y_hat); states_A.append(s.cpu().numpy())
        d2_A.append(mahalanobis2(x_std).cpu().numpy()); raw_rows.append(x_raw.cpu().numpy())

y_pred_A_raw = np.concatenate(preds_A) + VAL_BIAS_A
d2_all   = np.concatenate(d2_A)
Xte_raw  = np.concatenate(raw_rows, axis=0)
Xte_df   = pd.DataFrame(Xte_raw, columns=COLS)


# Model B raw on TEST
y_pred_B_raw = gbm_predict(Xte_df) + VAL_BIAS_B

# Data baselines
y_fallback_test = baseline_from_df(Xte_df)

# apply learned gate on TEST
# (reuse d2_all already computed from A's loop, or recompute here if needed)


# Learned gates
wA = 1.0 / (1.0 + np.exp(-(d2_all - tA)*aA + bA))
wB = 1.0 / (1.0 + np.exp(-(d2_all - tB)*aB + bB))
y_guard_A  = (1 - wA)*y_pred_A_raw + wA*y_fallback_test
y_guard_B = (1 - wB)*y_pred_B_raw + wB*y_fallback_test

y_true = np.asarray(y_test)

print("\n--- Test metrics (regression) ---")
_ = metrics_reg("A: MoE NN (raw)",             y_pred_A_raw, y_true)
_ = metrics_reg("A: MoE NN (learned OOD-blend)", y_guard_A,   y_true)
_ = metrics_reg(f"B: {gbm_name} (raw)",          y_pred_B_raw, y_true)
_ = metrics_reg(f"B: {gbm_name} (learned OOD-blend)", y_guard_B, y_true)
print(f"[Sandbox] Test OOD fraction (by τ_train95) = {(d2_all>TAU_D2).mean()*100:.1f}%")

# ----------------------- TRAIN quantile bins (for all models) -----------------------
q_edges = np.quantile(y_train, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(float)
for k in range(1, len(q_edges)):
    if q_edges[k] <= q_edges[k-1]:
        q_edges[k] = q_edges[k-1] + 1e-6
q_edges[0]  -= 1e-6
q_edges[-1] += 1e-6
bin_labels = ["Q1","Q2","Q3","Q4"]

def to_codes(vals):
    bins = pd.cut(vals, bins=q_edges, labels=bin_labels, include_lowest=True, right=True)
    return pd.Categorical(bins, categories=bin_labels).codes

def classify_and_scores(name, yhat, y_true):
    y_true_codes = to_codes(y_true)
    y_codes = to_codes(yhat)
    m = (y_codes>=0) & (y_true_codes>=0)
    acc = accuracy_score(y_true_codes[m], y_codes[m])
    f1  = f1_score(y_true_codes[m], y_codes[m], average="macro")
    print(f"\n--- {name} ---")
    print(classification_report(y_true_codes[m], y_codes[m],
                                labels=[0,1,2,3], target_names=bin_labels, zero_division=0))
    return y_codes, acc, f1, m

y_codes_A, acc_A, f1_A, mask_A = classify_and_scores("Model A (MoE→bins)", y_guard_A, y_true)
y_codes_B, acc_B, f1_B, mask_B = classify_and_scores(f"Model B ({gbm_name}→bins)", y_guard_B, y_true)

# -------------------- Confusion matrices & metrics plot --------------------
def plot_conf(ax, yhat_codes, y_true, mask, title):
    yref = to_codes(y_true)[mask]
    ConfusionMatrixDisplay.from_predictions(
        yref, yhat_codes[mask], labels=[0,1,2,3], display_labels=bin_labels,
        cmap="Blues", values_format="d", ax=ax
    )
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
plot_conf(axes[0], y_codes_A, y_true, mask_A, "A: MoE NN → bins")
plot_conf(axes[1], y_codes_B, y_true, mask_B, f"B: {gbm_name} → bins")
plt.tight_layout(); plt.show()

# Side-by-side bar chart of Accuracy and Macro-F1
labels_plot = ["A: MoE NN", f"B: {gbm_name}"]
accs = [acc_A, acc_B]; f1s  = [f1_A,  f1_B]
x = np.arange(len(labels_plot)); wbar = 0.35
fig, ax = plt.subplots(figsize=(6.8, 3.8))
ax.bar(x - wbar/2, accs, width=wbar, label="Accuracy")
ax.bar(x + wbar/2, f1s,  width=wbar, label="Macro F1")
ax.set_xticks(x); ax.set_xticklabels(labels_plot, rotation=0)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title("Classification metrics comparison")
ax.legend(); plt.tight_layout(); plt.show()

# ===================== ADD: Evaluate A/B on ID test set =====================
# Predict with A on ID
modelA.eval()
preds_A_id=[]; d2_A_id=[]; raw_rows_id=[]
with torch.no_grad():
    for x_std,x_raw,y,y_norm,s in dl_id:
        x_std,x_raw,y,y_norm,s = x_std.to(device), x_raw.to(device), y.to(device), y_norm.to(device), s.to(device)
        yhat_norm,_ = modelA(x_std, s)
        y_hat = unnormalize_y(yhat_norm.cpu().numpy(), s.cpu().numpy())
        preds_A_id.append(y_hat)
        d2_A_id.append(mahalanobis2(x_std).cpu().numpy())
        raw_rows_id.append(x_raw.cpu().numpy())

y_pred_A_raw_ID = np.concatenate(preds_A_id) + VAL_BIAS_A
d2_id   = np.concatenate(d2_A_id)
X_id_np = np.concatenate(raw_rows_id, axis=0)
X_id_df = pd.DataFrame(X_id_np, columns=COLS)

# Predict with B on ID
y_pred_B_raw_ID = gbm_predict(X_id_df) + VAL_BIAS_B

# Baseline and learned gates (reuse aA,bA,tA and aB,bB,tB)
y_base_ID = baseline_from_df(X_id_df)
wA_id = 1.0 / (1.0 + np.exp(-(d2_id - tA)*aA + bA))
wB_id = 1.0 / (1.0 + np.exp(-(d2_id - tB)*aB + bB))
y_guard_A_ID = (1 - wA_id)*y_pred_A_raw_ID + wA_id*y_base_ID
y_guard_B_ID = (1 - wB_id)*y_pred_B_raw_ID + wB_id*y_base_ID

y_true_ID_np = np.asarray(y_true_ID)

print("\n--- ID Test metrics (regression) ---")
_ = metrics_reg("A: MoE NN (raw, ID)",               y_pred_A_raw_ID, y_true_ID_np)
_ = metrics_reg("A: MoE NN (learned OOD-blend, ID)", y_guard_A_ID,    y_true_ID_np)
_ = metrics_reg(f"B: {gbm_name} (raw, ID)",          y_pred_B_raw_ID, y_true_ID_np)
_ = metrics_reg(f"B: {gbm_name} (learned OOD-blend, ID)", y_guard_B_ID, y_true_ID_np)

# Use the same (already-defined) bin edges & helpers (TRAIN quantiles above)
y_codes_A_ID, acc_A_ID, f1_A_ID, mask_A_ID = classify_and_scores("Model A (MoE→bins, ID)", y_guard_A_ID, y_true_ID_np)
y_codes_B_ID, acc_B_ID, f1_B_ID, mask_B_ID = classify_and_scores(f"Model B ({gbm_name}→bins, ID)", y_guard_B_ID, y_true_ID_np)

# Confusion matrices for ID
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
plot_conf(axes[0], y_codes_A_ID, y_true_ID_np, mask_A_ID, "A: MoE NN → bins (ID)")
plot_conf(axes[1], y_codes_B_ID, y_true_ID_np, mask_B_ID, f"B: {gbm_name} → bins (ID)")
plt.tight_layout(); plt.show()

# Side-by-side bars for ID
labels_plot = ["A: MoE NN (ID)", f"B: {gbm_name} (ID)"]
accs = [acc_A_ID, acc_B_ID]; f1s  = [f1_A_ID,  f1_B_ID]
x = np.arange(len(labels_plot)); wbar = 0.35
fig, ax = plt.subplots(figsize=(6.8, 3.8))
ax.bar(x - wbar/2, accs, width=wbar, label="Accuracy")
ax.bar(x + wbar/2, f1s,  width=wbar, label="Macro F1")
ax.set_xticks(x); ax.set_xticklabels(labels_plot, rotation=0)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title("Classification metrics (ID)")
ax.legend(); plt.tight_layout(); plt.show()

# ===================== time-schedule simulation & plot ======================
def one_row_df(geom_row, top_on, bot_on):
    d = {c: float(geom_row[c]) for c in geom_feats}
    d["d_top_on"]  = int(bool(top_on))
    d["d_bot_on"]  = int(bool(bot_on))
    d["d_both_on"] = d["d_top_on"] * d["d_bot_on"]
    for k in range(len([c for c in COLS if c.startswith("ds_")])):
        d[f"ds_{k}"] = 1.0 if k == 0 else 0.0
    for c in COLS:
        if c not in d and c not in geom_feats:
            d[c] = 0.0
    return pd.DataFrame([d], columns=COLS)

geom_med = X_train[geom_feats].median()
schedule = [
    (0,   0,0, "off"),
    (5,   1,0, "top"),
    (10,  0,1, "bot"),
    (15,  1,1, "both"),
    (20,  0,0, "off"),
    (25,  1,0, "top"),
    (30,  1,1, "both"),
]

rows = []
for t, top, bot, lab in schedule:
    df1 = one_row_df(geom_med, top, bot)
    df1["t"]   = t
    df1["state"]= lab
    rows.append(df1)
timeline = pd.concat(rows, ignore_index=True)

# MoE A on timeline
x_raw_np = timeline[COLS].values.astype(np.float32)
x_raw_t  = torch.tensor(x_raw_np, dtype=torch.float32, device=device)
x_std_t  = (x_raw_t - mu_X) / std_X
state_idx_t = torch.tensor(state_code_from_df(timeline), dtype=torch.long, device=device)
with torch.no_grad():
    yA_norm_t,_ = modelA(x_std_t, state_idx_t)
yA_raw = unnormalize_y(yA_norm_t.cpu().numpy(), state_idx_t.cpu().numpy()) + VAL_BIAS_A

# B on timeline
# B on timeline
yB_raw = gbm_predict(timeline[COLS]) + VAL_BIAS_B


# d^2 and learned gates
with torch.no_grad():
    d2_t = mahalanobis2(x_std_t).cpu().numpy()
wA_t = 1.0 / (1.0 + np.exp(-(d2_t - tA)*aA + bA))
wB_t = 1.0 / (1.0 + np.exp(-(d2_t - tB)*aB + bB))
base_t = baseline_from_df(timeline)
yA = (1 - wA_t)*yA_raw + wA_t*base_t
yB = (1 - wB_t)*yB_raw + wB_t*base_t

# ---------------- Plot frequency vs time ----------------
ts = timeline["t"].values
plt.figure(figsize=(7.6,4.0))
plt.plot(ts, yA, marker="o", label="Model A (MoE NN, learned blend)")
plt.plot(ts, yB, marker="o", label=f"Model B ({gbm_name}, learned blend)")
plt.plot(ts, base_t, marker="o", linestyle="--", label="Data baseline (per state)")
for t, _, _, _ in schedule:
    plt.axvline(t, linestyle="--", linewidth=0.7, alpha=0.5)
plt.xlabel("Time (arb.)"); plt.ylabel("Predicted stop-band f0 (GHz)")
plt.title("Predicted frequency vs time under diode switching (no physics baseline)")
plt.legend(); plt.tight_layout(); plt.show()

# ========================= Model C: Monotone Neural Additive Model (mNAM) =========================
# Idea: y_hat = sum_j g_j(x_j) + state_head[state] with g_j monotone per MONO_SIGNS.
# Monotonicity is enforced by parameterizing weights as softplus(...) >= 0, plus an output sign flip if needed.

class Monotone1D(nn.Module):
    """Small 1D monotone subnetwork: x -> softplus(W2 * softplus(W1 * x + b1) + b2)
       All weights are constrained >=0 via softplus; output sign enforced outside."""
    def __init__(self, hidden=16):
        super().__init__()
        self.w1 = nn.Parameter(torch.zeros(1, hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.zeros(hidden, 1))
        self.b2 = nn.Parameter(torch.zeros(1))
    def forward(self, x):  # x: [B,1]
        # positive weights via softplus
        h = torch.nn.functional.softplus(x @ torch.nn.functional.softplus(self.w1) + self.b1)
        out = torch.nn.functional.softplus(h @ torch.nn.functional.softplus(self.w2) + self.b2)
        return out.squeeze(1)  # [B]

class MNAM(nn.Module):
    def __init__(self, feature_names, mono_signs, state_count=4, hidden=16, dropout_p=0.0):
        super().__init__()
        self.feature_names = feature_names
        self.mono_signs = mono_signs
        self.f_layers = nn.ModuleDict({c: Monotone1D(hidden=hidden) for c in feature_names})
        # Per-feature scale/shift to help conditioning (learned, non-negative scale)
        self.f_scale = nn.ParameterDict({c: nn.Parameter(torch.ones(1)) for c in feature_names})
        self.f_shift = nn.ParameterDict({c: nn.Parameter(torch.zeros(1)) for c in feature_names})
        # Per-state linear head in normalized target space (like MoE heads but scalar)
        self.state_weight = nn.Parameter(torch.zeros(state_count))  # gain on sum
        self.state_bias   = nn.Parameter(torch.zeros(state_count))  # bias
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x_std, x_raw, state_idx):
        # x_std: standardized inputs; x_raw: unstandardized (for LC penalty); state_idx: [B]
        # Build additive sum of monotone 1D subnetworks
        s = 0.0
        for j, c in enumerate(self.feature_names):
            xj = x_std[:, j:j+1]  # standardized feature j
            gj = self.f_layers[c](xj * torch.nn.functional.softplus(self.f_scale[c]) + self.f_shift[c])
            # Enforce signed monotonicity: +1 increasing, -1 decreasing
            sign = float(self.mono_signs.get(c, 0))
            if sign == -1:
                gj = -gj
            # (sign==0 -> leave unconstrained; but our construction is monotone-increasing, so treat 0 as +1)
            s = s + gj
        s = self.dropout(s)
        # State-specific affine in normalized space
        gain = self.state_weight[state_idx]
        bias = self.state_bias[state_idx]
        y_norm_hat = gain * s + bias  # normalized y_hat (per-state target norm)
        return y_norm_hat

# ----------------------- Train Model C -----------------------
modelC = MNAM(feature_names=COLS, mono_signs=MONO_SIGNS, state_count=4, hidden=24, dropout_p=0.05).to(device)
optC = optim.Adam(modelC.parameters(), lr=1e-3, weight_decay=5e-5)
schedC = optim.lr_scheduler.ReduceLROnPlateau(optC, mode='min', factor=0.5, patience=8)
huber = nn.SmoothL1Loss()

def unnorm_from_state_norm(y_norm_pred, state_idx_tensor):
    # identical logic to unnormalize_y but on torch tensors
    y_hat = torch.empty_like(y_norm_pred)
    for ss in range(4):
        m = state_stats[ss]["mean"]; sd = state_stats[ss]["std"]
        mask = (state_idx_tensor == ss)
        if mask.any():
            y_hat[mask] = y_norm_pred[mask]*sd + m
    return y_hat

def eval_val_bias_C():
    modelC.eval()
    losses=[]; preds=[]; truths=[]; states=[]
    with torch.no_grad():
        for x_std,x_raw,y,y_norm,s in dl_va:
            x_std,x_raw,y,y_norm,s = x_std.to(device), x_raw.to(device), y.to(device), y_norm.to(device), s.to(device)
            yhat_norm = modelC(x_std, x_raw, s)
            y_hat = unnorm_from_state_norm(yhat_norm, s)
            losses.append(huber(y_hat, y).item()*len(y))
            preds.append(y_hat.cpu().numpy()); truths.append(y.cpu().numpy()); states.append(s.cpu().numpy())
    preds = np.concatenate(preds); truths=np.concatenate(truths)
    bias = np.median(truths - preds)
    return sum(losses)/max(len(ds_va),1), float(bias)

bestC=float("inf"); bestC_state=None
EPOCHS_C = 120
LAMBDA_LC_C = 0.003  # reuse LC invariance penalty (helps calibration)
for ep in range(1, EPOCHS_C+1):
    modelC.train()
    for x_std,x_raw,y,y_norm,s in dl_tr:
        x_std,x_raw,y,y_norm,s = x_std.to(device), x_raw.to(device), y.to(device), y_norm.to(device), s.to(device)
        yhat_norm = modelC(x_std, x_raw, s)
        # penalties use *unnormalized* predictions
        y_hat = unnorm_from_state_norm(yhat_norm, s)
        loss_data = huber(yhat_norm, y_norm)
        loss_lc   = lc_cv(y_hat, x_raw)
        loss = loss_data + LAMBDA_LC_C*loss_lc
        optC.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(modelC.parameters(), 1.0)
        optC.step()
    vloss, vbias = eval_val_bias_C()
    schedC.step(vloss)
    if vloss < bestC - 1e-4:
        bestC = vloss
        bestC_state = {"model": {k:v.detach().cpu().clone() for k,v in modelC.state_dict().items()},
                       "bias":  vbias}
    if ep % 10 == 0:
        print(f"[C ep {ep:3d}] val_loss={vloss:.4f}  bias={vbias:+.3f} GHz")

VAL_BIAS_C = 0.0
if bestC_state:
    modelC.load_state_dict(bestC_state["model"])
    VAL_BIAS_C = bestC_state["bias"]
print(f"[Model C VAL bias] b = {VAL_BIAS_C:+.3f} GHz")

# ----------------------- Evaluate Model C on TEST -----------------------
modelC.eval()
preds_C=[]; d2_C=[]; raw_rows_C=[]; states_C=[]
with torch.no_grad():
    for x_std,x_raw,y,y_norm,s in dl_te:
        x_std,x_raw,y,y_norm,s = x_std.to(device), x_raw.to(device), y.to(device), y_norm.to(device), s.to(device)
        yhat_norm = modelC(x_std, x_raw, s)
        y_hat = unnorm_from_state_norm(yhat_norm, s)
        preds_C.append(y_hat.cpu().numpy()); states_C.append(s.cpu().numpy())
        d2_C.append(mahalanobis2(x_std).cpu().numpy()); raw_rows_C.append(x_raw.cpu().numpy())

y_pred_C_raw = np.concatenate(preds_C) + VAL_BIAS_C
d2_C_all = np.concatenate(d2_C)
Xte_raw_C  = np.concatenate(raw_rows_C, axis=0)
Xte_df_C   = pd.DataFrame(Xte_raw_C, columns=COLS)

# Learned gate for Model C (fit on same VAL routine)
def collect_val_for_gate_C(modelC):
    modelC.eval()
    predsC=[]; yval=[]; xstds=[]
    with torch.no_grad():
        for x_std,x_raw,y,y_norm,s in dl_va:
            x_std,x_raw,y,y_norm,s = x_std.to(device), x_raw.to(device), y.to(device), y_norm.to(device), s.to(device)
            yhat_norm = modelC(x_std, x_raw, s)
            y_hat = unnorm_from_state_norm(yhat_norm, s).cpu().numpy()
            predsC.append(y_hat); yval.append(y.cpu().numpy()); xstds.append(x_std.cpu().numpy())
    C_pred = np.concatenate(predsC); yval = np.concatenate(yval); xstd = np.concatenate(xstds)
    d2_val = np.sum((xstd - MU_STD.cpu().numpy()) @ inv_cov * (xstd - MU_STD.cpu().numpy()), axis=1)
    return C_pred, yval, d2_val

C_val_pred, yval_C, d2_val_C = collect_val_for_gate_C(modelC)
aC,bC,tC = fit_learned_gate_on_val(C_val_pred, Xva_df_local, yval_C, d2_val_C)
print(f"[Gate C] a={aC:.3f} b={bC:.3f} tau={tC:.3f}")

# Apply learned OOD blend on TEST
y_fallback_test = baseline_from_df(Xte_df_C)
wC = 1.0 / (1.0 + np.exp(-(d2_C_all - tC)*aC + bC))
y_guard_C = (1 - wC)*y_pred_C_raw + wC*y_fallback_test

print("\n--- Test metrics (regression) ---")
_ = metrics_reg("C: mNAM (raw)",               y_pred_C_raw, y_true)
_ = metrics_reg("C: mNAM (learned OOD-blend)",  y_guard_C,    y_true)

# ----------------------- Classification (bins) -----------------------
y_codes_C, acc_C, f1_C, mask_C = classify_and_scores("Model C (mNAM→bins)", y_guard_C, y_true)

# ----------------------- Confusion matrix & Bars (append to your plots) -----------------------
# Add confusion matrix for C
fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.0))
plot_conf(ax, y_codes_C, y_true, mask_C, "C: mNAM → bins")
plt.tight_layout(); plt.show()

# Extend the side-by-side bar chart (Accuracy & Macro-F1)
labels_plot = ["A: MoE NN", f"B: {gbm_name}", "C: mNAM"]
accs = [acc_A, acc_B, acc_C]; f1s  = [f1_A,  f1_B,  f1_C]
x = np.arange(len(labels_plot)); wbar = 0.25
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.bar(x - wbar, accs, width=wbar, label="Accuracy")
ax.bar(x,        f1s,  width=wbar, label="Macro F1")
ax.set_xticks(x); ax.set_xticklabels(labels_plot)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title("Classification metrics comparison")
ax.legend(); plt.tight_layout(); plt.show()

# ----------------------- (Optional) timeline overlay for C -----------------------
with torch.no_grad():
    x_raw_np = timeline[COLS].values.astype(np.float32)
    x_raw_t  = torch.tensor(x_raw_np, dtype=torch.float32, device=device)
    x_std_t  = (x_raw_t - mu_X) / std_X
    state_idx_t = torch.tensor(state_code_from_df(timeline), dtype=torch.long, device=device)
    yC_norm_t = modelC(x_std_t, x_raw_t, state_idx_t)
    yC_raw = unnorm_from_state_norm(yC_norm_t, state_idx_t).cpu().numpy() + VAL_BIAS_C
    d2_t = mahalanobis2(x_std_t).cpu().numpy()
wC_t = 1.0 / (1.0 + np.exp(-(d2_t - tC)*aC + bC))
yC = (1 - wC_t)*yC_raw + wC_t*baseline_from_df(timeline)

plt.figure(figsize=(7.6,4.0))
plt.plot(ts, yA, marker="o", label="Model A (blend)")
plt.plot(ts, yB, marker="o", label=f"Model B ({gbm_name}, blend)")
plt.plot(ts, yC, marker="o", label="Model C (mNAM, blend)")
plt.plot(ts, base_t, marker="o", linestyle="--", label="Data baseline")
for t, _, _, _ in schedule:
    plt.axvline(t, linestyle="--", linewidth=0.7, alpha=0.5)
plt.xlabel("Time (arb.)"); plt.ylabel("Predicted stop-band f0 (GHz)")
plt.title("Predicted frequency vs time (A/B/C)")
plt.legend(); plt.tight_layout(); plt.show()

# ===================== ADD: Evaluate Model C on ID =====================
modelC.eval()
preds_C_id=[]; d2_C_id=[]; raw_rows_C_id=[]
with torch.no_grad():
    for x_std,x_raw,y,y_norm,s in dl_id:
        x_std,x_raw,y,y_norm,s = x_std.to(device), x_raw.to(device), y.to(device), y_norm.to(device), s.to(device)
        yhat_norm = modelC(x_std, x_raw, s)
        y_hat = unnorm_from_state_norm(yhat_norm, s)
        preds_C_id.append(y_hat.cpu().numpy())
        d2_C_id.append(mahalanobis2(x_std).cpu().numpy())
        raw_rows_C_id.append(x_raw.cpu().numpy())

y_pred_C_raw_ID = np.concatenate(preds_C_id) + VAL_BIAS_C
d2_C_id_all = np.concatenate(d2_C_id)
X_id_np_C = np.concatenate(raw_rows_C_id, axis=0)
X_id_df_C = pd.DataFrame(X_id_np_C, columns=COLS)

y_base_ID = baseline_from_df(X_id_df_C)
wC_id = 1.0 / (1.0 + np.exp(-(d2_C_id_all - tC)*aC + bC))
y_guard_C_ID = (1 - wC_id)*y_pred_C_raw_ID + wC_id*y_base_ID

print("\n--- ID Test metrics (regression) ---")
_ = metrics_reg("C: mNAM (raw, ID)",               y_pred_C_raw_ID, y_true_ID_np)
_ = metrics_reg("C: mNAM (learned OOD-blend, ID)", y_guard_C_ID,    y_true_ID_np)

y_codes_C_ID, acc_C_ID, f1_C_ID, mask_C_ID = classify_and_scores("Model C (mNAM→bins, ID)", y_guard_C_ID, y_true_ID_np)

fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.0))
plot_conf(ax, y_codes_C_ID, y_true_ID_np, mask_C_ID, "C: mNAM → bins (ID)")
plt.tight_layout(); plt.show()

labels_plot = ["A: MoE NN (ID)", f"B: {gbm_name} (ID)", "C: mNAM (ID)"]
accs = [acc_A_ID, acc_B_ID, acc_C_ID]; f1s  = [f1_A_ID,  f1_B_ID,  f1_C_ID]
x = np.arange(len(labels_plot)); wbar = 0.25
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.bar(x - wbar, accs, width=wbar, label="Accuracy")
ax.bar(x,        f1s,  width=wbar, label="Macro F1")
ax.set_xticks(x); ax.set_xticklabels(labels_plot)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title("Classification metrics — ID")
ax.legend(); plt.tight_layout(); plt.show()
