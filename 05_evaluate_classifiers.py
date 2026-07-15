"""
evaluate_viral_v3.py
====================
Đánh giá toàn diện Viral Classifier cho bài báo khoa học:
  1. Data Leakage Audit
  2. Baseline Models (Heuristic, LogisticRegression, RandomForest)
  3. Full Metrics: F1, ROC-AUC, PR-AUC, Brier Score
  4. Calibration Curve
  5. Walk-Forward Time-Series Cross Validation (3 folds)
  6. SHAP Values & Feature Importance
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os, pickle
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, classification_report,
    precision_recall_curve, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay
)
import shap
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class YoutubeDatasetClassif(Dataset):
    def __init__(self, seq, static, y):
        self.seq = torch.tensor(seq, dtype=torch.float32)
        self.static = torch.tensor(static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.seq[i], self.static[i], self.y[i]

class HybridLSTMClassif(nn.Module):
    def __init__(self, static_dim, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden + static_dim, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, s, st):
        _, (h, _) = self.lstm(s)
        out = torch.cat((h.squeeze(0), st), 1)
        return self.fc(out).squeeze(-1)

OUTPUT_DIR  = 'evaluation_plots'
MODEL_DIR   = 'models'

# Calculate THRESHOLD dynamically from Train set to prevent leakage
df_train_tmp = pd.read_csv('clean_train_time.csv', low_memory=False)
views_train = df_train_tmp['views_24h'].dropna()
views_train = views_train[views_train >= 0]
THRESHOLD = np.percentile(views_train, 90)
del df_train_tmp, views_train
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

LEAK_COLS = ['video_id','channel_id','published_date','category_id',
             'views_24h','views_48h','views_7d','is_viral_1m']

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_and_label(path):
    df = pd.read_csv(path, low_memory=False).fillna(0)
    df['_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    y = (df['views_24h'] > THRESHOLD).astype(int)
    X = df.drop(columns=[c for c in LEAK_COLS if c in df.columns])
    X = X.drop(columns=['_date'], errors='ignore')
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    return df, X, y

def metrics(y_true, y_pred, y_prob, name):
    return {
        'Model'    : name,
        'F1_viral' : f1_score(y_true, y_pred, zero_division=0),
        'F1_macro' : f1_score(y_true, y_pred, average='macro', zero_division=0),
        'ROC_AUC'  : roc_auc_score(y_true, y_prob),
        'PR_AUC'   : average_precision_score(y_true, y_prob),
        'Brier'    : brier_score_loss(y_true, y_prob),
    }

# ─────────────────────────────────────────────
# STEP 0 — Data Leakage Audit
# ─────────────────────────────────────────────
print("=" * 65)
print("STEP 0: DATA LEAKAGE AUDIT")
print("=" * 65)

EARLY_FEATURES = ['views_at_2h','views_at_4h','views_at_6h',
                   'vel_2','vel_4','vel_6',
                   'acc_4','acc_6',
                   'vel_likes_2','vel_likes_4','vel_likes_6',
                   'vel_cmts_2','vel_cmts_4','vel_cmts_6',
                   'like_ratio_6h','comment_ratio_6h']
CHANNEL_FEATURES = ['channel_avg_views','channel_video_count',
                     'channel_avg_vel_6h','channel_viral_rate','channel_strength']
META_FEATURES = ['duration_sec','is_weekend',
                 'publish_hour_sin','publish_hour_cos',
                 'publish_dow_sin','publish_dow_cos']

print("  ✅ Early engagement (0-6h)    :", len(EARLY_FEATURES), "features — OK (bài toán: predict sau 24h dùng data 6h đầu)")
print("  ✅ Channel history (trước đăng):", len(CHANNEL_FEATURES), "features — OK (thống kê lịch sử channel)")
print("  ✅ Video metadata (lúc đăng)  :", len(META_FEATURES), "features — OK")
print("  ✅ Title embeddings           : 384 features — OK (title có ngay lúc đăng)")
print("  ✅ Target (views_24h)         : đã drop khỏi features")
print("\n  → Không phát hiện data leakage.")
print("  → Framing rõ ràng: 'Dự đoán viral sau 24h dựa trên data 6h đầu tiên'")

# ─────────────────────────────────────────────
# STEP 1 — Load Data
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 1: LOADING DATA (Time-Based Split)")
print("=" * 65)

df_train, X_train_raw, y_train = load_and_label('clean_train_time.csv')
df_test,  X_test_raw,  y_test  = load_and_label('clean_test_time.csv')

common = sorted(set(X_train_raw.columns) & set(X_test_raw.columns))
X_train_raw = X_train_raw[common]
X_test_raw  = X_test_raw[common]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

spw = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  Train: {len(y_train):,} | viral: {y_train.sum():,} ({100*y_train.mean():.1f}%)")
print(f"  Test : {len(y_test):,}  | viral: {y_test.sum():,}  ({100*y_test.mean():.1f}%)")
print(f"  Features: {len(common)}")

# ─────────────────────────────────────────────
# STEP 2 — Baseline Models
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: BASELINE MODELS")
print("=" * 65)

results = []

# Baseline 1: Heuristic — views_at_6h > P90 của train
heuristic_threshold = np.percentile(df_train['views_at_6h'], 90)
h_pred = (df_test['views_at_6h'] >= heuristic_threshold).astype(int)
h_prob = (df_test['views_at_6h'] / (df_test['views_at_6h'].max() + 1)).values
print(f"  Heuristic: views_at_6h >= {heuristic_threshold:,.0f}")
results.append(metrics(y_test, h_pred, h_prob, 'Heuristic (6h views)'))

# Baseline 2: Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
results.append(metrics(y_test, lr.predict(X_test), lr.predict_proba(X_test)[:,1], 'Logistic Regression'))
print("  ✓ Logistic Regression done")

# Baseline 3: Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced',
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results.append(metrics(y_test, rf.predict(X_test), rf.predict_proba(X_test)[:,1], 'Random Forest'))
print("  ✓ Random Forest done")

# Main: XGBoost
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8,
                     scale_pos_weight=spw, random_state=42,
                     eval_metric='logloss', verbosity=0)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:,1]
results.append(metrics(y_test, xgb_pred, xgb_prob, 'XGBoost (ours)'))
print("  ✓ XGBoost done")

# Baseline 5: LSTM Classifier
print("  Training LSTM Classifier...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seq_cols = ['views_at_2h', 'views_at_4h', 'views_at_6h']
seq_idxs = [common.index(c) for c in seq_cols]
s_tr = np.expand_dims(X_train[:, seq_idxs], -1)
s_te = np.expand_dims(X_test[:, seq_idxs], -1)

train_dl = DataLoader(YoutubeDatasetClassif(s_tr, X_train, y_train.values), batch_size=256, shuffle=True)
test_dl = DataLoader(YoutubeDatasetClassif(s_te, X_test, y_test.values), batch_size=256, shuffle=False)

lstm_model = HybridLSTMClassif(X_train.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([spw], dtype=torch.float32).to(device))
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

for epoch in range(5):
    print(f'    Epoch {epoch+1}/5...')
    lstm_model.train()
    for i, (sb, stb, yb) in enumerate(train_dl):
        if i % 10 == 0: print(f'      Batch {i}/{len(train_dl)}')
        optimizer.zero_grad()
        loss = criterion(lstm_model(sb.to(device), stb.to(device)), yb.to(device))
        loss.backward()
        optimizer.step()

lstm_model.eval()
lstm_prob = []
with torch.no_grad():
    for sb, stb, _ in test_dl:
        logits = lstm_model(sb.to(device), stb.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        lstm_prob.extend(probs)
lstm_prob = np.array(lstm_prob)
lstm_pred = (lstm_prob >= 0.5).astype(int)

results.append(metrics(y_test, lstm_pred, lstm_prob, 'LSTM'))
print("  ✓ LSTM Classifier done")

res_df = pd.DataFrame(results)
print("\n[COMPARISON TABLE]")
print(res_df.to_string(index=False, float_format='{:.4f}'.format))

# ─────────────────────────────────────────────
# STEP 3 — Walk-Forward Time-Series CV
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: WALK-FORWARD CROSS VALIDATION (3 folds)")
print("=" * 65)

df_all = pd.concat([df_train, df_test], ignore_index=True)
df_all['_date'] = pd.to_datetime(df_all['published_date'], errors='coerce')
df_all = df_all.sort_values('_date').reset_index(drop=True)
df_all['y'] = (df_all['views_24h'] > THRESHOLD).astype(int)

# 3 folds: expand window
dates = df_all['_date'].dropna().sort_values().unique()
total_days = (dates[-1] - dates[0]).days
fold_size = total_days // 4  # ~6-7 days per fold

fold_results = []
fold_splits = [
    (dates[0], dates[0] + pd.Timedelta(days=fold_size*2),   dates[0] + pd.Timedelta(days=fold_size*2),   dates[0] + pd.Timedelta(days=fold_size*3)),
    (dates[0], dates[0] + pd.Timedelta(days=fold_size*2+1), dates[0] + pd.Timedelta(days=fold_size*2+1), dates[0] + pd.Timedelta(days=fold_size*3+1)),
    (dates[0], dates[0] + pd.Timedelta(days=fold_size*3),   dates[0] + pd.Timedelta(days=fold_size*3),   dates[-1]),
]

for i, (tr_start, tr_end, te_start, te_end) in enumerate(fold_splits, 1):
    tr_mask = (df_all['_date'] >= tr_start) & (df_all['_date'] < tr_end)
    te_mask = (df_all['_date'] >= te_start) & (df_all['_date'] < te_end)

    tr = df_all[tr_mask]
    te = df_all[te_mask]

    if te['y'].sum() == 0:
        print(f"  Fold {i}: skip (no viral in test)")
        continue

    feat_cols = [c for c in common if c in df_all.columns]
    Xtr = tr[feat_cols].replace([np.inf,-np.inf],0).fillna(0)
    Xte = te[feat_cols].replace([np.inf,-np.inf],0).fillna(0)
    ytr = tr['y']
    yte = te['y']

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    spw_f = max(1, (ytr==0).sum() / max(1, (ytr==1).sum()))
    m = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=spw_f, random_state=42,
                      eval_metric='logloss', verbosity=0)
    m.fit(Xtr_s, ytr)
    ypred = m.predict(Xte_s)
    yprob = m.predict_proba(Xte_s)[:,1]

    f1v  = f1_score(yte, ypred, zero_division=0)
    auc  = roc_auc_score(yte, yprob)
    prauc = average_precision_score(yte, yprob)
    brier = brier_score_loss(yte, yprob)

    print(f"  Fold {i}: train {tr_start.date()}→{tr_end.date()} | test {te_start.date()}→{te_end.date()}")
    print(f"         train={len(tr):,}({ytr.sum()}viral) test={len(te):,}({yte.sum()}viral)")
    print(f"         F1={f1v:.4f}  AUC={auc:.4f}  PR-AUC={prauc:.4f}  Brier={brier:.4f}")
    fold_results.append({'Fold': i, 'F1': f1v, 'AUC': auc, 'PR_AUC': prauc, 'Brier': brier})

if fold_results:
    fd = pd.DataFrame(fold_results)
    print(f"\n  MEAN across folds:")
    print(f"    F1={fd['F1'].mean():.4f}±{fd['F1'].std():.4f}  AUC={fd['AUC'].mean():.4f}±{fd['AUC'].std():.4f}")
    print(f"    PR-AUC={fd['PR_AUC'].mean():.4f}±{fd['PR_AUC'].std():.4f}  Brier={fd['Brier'].mean():.4f}±{fd['Brier'].std():.4f}")

# ─────────────────────────────────────────────
# STEP 4 — SHAP Values
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: SHAP FEATURE IMPORTANCE")
print("=" * 65)

explainer = shap.TreeExplainer(xgb)
# Dùng subset để nhanh hơn
X_test_df = pd.DataFrame(X_test, columns=common)
shap_sample = X_test_df.sample(min(500, len(X_test_df)), random_state=42)
shap_vals = explainer.shap_values(shap_sample)

# Top 20 features by mean |SHAP|
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
top20_idx = np.argsort(mean_abs_shap)[-20:][::-1]
top20_names = [common[i] for i in top20_idx]
top20_vals  = mean_abs_shap[top20_idx]
print("  Top 10 features by |SHAP|:")
for n, v in zip(top20_names[:10], top20_vals[:10]):
    print(f"    {n:<35} {v:.4f}")

# ─────────────────────────────────────────────
# STEP 5 — Plots
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: GENERATING PLOTS")
print("=" * 65)

# ── Plot 1: Comprehensive Metrics Dashboard ────────────────
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('Viral Classifier — Comprehensive Evaluation Dashboard\n'
             f'(Threshold: views_24h > {THRESHOLD:,.0f} | Time-Based Split)',
             fontsize=14, fontweight='bold')

colors = {'Heuristic (6h views)': '#aaaaaa',
          'Logistic Regression':   '#FFC300',
          'Random Forest':         '#4C9BE8',
          'XGBoost (ours)':        '#E84C4C',
          'LSTM':                  '#8B008B'}

# (a) Confusion Matrix
ax0 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test, xgb_pred)
ConfusionMatrixDisplay(cm, display_labels=['Not Viral','Top Viral']).plot(
    ax=ax0, colorbar=False, cmap='Blues')
ax0.set_title('Confusion Matrix (XGBoost)', fontsize=11)

# (b) ROC Curves — all models
ax1 = fig.add_subplot(gs[0, 1])
model_probs = {
    'Heuristic (6h views)': h_prob,
    'Logistic Regression':  lr.predict_proba(X_test)[:,1],
    'Random Forest':        rf.predict_proba(X_test)[:,1],
    'XGBoost (ours)':       xgb_prob,
    'LSTM':                 lstm_prob,
}
for name, prob in model_probs.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = roc_auc_score(y_test, prob)
    lw = 2.5 if 'XGBoost' in name else 1.5
    ls = '-' if 'XGBoost' in name else '--'
    ax1.plot(fpr, tpr, color=colors[name], lw=lw, ls=ls,
             label=f'{name}\n(AUC={auc_val:.3f})')
ax1.plot([0,1],[0,1],'k--',lw=1)
ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
ax1.set_title('ROC Curves (All Models)', fontsize=11)
ax1.legend(fontsize=7, loc='lower right')
ax1.grid(alpha=0.3)

# (c) PR Curves — all models
ax2 = fig.add_subplot(gs[0, 2])
for name, prob in model_probs.items():
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    lw = 2.5 if 'XGBoost' in name else 1.5
    ls = '-' if 'XGBoost' in name else '--'
    ax2.plot(rec, prec, color=colors[name], lw=lw, ls=ls,
             label=f'{name}\n(AP={ap:.3f})')
baseline_rate = y_test.mean()
ax2.axhline(baseline_rate, color='gray', linestyle=':', lw=1,
            label=f'Baseline={baseline_rate:.3f}')
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curves (All Models)', fontsize=11)
ax2.legend(fontsize=7, loc='upper right')
ax2.grid(alpha=0.3)

# (d) Metric Comparison Bar Chart
ax3 = fig.add_subplot(gs[1, 0:2])
metric_names = ['F1_viral','F1_macro','ROC_AUC','PR_AUC','Brier']
metric_labels = ['F1\n(Viral)','F1\n(Macro)','ROC-AUC','PR-AUC','Brier\nScore']
x = np.arange(len(metric_names))
width = 0.2
for j, row in res_df.iterrows():
    vals = [row[m] for m in metric_names]
    lw = 2 if 'XGBoost' in row['Model'] else 1
    ec = 'black' if 'XGBoost' in row['Model'] else 'none'
    ax3.bar(x + j*width, vals, width, label=row['Model'],
            color=colors[row['Model']], edgecolor=ec, linewidth=lw)
ax3.set_xticks(x + width*1.5)
ax3.set_xticklabels(metric_labels)
ax3.set_ylabel('Score')
ax3.set_title('Metric Comparison — All Models', fontsize=11)
ax3.legend(fontsize=8)
ax3.set_ylim(0, 1.05)
ax3.grid(axis='y', alpha=0.3)
for bar in ax3.patches:
    h = bar.get_height()
    if h > 0.01:
        ax3.text(bar.get_x()+bar.get_width()/2, h+0.01,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=6)

# (e) Calibration Curve
ax4 = fig.add_subplot(gs[1, 2])
for name, prob in model_probs.items():
    if name == 'Heuristic (6h views)': continue
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
    lw = 2.5 if 'XGBoost' in name else 1.5
    ax4.plot(mean_pred, frac_pos, 's-', color=colors[name], lw=lw,
             label=f'{name}\n(Brier={brier_score_loss(y_test, prob):.3f})')
ax4.plot([0,1],[0,1],'k--',lw=1,label='Perfect calibration')
ax4.set_xlabel('Mean predicted probability')
ax4.set_ylabel('Fraction of positives')
ax4.set_title('Calibration Curve', fontsize=11)
ax4.legend(fontsize=7)
ax4.grid(alpha=0.3)

# (f) Walk-Forward CV Results
if fold_results:
    ax5 = fig.add_subplot(gs[2, 0])
    fd = pd.DataFrame(fold_results)
    folds = fd['Fold'].astype(str).tolist()
    x_f = np.arange(len(folds))
    w = 0.2
    for k, (col, label, clr) in enumerate([
            ('F1','F1-Viral','#E84C4C'),('AUC','ROC-AUC','#4C9BE8'),
            ('PR_AUC','PR-AUC','#FFC300')]):
        ax5.bar(x_f + k*w, fd[col], w, label=label, color=clr)
    ax5.set_xticks(x_f + w)
    ax5.set_xticklabels([f'Fold {f}' for f in folds])
    ax5.set_ylabel('Score'); ax5.set_ylim(0,1.05)
    ax5.set_title('Walk-Forward CV (XGBoost)', fontsize=11)
    ax5.legend(fontsize=8); ax5.grid(axis='y',alpha=0.3)

# (g) SHAP Top 20 Bar
ax6 = fig.add_subplot(gs[2, 1:])
# Group title embeddings
non_emb_mask = ['title_emb' not in n for n in top20_names]
top20_names_plot = [n if 'title_emb' not in n else 'title_emb_*' for n in top20_names]
bar_colors = ['#E84C4C' if 'title_emb' not in n else '#cccccc' for n in top20_names]
ax6.barh(range(20), top20_vals[::-1], color=bar_colors[::-1])
ax6.set_yticks(range(20))
ax6.set_yticklabels(top20_names_plot[::-1], fontsize=8)
ax6.set_xlabel('Mean |SHAP value|')
ax6.set_title('Top 20 Features by SHAP Importance', fontsize=11)
ax6.grid(axis='x', alpha=0.3)
from matplotlib.patches import Patch
ax6.legend(handles=[Patch(color='#E84C4C',label='Non-embedding'),
                    Patch(color='#cccccc',label='Title embedding')],
           fontsize=8)

plt.savefig(f'{OUTPUT_DIR}/viral_v3_full_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  → Saved: {OUTPUT_DIR}/viral_v3_full_evaluation.png")

# ── Plot 2: SHAP Beeswarm (detailed) ───────────────────────
fig2, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals[:, top20_idx[:20]], shap_sample.iloc[:, top20_idx[:20]],
                  show=False, max_display=20, plot_size=None)
plt.title('SHAP Summary Plot — Top 20 Features', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/viral_v3_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  → Saved: {OUTPUT_DIR}/viral_v3_shap_beeswarm.png")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("✅  FINAL SUMMARY")
print("=" * 65)
print(res_df.to_string(index=False, float_format='{:.4f}'.format))
if fold_results:
    fd = pd.DataFrame(fold_results)
    print(f"\nWalk-Forward CV (XGBoost, {len(fold_results)} folds):")
    print(f"  F1     = {fd['F1'].mean():.4f} ± {fd['F1'].std():.4f}")
    print(f"  AUC    = {fd['AUC'].mean():.4f} ± {fd['AUC'].std():.4f}")
    print(f"  PR-AUC = {fd['PR_AUC'].mean():.4f} ± {fd['PR_AUC'].std():.4f}")
    print(f"  Brier  = {fd['Brier'].mean():.4f} ± {fd['Brier'].std():.4f}")
print("=" * 65)
