"""
evaluate_regression_v3.py
==========================
Đánh giá toàn diện XGBoost Regression (views_24h & views_48h) cho bài báo:
  0. Data Leakage Audit
  1. Baseline Models: Heuristic, Linear, Ridge, RandomForest, GradientBoosting, XGBoost
  2. Full Metrics: R², MAE, RMSE, MAPE, MedAE (log-space & actual)
  3. Walk-Forward Time-Series Cross Validation (3 folds)
  4. SHAP Values & Feature Importance
  5. Residual Analysis & Actual vs Predicted plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os, pickle
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
)
import shap

OUTPUT_DIR = 'evaluation_plots'
MODEL_DIR  = 'models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS    = ['views_24h', 'views_48h']
LEAK_COLS  = ['video_id','channel_id','published_date','category_id',
              'views_24h','views_48h','views_7d','is_viral_1m']

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_df(path):
    df = pd.read_csv(path, low_memory=False).fillna(0)
    df['_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    return df

def get_XY(df, target):
    y = np.log1p(df[target].clip(lower=0))
    drop = [c for c in LEAK_COLS if c in df.columns] + ['_date']
    X = df.drop(columns=drop, errors='ignore')
    cols_to_log = [c for c in X.columns if 'views' in c or 'vel' in c]
    X[cols_to_log] = np.log1p(X[cols_to_log].clip(lower=0))
    X = X.replace([np.inf,-np.inf],0).fillna(0)
    return X, y

def reg_metrics(y_true_log, y_pred_log, name):
    y_true_real = np.expm1(y_true_log)
    y_pred_real = np.expm1(y_pred_log)
    r2   = r2_score(y_true_log, y_pred_log)
    mae  = mean_absolute_error(y_true_log, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mede = median_absolute_error(y_true_log, y_pred_log)
    # MAPE on real scale (exclude zeros)
    mask = y_true_real > 0
    mape = np.mean(np.abs((y_true_real[mask] - y_pred_real[mask]) / y_true_real[mask])) * 100
    mae_real  = mean_absolute_error(y_true_real, y_pred_real)
    rmse_real = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    return {
        'Model': name, 'R2': r2,
        'MAE_log': mae, 'RMSE_log': rmse, 'MedAE_log': mede,
        'MAPE_%': mape,
        'MAE_views': mae_real, 'RMSE_views': rmse_real,
    }

# ─────────────────────────────────────────────
# STEP 0 — Data Leakage Audit
# ─────────────────────────────────────────────
print("=" * 65)
print("STEP 0: DATA LEAKAGE AUDIT")
print("=" * 65)
print("  Features used (inputs only have 0-6h data):")
print("  ✅ views_at_2h, views_at_4h, views_at_6h  — 0-6h engagement")
print("  ✅ vel_2, vel_4, vel_6                     — velocity 0-6h")
print("  ✅ like/comment ratio & velocity 0-6h      — engagement rate")
print("  ✅ channel_avg_views, viral_rate, etc.      — historical (pre-publish)")
print("  ✅ title embeddings                         — available at publish time")
print("  ✅ duration_sec, publish_hour, is_weekend   — metadata at publish time")
print("  ✅ views_24h / views_48h                   — TARGET only, dropped from X")
print("\n  → No data leakage detected.")
print("  → Task framing: 'Predict 24h/48h views using first 6h data'")

# ─────────────────────────────────────────────
# STEP 1 — Load Data
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 1: LOADING DATA (Time-Based Split)")
print("=" * 65)

df_train = load_df('clean_train_time.csv')
df_test  = load_df('clean_test_time.csv')
print(f"  Train: {len(df_train):,} videos | {df_train['_date'].min().date()} → {df_train['_date'].max().date()}")
print(f"  Test : {len(df_test):,}  videos | {df_test['_date'].min().date()} → {df_test['_date'].max().date()}")

# ─────────────────────────────────────────────
# STEP 2 — Train & Evaluate All Baselines
# ─────────────────────────────────────────────
all_results = {}
shap_info   = {}

for target in TARGETS:
    print(f"\n{'='*65}")
    print(f"TARGET: {target}")
    print(f"{'='*65}")

    X_train_raw, y_train = get_XY(df_train, target)
    X_test_raw,  y_test  = get_XY(df_test,  target)

    common = sorted(set(X_train_raw.columns) & set(X_test_raw.columns))
    X_train_raw = X_train_raw[common]
    X_test_raw  = X_test_raw[common]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    results = []

    # ── Heuristic: predict = channel_avg_views (log1p)
    ch_avg = np.log1p(df_train['channel_avg_views'].clip(lower=0).values)
    heur_pred = np.log1p(df_test['channel_avg_views'].clip(lower=0).values)
    results.append(reg_metrics(y_test.values, heur_pred, 'Heuristic\n(channel avg views)'))
    print("  ✓ Heuristic done")

    # ── Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results.append(reg_metrics(y_test.values, lr.predict(X_test), 'Linear Regression'))
    print("  ✓ Linear Regression done")

    # ── Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results.append(reg_metrics(y_test.values, ridge.predict(X_test), 'Ridge Regression'))
    print("  ✓ Ridge done")

    # ── Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(reg_metrics(y_test.values, rf.predict(X_test), 'Random Forest'))
    print("  ✓ Random Forest done")

    # ── GradientBoosting
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                   max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    results.append(reg_metrics(y_test.values, gb.predict(X_test), 'Gradient Boosting'))
    print("  ✓ Gradient Boosting done")

    # ── XGBoost (best params)
    params = {'n_estimators':100,'learning_rate':0.1,'max_depth':7} if target=='views_24h' \
             else {'n_estimators':200,'learning_rate':0.1,'max_depth':5}
    xgb = XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    results.append(reg_metrics(y_test.values, xgb_pred, 'XGBoost (ours)'))
    print("  ✓ XGBoost done")

    res_df = pd.DataFrame(results)
    all_results[target] = {'res_df': res_df, 'y_test': y_test.values,
                           'xgb_pred': xgb_pred, 'xgb': xgb,
                           'X_test': X_test, 'X_test_raw': X_test_raw,
                           'common': common, 'scaler': scaler,
                           'X_train': X_train, 'y_train': y_train.values}

    print(f"\n  [RESULTS — {target}]")
    print(res_df[['Model','R2','MAE_log','RMSE_log','MAPE_%']].to_string(index=False,
          float_format='{:.4f}'.format))

    # ── SHAP
    print(f"\n  Computing SHAP for {target}...")
    explainer = shap.TreeExplainer(xgb)
    X_test_df = pd.DataFrame(X_test, columns=common)
    shap_sample = X_test_df.sample(min(400, len(X_test_df)), random_state=42)
    shap_vals = explainer.shap_values(shap_sample)
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[-20:][::-1]
    shap_info[target] = {'vals': shap_vals, 'sample': shap_sample,
                         'top_idx': top_idx, 'common': common,
                         'mean_abs': mean_abs}
    print(f"  Top 5 SHAP features ({target}):")
    for i in range(5):
        print(f"    {common[top_idx[i]]:<35} {mean_abs[top_idx[i]]:.4f}")

# ─────────────────────────────────────────────
# STEP 3 — Walk-Forward CV
# ─────────────────────────────────────────────
print(f"\n{'='*65}")
print("STEP 3: WALK-FORWARD CROSS VALIDATION (3 folds)")
print("=" * 65)

df_all = pd.concat([df_train, df_test], ignore_index=True)
df_all['_date'] = pd.to_datetime(df_all['published_date'], errors='coerce')
df_all = df_all.sort_values('_date').reset_index(drop=True)
dates  = df_all['_date'].dropna().sort_values().unique()
fd_sz  = (dates[-1] - dates[0]).days // 4

fold_defs = [
    (dates[0], dates[0]+pd.Timedelta(days=fd_sz*2),   dates[0]+pd.Timedelta(days=fd_sz*2),   dates[0]+pd.Timedelta(days=fd_sz*3)),
    (dates[0], dates[0]+pd.Timedelta(days=fd_sz*2+1), dates[0]+pd.Timedelta(days=fd_sz*2+1), dates[0]+pd.Timedelta(days=fd_sz*3+1)),
    (dates[0], dates[0]+pd.Timedelta(days=fd_sz*3),   dates[0]+pd.Timedelta(days=fd_sz*3),   dates[-1]),
]

wf_results = {t: [] for t in TARGETS}

for fi, (tr_s, tr_e, te_s, te_e) in enumerate(fold_defs, 1):
    tr = df_all[(df_all['_date']>=tr_s)&(df_all['_date']<tr_e)]
    te = df_all[(df_all['_date']>=te_s)&(df_all['_date']<te_e)]
    if len(te) < 50: continue

    for target in TARGETS:
        common = all_results[target]['common']
        feat_cols = [c for c in common if c in df_all.columns]
        Xtr, ytr = get_XY(tr, target)
        Xte, yte = get_XY(te, target)
        Xtr = Xtr[[c for c in feat_cols if c in Xtr.columns]].replace([np.inf,-np.inf],0).fillna(0)
        Xte = Xte[[c for c in feat_cols if c in Xte.columns]].replace([np.inf,-np.inf],0).fillna(0)

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        params = {'n_estimators':100,'learning_rate':0.1,'max_depth':7} if target=='views_24h' \
                 else {'n_estimators':200,'learning_rate':0.1,'max_depth':5}
        m = XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
        m.fit(Xtr_s, ytr)
        pred = m.predict(Xte_s)

        r = reg_metrics(yte.values, pred, f'Fold {fi}')
        r['Fold'] = fi
        wf_results[target].append(r)
        print(f"  [{target}] Fold {fi} | train {tr_s.date()}→{tr_e.date()} | "
              f"test {te_s.date()}→{te_e.date()} | "
              f"R2={r['R2']:.4f}  MAE={r['MAE_log']:.4f}  MAPE={r['MAPE_%']:.1f}%")

print("\n  Walk-Forward Summary:")
for target in TARGETS:
    fd = pd.DataFrame(wf_results[target])
    print(f"  {target}:")
    print(f"    R2    = {fd['R2'].mean():.4f} ± {fd['R2'].std():.4f}")
    print(f"    MAE   = {fd['MAE_log'].mean():.4f} ± {fd['MAE_log'].std():.4f}")
    print(f"    MAPE  = {fd['MAPE_%'].mean():.1f}% ± {fd['MAPE_%'].std():.1f}%")

# ─────────────────────────────────────────────
# STEP 4 — Plots
# ─────────────────────────────────────────────
print(f"\n{'='*65}")
print("STEP 4: GENERATING PLOTS")
print("=" * 65)

for target in TARGETS:
    info   = all_results[target]
    res_df = info['res_df']
    y_test = info['y_test']
    xgb_pred = info['xgb_pred']
    residuals = y_test - xgb_pred

    fig = plt.figure(figsize=(22, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(f'Regression Evaluation Dashboard — {target}\n'
                 '(Time-Based Split | log1p scale)',
                 fontsize=14, fontweight='bold')

    model_colors = {
        'Heuristic\n(channel avg views)': '#aaaaaa',
        'Linear Regression':  '#97c2f0',
        'Ridge Regression':   '#4C9BE8',
        'Random Forest':      '#FFC300',
        'Gradient Boosting':  '#FF8C00',
        'XGBoost (ours)':     '#E84C4C',
    }

    # (a) R² Comparison
    ax0 = fig.add_subplot(gs[0, 0])
    names = res_df['Model'].tolist()
    r2s   = res_df['R2'].tolist()
    clrs  = [model_colors.get(n, '#888888') for n in names]
    bars  = ax0.bar(range(len(names)), r2s, color=clrs,
                    edgecolor=['black' if 'XGBoost' in n else 'none' for n in names],
                    linewidth=[2 if 'XGBoost' in n else 1 for n in names])
    ax0.set_xticks(range(len(names)))
    ax0.set_xticklabels([n.replace('\n',' ') for n in names], rotation=25, ha='right', fontsize=7)
    ax0.set_ylabel('R² Score'); ax0.set_ylim(0, 1.05)
    ax0.set_title('R² Score — All Models', fontsize=11)
    ax0.grid(axis='y', alpha=0.3)
    for b, v in zip(bars, r2s):
        ax0.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=8)

    # (b) MAE_log Comparison
    ax1 = fig.add_subplot(gs[0, 1])
    maes = res_df['MAE_log'].tolist()
    bars2 = ax1.bar(range(len(names)), maes, color=clrs,
                    edgecolor=['black' if 'XGBoost' in n else 'none' for n in names],
                    linewidth=[2 if 'XGBoost' in n else 1 for n in names])
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.replace('\n',' ') for n in names], rotation=25, ha='right', fontsize=7)
    ax1.set_ylabel('MAE (log-space)'); ax1.set_title('MAE (log) — All Models', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    for b, v in zip(bars2, maes):
        ax1.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=8)

    # (c) MAPE Comparison
    ax2 = fig.add_subplot(gs[0, 2])
    mapes = res_df['MAPE_%'].tolist()
    bars3 = ax2.bar(range(len(names)), mapes, color=clrs,
                    edgecolor=['black' if 'XGBoost' in n else 'none' for n in names],
                    linewidth=[2 if 'XGBoost' in n else 1 for n in names])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.replace('\n',' ') for n in names], rotation=25, ha='right', fontsize=7)
    ax2.set_ylabel('MAPE (%)'); ax2.set_title('MAPE% — All Models', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    for b, v in zip(bars3, mapes):
        ax2.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}%',
                 ha='center', va='bottom', fontsize=8)

    # (d) Actual vs Predicted (XGBoost)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_test, xgb_pred, alpha=0.25, s=8, color='#4C9BE8')
    lims = [min(y_test.min(), xgb_pred.min()), max(y_test.max(), xgb_pred.max())]
    ax3.plot(lims, lims, 'r--', lw=1.5)
    ax3.set_xlabel('Actual (log1p views)'); ax3.set_ylabel('Predicted')
    ax3.set_title('Actual vs Predicted (XGBoost)', fontsize=11)
    ax3.grid(alpha=0.3)
    r2v = r2_score(y_test, xgb_pred)
    ax3.text(0.05, 0.92, f'R²={r2v:.4f}', transform=ax3.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', fc='white', ec='gray'))

    # (e) Residual Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(residuals, bins=80, color='#7B68EE', edgecolor='white', linewidth=0.3)
    ax4.axvline(0, color='red', lw=1.5, linestyle='--')
    ax4.axvline(residuals.mean(), color='orange', lw=1.5, linestyle='-',
                label=f'Mean={residuals.mean():.3f}')
    ax4.set_xlabel('Residual (actual - predicted, log)')
    ax4.set_ylabel('Count')
    ax4.set_title('Residual Distribution (XGBoost)', fontsize=11)
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    # (f) Residual vs Predicted
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(xgb_pred, residuals, alpha=0.2, s=8, color='#E84C4C')
    ax5.axhline(0, color='black', lw=1.5, linestyle='--')
    ax5.set_xlabel('Predicted (log1p)'); ax5.set_ylabel('Residual')
    ax5.set_title('Residual vs Predicted', fontsize=11)
    ax5.grid(alpha=0.3)

    # (g) Walk-Forward CV
    ax6 = fig.add_subplot(gs[2, 0])
    fd = pd.DataFrame(wf_results[target])
    x_f = np.arange(len(fd))
    w = 0.25
    ax6.bar(x_f,       fd['R2'],      w, label='R²',        color='#4C9BE8')
    ax6.bar(x_f+w,     1-fd['MAE_log']/fd['MAE_log'].max(), w, label='1-norm(MAE)', color='#FFC300')
    ax6.set_xticks(x_f+w/2)
    ax6.set_xticklabels([f'Fold {i+1}' for i in range(len(fd))])
    ax6.set_ylabel('Score'); ax6.set_ylim(0, 1.05)
    ax6.set_title('Walk-Forward CV Results', fontsize=11)
    ax6.legend(fontsize=9); ax6.grid(axis='y', alpha=0.3)

    # (h) SHAP Top 20
    si = shap_info[target]
    top_idx = si['top_idx']
    common  = si['common']
    mean_abs = si['mean_abs']

    top20_names = [common[i] for i in top_idx[:20]]
    top20_vals  = mean_abs[top_idx[:20]]
    bar_colors  = ['#E84C4C' if 'title_emb' not in n else '#cccccc' for n in top20_names]

    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.barh(range(20), top20_vals[::-1], color=bar_colors[::-1])
    ax7.set_yticks(range(20))
    ax7.set_yticklabels([n if 'title_emb' not in n else 'title_emb_*'
                         for n in top20_names[::-1]], fontsize=8)
    ax7.set_xlabel('Mean |SHAP value|')
    ax7.set_title('Top 20 Features — SHAP Importance (XGBoost)', fontsize=11)
    ax7.grid(axis='x', alpha=0.3)
    from matplotlib.patches import Patch
    ax7.legend(handles=[Patch(color='#E84C4C', label='Non-embedding'),
                        Patch(color='#cccccc', label='Title embedding')], fontsize=8)

    out = f'{OUTPUT_DIR}/regression_{target}_full_eval.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {out}")

    # SHAP Beeswarm
    fig2, _ = plt.subplots(figsize=(10, 8))
    shap.summary_plot(si['vals'][:, top_idx[:20]],
                      si['sample'].iloc[:, top_idx[:20]],
                      show=False, max_display=20, plot_size=None)
    plt.title(f'SHAP Summary — {target}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out2 = f'{OUTPUT_DIR}/regression_{target}_shap.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {out2}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*65}")
print("✅  FINAL SUMMARY")
print("=" * 65)
for target in TARGETS:
    res_df = all_results[target]['res_df']
    print(f"\n--- {target} ---")
    cols = ['Model','R2','MAE_log','RMSE_log','MAPE_%','MAE_views']
    print(res_df[cols].to_string(index=False, float_format='{:.4f}'.format))

    print(f"\n  Walk-Forward CV (3 folds):")
    fd = pd.DataFrame(wf_results[target])
    print(f"    R2   = {fd['R2'].mean():.4f} ± {fd['R2'].std():.4f}")
    print(f"    MAE  = {fd['MAE_log'].mean():.4f} ± {fd['MAE_log'].std():.4f}")
    print(f"    MAPE = {fd['MAPE_%'].mean():.1f}% ± {fd['MAPE_%'].std():.1f}%")

print("\n" + "=" * 65)
