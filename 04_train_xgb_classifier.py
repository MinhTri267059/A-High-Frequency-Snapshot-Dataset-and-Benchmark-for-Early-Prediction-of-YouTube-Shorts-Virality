"""
train_viral_v2.py
=================
Bài toán phân lớp: Top Viral vs Not Viral (dựa trên views_24h)

Pipeline:
1. Vẽ histogram views_24h → chọn threshold
2. Label ngay từ đầu: is_top_viral = 1 nếu views_24h >= THRESHOLD
3. Train XGBClassifier (time-based split)
4. Evaluate: F1-score, ROC-AUC, Classification Report
5. Lưu model + scaler + threshold
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve,
    roc_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
THRESHOLD_PERCENTILE = 90           # top 10% → "Top Viral"
OUTPUT_DIR = 'evaluation_plots'
MODEL_DIR  = 'models'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — Vẽ Histogram + Chọn Threshold
# ─────────────────────────────────────────────
def plot_histogram_and_choose_threshold(train_path, test_path):
    print("=" * 60)
    print("STEP 1: Analysing views_24h distribution (Train set only)...")
    print("=" * 60)

    # Strict Isolation: Calculate threshold using ONLY the training set
    df_train = pd.read_csv(train_path, low_memory=False)
    
    views = df_train['views_24h'].dropna()
    views = views[views >= 0]

    # Tính các percentiles
    percentiles = [50, 75, 80, 85, 90, 95, 99]
    pct_vals = {p: np.percentile(views, p) for p in percentiles}

    print(f"\nBasic stats (views_24h):")
    print(views.describe().to_string())
    print("\nPercentiles:")
    for p, v in pct_vals.items():
        print(f"  {p}th : {v:>12,.0f} views")

    threshold = pct_vals[THRESHOLD_PERCENTILE]
    print(f"\n→ Chosen threshold (top {100-THRESHOLD_PERCENTILE}%): {threshold:,.0f} views")
    n_viral = (views >= threshold).sum()
    pct_viral = 100.0 * n_viral / len(views)
    print(f"  Viral videos  : {n_viral:,} / {len(views):,} ({pct_viral:.1f}%)")

    # ── Plot ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Distribution of views_24h (YouTube Shorts)', fontsize=15, fontweight='bold')

    # (a) Raw histogram (capped at 99th pct for readability)
    cap = pct_vals[99]
    data_capped = views[views <= cap]
    axes[0].hist(data_capped, bins=120, color='#4C9BE8', edgecolor='white', linewidth=0.4)
    axes[0].set_title(f'Histogram (capped at 99th pct = {cap:,.0f})', fontsize=12)
    axes[0].set_xlabel('views_24h')
    axes[0].set_ylabel('Number of videos')
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    axes[0].axvline(threshold, color='#E84C4C', linestyle='--', linewidth=2,
                    label=f'Threshold {THRESHOLD_PERCENTILE}th pct\n({threshold:,.0f} views)')
    for p, color in zip([75, 85], ['#FFC300', '#FF5733']):
        axes[0].axvline(pct_vals[p], color=color, linestyle=':', linewidth=1.5,
                        label=f'{p}th pct ({pct_vals[p]:,.0f})')
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.4)

    # (b) Log-scale histogram (full range)
    axes[1].hist(views + 1, bins=120, color='#7B68EE', edgecolor='white', linewidth=0.4)
    axes[1].set_xscale('log')
    axes[1].set_title('Histogram (log scale, full range)', fontsize=12)
    axes[1].set_xlabel('views_24h (log scale)')
    axes[1].set_ylabel('Number of videos')
    axes[1].axvline(threshold, color='#E84C4C', linestyle='--', linewidth=2,
                    label=f'Threshold: {threshold:,.0f}')
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.4)

    # Annotate threshold band
    for ax in axes:
        ax.axvspan(threshold, ax.get_xlim()[1] if ax.get_xlim()[1] > threshold else threshold * 10,
                   alpha=0.08, color='red', label='_nolegend_')

    plt.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, 'views_24h_histogram_v2.png')
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"\n→ Saved histogram → {hist_path}")

    return threshold


# ─────────────────────────────────────────────
# STEP 2 — Prepare Data (label from scratch)
# ─────────────────────────────────────────────
def prepare_data(train_path, test_path, threshold):
    print("\n" + "=" * 60)
    print("STEP 2: Preparing data with new labels...")
    print("=" * 60)

    train_df = pd.read_csv(train_path, low_memory=False).fillna(0)
    test_df  = pd.read_csv(test_path,  low_memory=False).fillna(0)

    # Tạo label từ threshold (không dùng is_viral_1m cũ nữa)
    # Tạo label từ threshold (Strict isolation: > threshold)
    # Ghi chú: Sử dụng > thay vì >= để tránh tình trạng số lượng bằng nhau ở ngưỡng làm thay đổi phân phối lớp
    y_train = (train_df['views_24h'] > threshold).astype(int)
    y_test  = (test_df['views_24h']  > threshold).astype(int)

    print(f"  Train  → viral: {y_train.sum():,} / {len(y_train):,} ({100*y_train.mean():.1f}%)")
    print(f"  Test   → viral: {y_test.sum():,}  / {len(y_test):,} ({100*y_test.mean():.1f}%)")

    # Drop leak columns (các cột liên quan đến views, id, dates)
    leak_cols = ['video_id', 'channel_id', 'published_date', 'category_id',
                 'views_24h', 'views_48h', 'views_7d', 'is_viral_1m']

    drop_train = [c for c in leak_cols if c in train_df.columns]
    drop_test  = [c for c in leak_cols if c in test_df.columns]

    X_train = train_df.drop(columns=drop_train).replace([np.inf, -np.inf], 0).fillna(0)
    X_test  = test_df.drop(columns=drop_test).replace([np.inf, -np.inf], 0).fillna(0)

    # Align columns
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test  = X_test[common_cols]

    print(f"  Feature count : {len(common_cols)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, y_train, X_test_s, y_test, scaler, common_cols


# ─────────────────────────────────────────────
# STEP 3 — Train & Evaluate
# ─────────────────────────────────────────────
def train_and_evaluate(X_train, y_train, X_test, y_test, threshold):
    print("\n" + "=" * 60)
    print("STEP 3: Training XGBClassifier (Top Viral vs Not Viral)...")
    print("=" * 60)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,   # handle class imbalance
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
    )

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Metrics ──────────────────────────────
    f1  = f1_score(y_test, y_pred, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_pred,
                                target_names=['Not Viral', 'Top Viral'],
                                zero_division=0))
    print(f"  F1-score  (Top Viral)  : {f1:.4f}")
    print(f"  F1-score  (Macro avg)  : {f1_macro:.4f}")
    print(f"  ROC-AUC               : {auc:.4f}")

    # ── Plots ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Classifier Evaluation  |  Threshold: ≥{threshold:,.0f} views (top {100-THRESHOLD_PERCENTILE}%)',
                 fontsize=13, fontweight='bold')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Not Viral', 'Top Viral'])
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Confusion Matrix')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color='#4C9BE8', lw=2, label=f'AUC = {auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    axes[2].plot(rec, prec, color='#E84C4C', lw=2, label=f'PR-AUC = {pr_auc:.3f}')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    eval_path = os.path.join(OUTPUT_DIR, 'viral_classifier_v2_eval.png')
    plt.savefig(eval_path, dpi=150)
    plt.close()
    print(f"\n→ Saved evaluation plots → {eval_path}")

    return model, {'F1_viral': f1, 'F1_macro': f1_macro, 'AUC': auc}


# ─────────────────────────────────────────────
# STEP 4 — Save Artifacts
# ─────────────────────────────────────────────
def save_artifacts(model, scaler, threshold, feature_cols):
    print("\n" + "=" * 60)
    print("STEP 4: Saving model artifacts...")
    print("=" * 60)

    artifacts = {
        'model'       : model,
        'scaler'      : scaler,
        'threshold'   : threshold,
        'feature_cols': feature_cols,
        'label_info'  : {
            'target'     : 'is_top_viral',
            'definition' : f'views_24h > {threshold:,.0f}',
            'percentile' : THRESHOLD_PERCENTILE,
            'class_0'    : 'Not Viral',
            'class_1'    : 'Top Viral',
        }
    }

    save_path = os.path.join(MODEL_DIR, 'xgb_viral_classifier_v2.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(artifacts, f)

    print(f"  → Saved: {save_path}")
    print(f"  → Keys : {list(artifacts.keys())}")
    print(f"  → Label: views_24h > {threshold:,.0f}  →  Top Viral")

    return save_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    TRAIN = 'clean_train_time.csv'
    TEST  = 'clean_test_time.csv'

    # 1. Histogram + threshold
    threshold = plot_histogram_and_choose_threshold(TRAIN, TEST)

    # 2. Prepare (label from scratch)
    X_train, y_train, X_test, y_test, scaler, feature_cols = prepare_data(TRAIN, TEST, threshold)

    # 3. Train + evaluate
    model, metrics = train_and_evaluate(X_train, y_train, X_test, y_test, threshold)

    # 4. Save
    save_path = save_artifacts(model, scaler, threshold, feature_cols)

    print("\n" + "=" * 60)
    print("✅  DONE!")
    print(f"   Threshold : views_24h > {threshold:,.0f}  (top {100-THRESHOLD_PERCENTILE}%)")
    print(f"   F1-Viral  : {metrics['F1_viral']:.4f}")
    print(f"   F1-Macro  : {metrics['F1_macro']:.4f}")
    print(f"   ROC-AUC   : {metrics['AUC']:.4f}")
    print(f"   Model     : {save_path}")
    print("=" * 60)
