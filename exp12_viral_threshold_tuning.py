"""
exp12_viral_threshold_tuning.py
================================
EXP-12: Tìm threshold tối ưu cho Viral Classifier một cách nghiêm ngặt.
QUY TRÌNH:
1. Sử dụng 5-fold Cross Validation trên tập TRAIN để lấy Out-of-Fold (OOF) probabilities.
2. Tìm threshold tốt nhất dựa trên F1-score của oof_probs (Dữ liệu Train).
3. Áp dụng threshold này vào tập TEST để báo cáo kết quả cuối cùng.
"""
import pandas as pd, numpy as np, pickle, os, matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, classification_report

THRESHOLD_VIEWS = 54_961
LEAK_COLS = ['video_id','channel_id','published_date','category_id', 'views_24h','views_48h','views_7d','is_viral_1m']
MODEL_DIR = 'models'
OUTPUT_DIR = 'evaluation_plots'

def run_rigorous_threshold_tuning():
    print("🚀 Khởi động quy trình Threshold Tuning nghiêm ngặt...")
    
    # --- 1. LOAD DỮ LIỆU ---
    tr = pd.read_csv('clean_train_time.csv', low_memory=False).fillna(0)
    te = pd.read_csv('clean_test_time.csv', low_memory=False).fillna(0)

    y_train = (tr['views_24h'] >= THRESHOLD_VIEWS).astype(int)
    y_test  = (te['views_24h'] >= THRESHOLD_VIEWS).astype(int)

    # Lấy danh sách feature từ model đã train tốt nhất để đồng bộ
    with open('models/xgb_viral_classifier_v3_time_best.pkl', 'rb') as f:
        artifact = pickle.load(f)
    feature_cols = artifact['feature_cols']
    best_params = artifact['best_params']

    X_train = tr[feature_cols].replace([np.inf,-np.inf],0).fillna(0)
    X_test  = te[feature_cols].replace([np.inf,-np.inf],0).fillna(0)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    # --- 2. TÌM THRESHOLD TRÊN TẬP TRAIN (SỬ DỤNG OOF) ---
    # Minh chứng: Tuyệt đối không dùng Xte hay y_test ở bước này
    print("  -> Bước 1: Tìm threshold tối ưu trên tập TRAIN (via 5-fold CV)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(Xtr))
    
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    best_params['scale_pos_weight'] = spw # Đảm bảo xử lý imbalanced

    for train_idx, val_idx in skf.split(Xtr, y_train):
        X_t, X_v = Xtr[train_idx], Xtr[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        clf = XGBClassifier(**best_params, eval_metric='logloss', n_jobs=-1, random_state=42)
        clf.fit(X_t, y_t)
        oof_probs[val_idx] = clf.predict_proba(X_v)[:, 1]

    # Tính toán F1 cho từng ngưỡng dựa trên oof_probs (toàn bộ là từ tập Train)
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probs)
    f1_scores = [2*p*r/(p+r) if (p+r)>0 else 0 for p, r in zip(precisions, recalls)]
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"  ✅ Best Threshold (from Train OOF): {best_threshold:.4f}")
    print(f"  ✅ Max F1 on Train (OOF): {f1_scores[best_idx]:.4f}")

    # --- 3. KIỂM CHỨNG TRÊN TẬP TEST (BLIND TEST) ---
    # Minh chứng: Bây giờ mới dùng tập Test để kiểm tra độ tin cậy của ngưỡng trên
    print("\n  -> Bước 2: Kiểm chứng ngưỡng trên tập TEST...")
    # Train lại trên toàn bộ tập Train
    final_clf = XGBClassifier(**best_params, eval_metric='logloss', n_jobs=-1, random_state=42)
    final_clf.fit(Xtr, y_train)
    test_probs = final_clf.predict_proba(Xte)[:, 1]

    y_pred_test = (test_probs >= best_threshold).astype(int)
    
    print("\n[BÁO CÁO KẾT QUẢ TẬP TEST]")
    print(classification_report(y_test, y_pred_test, target_names=['Not Viral', 'Top Viral']))

    # --- 4. LƯU KẾT QUẢ & VẼ HÌNH ---
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores[:-1], label='F1 (Train OOF)', color='blue')
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold ({best_threshold:.2f})')
    plt.xlabel('Probability Threshold')
    plt.ylabel('F1 Score')
    plt.title('Threshold Optimization on Training Set (via OOF)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/viral_threshold_tuning_rigorous.png", dpi=150)
    
    artifact['best_probability_threshold'] = best_threshold
    artifact['rigorous_tuning'] = True
    
    save_path = f'{MODEL_DIR}/xgb_viral_classifier_v3_time_optimized.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"\n✅ Đã lưu model và hình ảnh minh chứng sạch.")

if __name__ == "__main__":
    run_rigorous_threshold_tuning()
