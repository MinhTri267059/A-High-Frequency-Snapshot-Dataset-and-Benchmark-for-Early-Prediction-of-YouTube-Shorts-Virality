"""
exp14_optuna_viral_classifier.py
==================================
EXP-14: Sử dụng Optuna (Bayesian Optimization) để tìm hyperparameters tốt nhất cho Viral Classifier.
QUY TRÌNH:
1. Optimization: Tìm tham số tốt nhất qua Cross-Validation trên tập TRAIN.
2. Final Training: Train lại model trên toàn bộ tập TRAIN với tham số tốt nhất.
3. Final Testing: Kiểm chứng cuối cùng trên tập TEST để báo cáo kết quả.
"""
import optuna
import pandas as pd, numpy as np, pickle, warnings
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score, classification_report, roc_auc_score

warnings.filterwarnings('ignore')

THRESHOLD = 54_961
LEAK_COLS = ['video_id','channel_id','published_date','category_id', 'views_24h','views_48h','views_7d','is_viral_1m']

def objective(trial):
    # Đọc dữ liệu Train bên trong objective để đảm bảo cô lập
    tr = pd.read_csv('clean_train_time.csv', low_memory=False).fillna(0)
    y_train = (tr['views_24h'] >= THRESHOLD).astype(int)
    X_train = tr.drop(columns=[c for c in LEAK_COLS if c in tr.columns]).replace([np.inf,-np.inf],0).fillna(0)
    common = sorted(X_train.columns)
    X_train = X_train[common]
    
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    spw = (y_train == 0).sum() / (y_train == 1).sum()

    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'scale_pos_weight': spw,
        'random_state': 42, 'eval_metric': 'logloss', 'n_jobs': -1
    }
    
    clf = XGBClassifier(**param)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # Tối ưu hóa dựa trên PR-AUC (Average Precision)
    score = cross_val_score(clf, Xtr, y_train, cv=cv, scoring='average_precision').mean()
    return score

def run_optuna_full_pipeline():
    print("🚀 BƯỚC 1: Bắt đầu tối ưu hóa Hyperparameters (Chỉ dùng tập TRAIN)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15) 
    
    print(f"\n✅ Tối ưu hóa xong! Best PR-AUC (CV): {study.best_value:.4f}")
    print(f"   Best Params: {study.best_params}")

    # --- BƯỚC 2: TEST LẠI TRÊN TẬP TEST ---
    print("\n🚀 BƯỚC 2: Kiểm chứng kết quả trên tập TEST (Blind Test)...")
    
    # Load cả Train và Test
    tr = pd.read_csv('clean_train_time.csv', low_memory=False).fillna(0)
    te = pd.read_csv('clean_test_time.csv', low_memory=False).fillna(0)
    
    y_train = (tr['views_24h'] >= THRESHOLD).astype(int)
    y_test  = (te['views_24h'] >= THRESHOLD).astype(int)
    
    # Đảm bảo feature cols đồng bộ
    X_train_raw = tr.drop(columns=[c for c in LEAK_COLS if c in tr.columns]).replace([np.inf,-np.inf],0).fillna(0)
    common = sorted(X_train_raw.columns)
    X_train_raw = X_train_raw[common]
    X_test_raw = te[common].replace([np.inf,-np.inf],0).fillna(0)
    
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train_raw)
    Xte = scaler.transform(X_test_raw)
    
    # Train model cuối cùng với tham số tốt nhất
    best_clf = XGBClassifier(**study.best_params, 
                             scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                             random_state=42, eval_metric='logloss', n_jobs=-1)
    best_clf.fit(Xtr, y_train)
    
    # Dự đoán trên tập Test
    y_prob_test = best_clf.predict_proba(Xte)[:, 1]
    y_pred_test = (y_prob_test >= 0.5).astype(int) # Dùng ngưỡng mặc định 0.5 trước khi tune threshold
    
    print("\n[BÁO CÁO KẾT QUẢ CUỐI CÙNG TRÊN TẬP TEST]")
    print(classification_report(y_test, y_pred_test, target_names=['Not Viral', 'Top Viral']))
    print(f"ROC-AUC trên tập Test: {roc_auc_score(y_test, y_prob_test):.4f}")
    print(f"PR-AUC (AP) trên tập Test: {average_precision_score(y_test, y_prob_test):.4f}")

    # Lưu best params
    with open('models/optuna_best_params_viral.pkl', 'wb') as f:
        pickle.dump(study.best_params, f)
    print("\n✅ Đã lưu Best Parameters và hoàn tất kiểm chứng.")

if __name__ == "__main__":
    run_optuna_full_pipeline()
