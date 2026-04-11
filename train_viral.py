import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

def prepare_data(train_path, test_path, target_col='is_viral_1m'):
    train_df = pd.read_csv(train_path).fillna(0)
    test_df = pd.read_csv(test_path).fillna(0)
    
    # Drop leak features (like target views columns)
    drop_cols = ['video_id', 'channel_id', 'published_date', 'category_id', 
                'views_24h', 'views_48h', 'views_7d', target_col]
    
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)
    y_test = test_df[target_col]
    
    # Standardize input to avoid XGBoost log scale warnings if data varies
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test

def train_and_eval(X_train, y_train, X_test, y_test, split_name):
    print(f"\n--- 🚀 ĐÀO TẠO PHÂN LOẠI VIRAL: {split_name} ---")
    
    # Kiểm tra phân phối tập Train
    train_counts = y_train.value_counts()
    print(f"Phân phối Train (0/1): {train_counts.get(0,0)} / {train_counts.get(1,0)}")
    
    if train_counts.get(1, 0) == 0:
        print("LỖI: Không có video Viral trong tập Train này!")
        return None
        
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else np.zeros(len(y_test))
    
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = np.nan
        
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n[BÁO CÁO PHÂN LOẠI]")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Lưu Model
    os.makedirs('models', exist_ok=True)
    model_name = "time_based" if "Time" in split_name else "channel_based"
    model_path = f'models/best_xgb_viral_{model_name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"-> Đã lưu mô hình tại: {model_path}")
    
    return {'Split': split_name, 'AUC': auc, 'F1': f1}

if __name__ == "__main__":
    results = []
    
    # 1. Tập Channel-Based
    try:
        X_tr, y_tr, X_te, y_te = prepare_data('clean_train_channel.csv', 'clean_test_channel.csv')
        res1 = train_and_eval(X_tr, y_tr, X_te, y_te, "Channel-Based Split")
        if res1: results.append(res1)
    except Exception as e:
        print(f"Lỗi khi xử lý Channel-based: {e}")
        
    # 2. Tập Time-Based
    try:
        X_tr, y_tr, X_te, y_te = prepare_data('clean_train_time.csv', 'clean_test_time.csv')
        res2 = train_and_eval(X_tr, y_tr, X_te, y_te, "Time-Based Split")
        if res2: results.append(res2)
    except Exception as e:
        print(f"Lỗi khi xử lý Time-based: {e}")
        
    # 3. So Sánh
    if results:
        compare_df = pd.DataFrame(results)
        print("\n🏆 TỔNG HỢP SO SÁNH PERFORMANCES (is_viral_1m):")
        print(compare_df.to_string(index=False))
