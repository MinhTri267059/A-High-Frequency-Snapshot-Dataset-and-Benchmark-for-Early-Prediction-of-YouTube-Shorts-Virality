import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import os

def prepare_data(train_path, test_path, target_col):
    train_df = pd.read_csv(train_path).fillna(0)
    test_df = pd.read_csv(test_path).fillna(0)
    
    # Base drop cols
    drop_cols = ['video_id', 'channel_id', 'published_date', 'category_id', 'is_viral_1m', 'views_24h', 'views_48h', 'views_7d']
    
    if target_col in train_df.columns:
        y_train = np.log1p(train_df[target_col].clip(lower=0))
        y_test = np.log1p(test_df[target_col].clip(lower=0))
    else:
        return None, None, None, None
        
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    
    # Log scale inputs
    cols_to_log = [c for c in X_train.columns if 'views' in c or 'vel' in c]
    X_train[cols_to_log] = np.log1p(X_train[cols_to_log].clip(lower=0))
    X_test[cols_to_log] = np.log1p(X_test[cols_to_log].clip(lower=0))
    
    X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)
    
    return X_train, y_train, X_test, y_test

def train_xgb(X_train, y_train, X_test, y_test, name, target):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return {'Target': target, 'Split': name, 'R2': r2, 'MAE': mae}

if __name__ == '__main__':
    targets = ['views_24h', 'views_48h']
    splits = {
        'Channel-Based': ('clean_train_channel.csv', 'clean_test_channel.csv'),
        'Time-Based': ('clean_train_time.csv', 'clean_test_time.csv')
    }
    
    results = []
    
    for target in targets:
        for split_name, (train_path, test_path) in splits.items():
            try:
                X_tr, y_tr, X_te, y_te = prepare_data(train_path, test_path, target)
                if X_tr is not None:
                    res = train_xgb(X_tr, y_tr, X_te, y_te, split_name, target)
                    results.append(res)
            except Exception as e:
                print(f"Error {split_name} {target}: {e}")
                
    df_res = pd.DataFrame(results)
    
    print("\n=== SO SÁNH BÀI TOÁN CŨ TRÊN DATA MỚI VÀ DATA CŨ ===")
    for target in targets:
        print(f"\nTarget: {target}")
        sub = df_res[df_res['Target'] == target]
        if not sub.empty:
            print(sub[['Split', 'R2', 'MAE']].to_string(index=False))
