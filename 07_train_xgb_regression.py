"""
exp11_xgb_regression_time_tuned.py
====================================
EXP-11: XGBoost Regression trên Time-Based Split với RandomizedSearchCV (n_iter=30)
Lưu vào models/xgb_reg_24h_time_tuned.pkl và models/xgb_reg_48h_time_tuned.pkl
KHÔNG ghi đè các model cũ.
"""
import pandas as pd, numpy as np, pickle, os, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error

OUTPUT_DIR = 'evaluation_plots'
MODEL_DIR  = 'models'
LEAK_COLS  = ['video_id','channel_id','published_date','category_id',
              'views_24h','views_48h','views_7d','is_viral_1m']

def load_data(target):
    tr = pd.read_csv('clean_train_time.csv', low_memory=False).fillna(0)
    te = pd.read_csv('clean_test_time.csv',  low_memory=False).fillna(0)
    y_train = np.log1p(tr[target].clip(lower=0))
    y_test  = np.log1p(te[target].clip(lower=0))
    drop = [c for c in LEAK_COLS if c in tr.columns]
    X_train = tr.drop(columns=drop)
    X_test  = te.drop(columns=[c for c in LEAK_COLS if c in te.columns])
    for df in [X_train, X_test]:
        cols = [c for c in df.columns if 'views' in c or 'vel' in c]
        df[cols] = np.log1p(df[cols].clip(lower=0))
    X_train = X_train.replace([np.inf,-np.inf],0).fillna(0)
    X_test  = X_test.replace([np.inf,-np.inf],0).fillna(0)
    common  = sorted(set(X_train.columns) & set(X_test.columns))
    return X_train[common], y_train, X_test[common], y_test, common

param_dist = {
    'n_estimators'    : [200, 300, 400, 500, 600],
    'learning_rate'   : [0.01, 0.03, 0.05, 0.08, 0.1],
    'max_depth'       : [4, 5, 6, 7, 8],
    'subsample'       : [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha'       : [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda'      : [0.5, 1.0, 2.0, 5.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma'           : [0, 0.1, 0.2, 0.5],
}

results_all = []
for target in ['views_24h', 'views_48h']:
    print(f"\n{'='*60}\nEXP-11: XGBoost Regression Tuned — {target}\n{'='*60}")
    X_train, y_train, X_test, y_test, common = load_data(target)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(common)}")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    base = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0,
                        objective='reg:squarederror')
    cv   = KFold(n_splits=5, shuffle=True, random_state=42)

    print("  Running RandomizedSearchCV (n_iter=30, cv=5, scoring=R2)...")
    search = RandomizedSearchCV(base, param_dist, n_iter=30, cv=cv,
                                scoring='r2', n_jobs=-1,
                                random_state=42, verbose=0)
    search.fit(Xtr, y_train)

    best = search.best_estimator_
    pred = best.predict(Xte)

    r2   = r2_score(y_test, pred)
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mede = median_absolute_error(y_test, pred)
    mape = np.mean(np.abs((np.expm1(y_test) - np.expm1(pred)) /
                           np.expm1(y_test).clip(lower=1))) * 100

    print(f"\n  Best CV R2 : {search.best_score_:.4f}")
    print(f"  Test R2    : {r2:.4f}")
    print(f"  Test MAE   : {mae:.4f}  RMSE: {rmse:.4f}  MAPE: {mape:.1f}%")
    print(f"  Best Params:")
    for k, v in sorted(search.best_params_.items()):
        print(f"    {k:<25} = {v}")

    results_all.append({
        'Target': target, 'Split': 'Time-Based (Tuned)',
        'R2': r2, 'MAE_log': mae, 'RMSE_log': rmse, 'MAPE_%': mape,
        'Best_Params': search.best_params_, 'CV_R2': search.best_score_
    })

    # Save — new file, no overwrite
    save_path = f'{MODEL_DIR}/xgb_reg_{target.replace("views_","")}_time_tuned.pkl'
    artifact = {
        'model': best, 'scaler': scaler,
        'feature_cols': common,
        'best_params': search.best_params_,
        'cv_r2': search.best_score_,
        'split': 'time-based',
        'target': target,
        'test_metrics': {'R2': r2, 'MAE_log': mae, 'RMSE_log': rmse, 'MAPE_%': mape}
    }
    with open(save_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"\n  ✅ Saved → {save_path}")

print("\n\n[SUMMARY EXP-11]")
for r in results_all:
    print(f"  {r['Target']}: R²={r['R2']:.4f}  MAE={r['MAE_log']:.4f}  MAPE={r['MAPE_%']:.1f}%  CV_R²={r['CV_R2']:.4f}")
