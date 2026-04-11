import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('evaluation_plots', exist_ok=True)

def load_split_data():
    train = pd.read_csv('clean_train.csv')
    test = pd.read_csv('clean_test.csv')
    return train.fillna(0), test.fillna(0)

def run_evaluation():
    df_train, df_test = load_split_data()
    print(f"Nạp dữ liệu Clean thành công. Train: {df_train.shape}, Test: {df_test.shape}")
    
    targets = ['views_24h', 'views_48h']
    
    # 1. Định nghĩa Không gian tham số (Search Space)
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [15, 20, None]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 7]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 7]
        }
    }

    # Gom nhóm khởi tạo
    models_base = {
        'Linear': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, n_jobs=-1)
    }

    performance = []
    
    for target in targets:
        if target not in df_train.columns: continue
        
        y_train = np.log1p(df_train[target])
        y_test = np.log1p(df_test[target])
        
        drop_cols = ['video_id', 'channel_id', 'published_date', 'category_id'] + ['views_24h', 'views_48h']
        X = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors='ignore')
        X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors='ignore')

        # 📊 CHUẨN HÓA LOG1P CHO INPUT FEATURES (Views & Velocity)
        # Đồng nhất với logic log1p của Target để tăng độ chính xác
        cols_to_log = [c for c in X.columns if 'views' in c or 'vel' in c]
        X[cols_to_log] = np.log1p(X[cols_to_log].clip(lower=0))
        X_test[cols_to_log] = np.log1p(X_test[cols_to_log].clip(lower=0))

        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

        print(f"\n--- 🚀 FINE-TUNING TARGET: {target} ---")
        
        for name, model in models_base.items():
            try:
                print(f" -> Đang xử lý {name}...", end=" ")
                if name in param_grids:
                    # Áp dụng RandomizedSearchCV cho Tuning
                    search = RandomizedSearchCV(
                        model, 
                        param_distributions=param_grids[name], 
                        n_iter=4, # Nhỏ để chạy nhanh trên full data
                        cv=3, 
                        scoring='r2', 
                        n_jobs=-1, 
                        random_state=42
                    )
                    search.fit(X, y_train)
                    best_model = search.best_estimator_
                    print(f"Best Params: {search.best_params_}")
                else:
                    # Mặc định fit cho linear
                    best_model = model.fit(X, y_train)
                    print("Default Fit")

                y_pred = best_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                performance.append({'Target': target, 'Model': name, 'R2': r2, 'MAE': mae})
                
                # 💾 LƯU MODEL XGBOOST (Dành cho triển khai Web)
                if name == 'XGBoost':
                    model_path = f'models/best_xgboost_{target}.pkl'
                    os.makedirs('models', exist_ok=True)
                    with open(model_path, 'wb') as f:
                        pickle.dump(best_model, f)
                    print(f" -> Saved: {model_path}")

                    # Lưu danh sách features để Web app dùng đúng thứ tự
                    with open('models/features_list.pkl', 'wb') as f:
                        pickle.dump(X.columns.tolist(), f)
                
                # Vẽ Scatter đồ thị
                plt.figure(figsize=(6, 5))
                plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Thực tế (Log Views)')
                plt.ylabel('Dự đoán (Log Views)')
                plt.title(f'{name} - {target} (Tuned)')
                plt.savefig(f'evaluation_plots/{target}_{name}_scatter.png')
                plt.close()

            except Exception as e:
                performance.append({'Target': target, 'Model': name, 'R2': np.nan, 'MAE': np.nan, 'Error': str(e)})
                print(f"LỖI: {e}")

    # In Master Table
    if performance:
        master_df = pd.DataFrame(performance)
        print("\n\n📊 BẢNG SO SÁNH TỔNG HỢP SAU FINE-TUNING 📊")
        pivot_r2 = master_df.pivot_table(index='Target', columns='Model', values='R2')
        print("\n=== Điểm số R2 (Tuned) ===")
        print(pivot_r2.to_string())
        
        pivot_mae = master_df.pivot_table(index='Target', columns='Model', values='MAE')
        print("\n=== Điểm số MAE (Tuned) ===")
        print(pivot_mae.to_string())

        # Vẽ biểu đồ So sánh R2 (Bar Chart)
        plt.figure(figsize=(10, 5))
        pivot_r2.plot(kind='bar', figsize=(10, 5))
        plt.title('So sánh R2 Score sau Tuning (XGBoost tích hợp)')
        plt.ylabel('R2 Score')
        plt.xlabel('Target')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig('evaluation_plots/r2_tuned_comparison.png')
        plt.close()

if __name__ == "__main__":
    run_evaluation()
