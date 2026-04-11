import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('evaluation_plots', exist_ok=True)

def create_feature_subsets(df, hour_limit):
    """
    Drop các feature tương lai theo từng mốc thời gian quy định
    """
    # Các features liên quan đến thời gian cụ thể
    drop_cols = []
    
    if hour_limit < 6:
        drop_cols.extend(['views_at_6h', 'vel_6', 'vel_likes_6', 'vel_cmts_6', 'like_ratio_6h', 'comment_ratio_6h', 'channel_avg_vel_6h'])
    if hour_limit < 4:
        drop_cols.extend(['views_at_4h', 'vel_4', 'vel_likes_4', 'vel_cmts_4'])
    if hour_limit < 2:
        drop_cols.extend(['views_at_2h', 'vel_2', 'vel_likes_2', 'vel_cmts_2'])
        
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

def run_earliest_prediction():
    # Sử dụng tập Time-Based vì chúng phản ánh đúng hơn sức mạnh trên dòng thời gian mới (Tương lai)
    try:
        df_train = pd.read_csv('clean_train_time.csv').fillna(0)
        df_test = pd.read_csv('clean_test_time.csv').fillna(0)
    except FileNotFoundError:
        print("Chưa tìm thấy dữ liệu! Bạn có chắc là clean_data.py đã chạy xong?")
        return
        
    targets = ['views_24h', 'views_48h', 'views_7d']
    time_windows = [0, 2, 4, 6]
    
    # Base columns cần drop khỏi Features chung (Target forms + meta)
    base_drop_cols = ['video_id', 'channel_id', 'published_date', 'category_id', 'is_viral_1m'] + targets
    
    results = []
    
    for target in targets:
        if target not in df_train.columns: 
            print(f"Target {target} không tồn tại.")
            continue
            
        y_train = np.log1p(df_train[target].clip(lower=0))
        y_test = np.log1p(df_test[target].clip(lower=0))
        
        print(f"\n🚀 ĐÁNH GIÁ TARGET: {target} (Tính LOG)")
        
        for tw in time_windows:
            print(f" -> Đang chạy Mô hình quan sát sau {tw} giờ (Subsets: {tw}h)...", end=" ")
            
            # Khử nhiễu tương lai
            X_train = create_feature_subsets(df_train, tw)
            X_test = create_feature_subsets(df_test, tw)
            
            # Khử base metadata
            X_train = X_train.drop(columns=[c for c in base_drop_cols if c in X_train.columns], errors='ignore')
            X_test = X_test.drop(columns=[c for c in base_drop_cols if c in X_test.columns], errors='ignore')
            
            # Log transform input features 
            cols_to_log = [c for c in X_train.columns if 'views' in c or 'vel' in c]
            X_train[cols_to_log] = np.log1p(X_train[cols_to_log].clip(lower=0))
            X_test[cols_to_log] = np.log1p(X_test[cols_to_log].clip(lower=0))
            
            X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
            X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Model training
            model = XGBRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=5, 
                random_state=42, 
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results.append({
                'Target': target,
                'Obs_Hours': tw,
                'R2_Score': r2,
                'MAE': mae
            })
            
            print(f"OK (R2={r2:.4f}, MAE={mae:.4f})")
            
    # Xử lý báo cáo
    results_df = pd.DataFrame(results)
    
    print("\n\n📊 BẢNG KẾT QUẢ ĐÁNH GIÁ ĐIỂM SỚM NHẤT THÍCH HỢP 📊")
    pivot_r2 = results_df.pivot(index='Target', columns='Obs_Hours', values='R2_Score')
    print("=== Điểm số R2 thay đổi qua từng giờ quan sát ===")
    print(pivot_r2.to_string())
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for target in targets:
        sub = results_df[results_df['Target'] == target]
        plt.plot(sub['Obs_Hours'], sub['R2_Score'], marker='o', label=f"{target} (R2)")
        
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Ngưỡng R2 = 0.8 (Khá tốt)')
    plt.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Ngưỡng R2 = 0.9 (Rất Tốt)')
    plt.xlabel('Số Giờ Quan Sát (Hours)')
    plt.ylabel('R2 Score (Độ chính xác)')
    plt.title('Khả năng dự đoán View Video qua mức độ quan sát ban đầu')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('evaluation_plots/earliest_prediction_time.png')
    plt.close()
    
    print("\n✓ Đã lưu biểu đồ vào `evaluation_plots/earliest_prediction_time.png`")

if __name__ == "__main__":
    run_earliest_prediction()
