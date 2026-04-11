import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def bass_model_cumulative(t, p, q, m):
    if p <= 0 or q <= 0 or m <= 0: return np.zeros_like(t)
    e_val = np.exp(-(p + q) * t)
    return m * ((1 - e_val) / (1 + (q / p) * e_val))

def load_video_series():
    df_clean = pd.read_csv('clean_train.csv')
    valid_ids = df_clean['video_id'].unique()
    
    from clean_data import load_data_full
    raw_list = load_data_full(base_dir='/Users/minhtri86/Downloads/raw', force_month=None)
    df_raw = pd.DataFrame(raw_list)
    
    df_raw['published_at'] = pd.to_datetime(df_raw['published_at'], utc=True).dt.tz_localize(None)
    df_raw['crawled_at'] = pd.to_datetime(df_raw['crawled_at'], errors='coerce', utc=True).dt.tz_localize(None)
    df_raw['age_hours'] = (df_raw['crawled_at'] - df_raw['published_at']).dt.total_seconds() / 3600.0
    
    if 'statistics' in df_raw.columns:
        df_raw['views'] = df_raw['statistics'].apply(lambda x: int(x.get('viewCount', 0)) if isinstance(x, dict) else 0)
    else:
        df_raw['views'] = 0

    df_raw = df_raw[df_raw['video_id'].isin(valid_ids)]
    return df_raw

def run_bass_fitting():
    df_raw = load_video_series()
    df_raw = df_raw.dropna(subset=['age_hours', 'views'])
    
    # 🌟 Tìm các video có điểm dữ liệu thực tế <= 6.0h
    early_vids = df_raw[df_raw['age_hours'] <= 6.0]['video_id'].unique()
    print(f"\n📊 THỐNG KÊ DATASET")
    print(f" -> Tổng số Video hợp lệ: {df_raw['video_id'].nunique()}")
    print(f" -> Số video có dữ liệu <= 6h: {len(early_vids)}")

    if len(early_vids) == 0:
        print("❌ Không có video nào có điểm dữ liệu sơ cấp <= 6h để khớp mút!")
        return

    results = []
    grouped = df_raw.groupby('video_id')
    count = 0
    
    for vid in early_vids[:2000]: # Lấy tối đa 2000 video để fit nhanh
        sub = grouped.get_group(vid).sort_values('age_hours')
        
        sub_6h = sub[sub['age_hours'] <= 6.0]
        sub_24h = sub[(sub['age_hours'] >= 18.0) & (sub['age_hours'] <= 30.0)]
        
        if len(sub_6h) < 3 or sub_24h.empty:
            continue
            
        t_fit = sub_6h['age_hours'].values
        y_fit = sub_6h['views'].values
        actual_24h = sub_24h.iloc[0]['views']
        
        max_y = max(y_fit) if len(y_fit) > 0 else 1
        bounds = ([0.0001, 0.0001, max_y], [1.0, 1.0, max_y * 1000])
        p0 = [0.01, 0.1, max_y * 1.5]
        
        try:
            popt, _ = curve_fit(bass_model_cumulative, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=1500)
            p_fitted, q_fitted, m_fitted = popt
            
            pred_24h = bass_model_cumulative(24.0, p_fitted, q_fitted, m_fitted)
            results.append({'video_id': vid, 'actual_24h': actual_24h, 'pred_24h': pred_24h})
            count += 1
        except Exception:
            continue

    print(f"✅ Đã khớp mút thành công cho {count} videos.")
    if count == 0:
        print("❌ Cực ít điểm dữ liệu cho curve: Không thể tính R2")
        return

    res_df = pd.DataFrame(results).dropna(subset=['actual_24h', 'pred_24h'])
    y_true_24 = np.log1p(res_df['actual_24h'])
    y_pred_24 = np.log1p(np.clip(res_df['pred_24h'], 0, None))
    
    r2_24 = r2_score(y_true_24, y_pred_24)
    print(f"\n📊 KẾT QUẢ EXTRAPOLATION BASS MODEL (24h) ({count} clips) 📊")
    print(f" -> R2 (Log) = {r2_24:.4f}")

if __name__ == "__main__":
    run_bass_fitting()
