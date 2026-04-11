import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

os.makedirs('evaluation_plots', exist_ok=True)

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
    grouped = df_raw.groupby('video_id')
    
    sample_videos = df_raw['video_id'].unique()[:10]
    
    print("\n=== 🛠️ DEBUG skipping logic ===")
    for vid in sample_videos:
        sub = grouped.get_group(vid).sort_values('age_hours')
        sub_6h = sub[sub['age_hours'] <= 6.0]
        sub_24h = sub[(sub['age_hours'] >= 18.0) & (sub['age_hours'] <= 30.0)]
        
        print(f"Video: {vid}")
        print(f" -> Cả cụm: {len(sub)} dòng")
        print(f" -> sub_6h: {len(sub_6h)} dòng")
        print(f" -> sub_24h: {len(sub_24h)} dòng")
        if len(sub_6h) > 0:
            print(f"    * age_hours <= 6h: {sub_6h['age_hours'].values}")
            print(f"    * views <= 6h: {sub_6h['views'].values}")
        print("---")

if __name__ == "__main__":
    run_bass_fitting()
