import pandas as pd
import numpy as np
import os

def load_video_series():
    df_clean = pd.read_csv('clean_train.csv')
    valid_ids = df_clean['video_id'].unique()
    
    from clean_data import load_data_full
    raw_list = load_data_full(base_dir='/Users/minhtri86/Downloads/raw', force_month=None)
    df_raw = pd.DataFrame(raw_list)
    
    df_raw['published_at'] = pd.to_datetime(df_raw['published_at'], utc=True).dt.tz_localize(None)
    df_raw['crawled_at'] = pd.to_datetime(df_raw['crawled_at'], errors='coerce', utc=True).dt.tz_localize(None)
    df_raw['age_hours'] = (df_raw['crawled_at'] - df_raw['published_at']).dt.total_seconds() / 3600.0
    
    df_raw = df_raw[df_raw['video_id'].isin(valid_ids)]
    return df_raw

def run_bass_fitting():
    df_raw = load_video_series()
    grouped = df_raw.groupby('video_id')
    
    sample_videos = df_raw['video_id'].unique()[:5]
    
    print("\n=== 🛠️ DEBUG age_hours values ===")
    for vid in sample_videos:
        sub = grouped.get_group(vid).sort_values('age_hours')
        print(f"\nVideo: {vid}")
        print(f" -> Cả cụm: {len(sub)} dòng")
        print(f" -> Giá trị age_hours (15 mốc đầu): {sub['age_hours'].values[:15]}")
        print(f" -> mốc views tương ứng: {sub['views'].values[:15] if 'views' in sub.columns else 'No views-col'}")
    print("---")

if __name__ == "__main__":
    run_bass_fitting()
