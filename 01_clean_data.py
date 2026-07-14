import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

BASE_DIR = "/Users/minhtri86/Downloads/raw"
OUTPUT_TRAIN_CHAN = "/Users/minhtri86/Downloads/raw/clean_train_channel.csv"
OUTPUT_TEST_CHAN = "/Users/minhtri86/Downloads/raw/clean_test_channel.csv"
OUTPUT_TRAIN_TIME = "/Users/minhtri86/Downloads/raw/clean_train_time.csv"
OUTPUT_TEST_TIME = "/Users/minhtri86/Downloads/raw/clean_test_time.csv"

def get_cyclic_time_vec(val, max_val):
    sin = np.sin(2 * np.pi * val / max_val)
    cos = np.cos(2 * np.pi * val / max_val)
    return sin, cos

def load_data_full(base_dir, force_month='03'):
    all_video_data = []
    year = '2026'
    months = [force_month] if force_month else ['02', '03']
    
    print("Bắt đầu duyệt tệp tin...")
    for m in months:
        m_path = os.path.join(base_dir, year, m)
        if not os.path.exists(m_path): continue
        for d in sorted(os.listdir(m_path)):
            if d.startswith('.'): continue
            d_path = os.path.join(m_path, d)
            for h in sorted(os.listdir(d_path)):
                if h.startswith('.'): continue
                h_path = os.path.join(d_path, h)
                for ch in os.listdir(h_path):
                    if ch.startswith('.'): continue
                    ch_path = os.path.join(h_path, ch)
                    try: files = os.listdir(ch_path)
                    except: continue
                    for f in files:
                        if not f.endswith('.json'): continue
                        try:
                            with open(os.path.join(ch_path, f), 'r', encoding='utf-8') as jf:
                                all_video_data.append(json.load(jf))
                        except: continue
    print(f"-> Đã tải {len(all_video_data)} bản ghi.")
    return all_video_data

def clean_and_impute_fast(raw_records):
    print("Đang dựng DataFrame...")
    df = pd.DataFrame(raw_records)
    if df.empty: return df

    # Bóc tách Tiêu đề an toàn
    if 'snippet' in df.columns:
        df['title'] = df['snippet'].apply(lambda x: x.get('title', '') if isinstance(x, dict) else '')
    else:
        df['title'] = ''

    print("Xử lý mốc thời gian diện rộng (Vectorized)...")
    # 1. Chuyển đổi Datetime Vectorized
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True).dt.tz_localize(None)
    df['crawled_at'] = pd.to_datetime(df['crawled_at'], errors='coerce', utc=True).dt.tz_localize(None)
    df['age_hours'] = (df['crawled_at'] - df['published_at']).dt.total_seconds() / 3600.0
    df['published_date'] = df['published_at'].dt.strftime('%Y-%m-%d')
    
    # Thời gian chu kỳ
    df['pub_hour'] = df['published_at'].dt.hour
    df['pub_dow'] = df['published_at'].dt.weekday
    df['publish_hour_sin'], df['publish_hour_cos'] = get_cyclic_time_vec(df['pub_hour'], 24)
    df['publish_dow_sin'], df['publish_dow_cos'] = get_cyclic_time_vec(df['pub_dow'], 7)
    df['is_weekend'] = (df['pub_dow'] >= 5).astype(int)

    # Trích xuất thống kê
    df['views'] = df['statistics'].apply(lambda x: int(x.get('viewCount', 0)) if isinstance(x, dict) else 0)
    df['likes'] = df['statistics'].apply(lambda x: int(x.get('likeCount', 0)) if isinstance(x, dict) else 0)
    df['comments'] = df['statistics'].apply(lambda x: int(x.get('commentCount', 0)) if isinstance(x, dict) else 0)

    print("Chiết tách tính năng tĩnh và cấu trúc Time-Series...")
    # Lấy features tĩnh per video (first snapshot)
    static_cols = ['video_id', 'channel_id', 'published_date', 'published_at', 'duration_sec', 'is_weekend', 
                   'publish_hour_sin', 'publish_hour_cos', 'publish_dow_sin', 'publish_dow_cos', 'title']
    df_static = df.groupby('video_id')[static_cols[1:]].first().reset_index()
    
    # Khôi phục Category
    cat_vals = df.groupby('video_id')['category_id'].first().fillna('Unknown').reset_index()
    df_static = df_static.merge(cat_vals, on='video_id', how='left')

    df_sorted = df.sort_values(['video_id', 'age_hours'])

    def filter_closest(target_age, threshold=4.0):
        mask = (df_sorted['age_hours'] >= target_age - threshold) & (df_sorted['age_hours'] <= target_age + threshold)
        sub = df_sorted[mask].copy()
        if sub.empty: return pd.DataFrame(columns=['video_id'])
        sub['diff'] = abs(sub['age_hours'] - target_age)
        best = sub.sort_values(['video_id', 'diff']).drop_duplicates('video_id', keep='first')
        return best

    v0 = filter_closest(0.0).rename(columns={'views':'v0', 'likes':'l0', 'comments':'c0'})[['video_id', 'v0', 'l0', 'c0']]
    v2 = filter_closest(2.0, threshold=2.0).rename(columns={'views':'v2', 'likes':'l2', 'comments':'c2'})[['video_id', 'v2', 'l2', 'c2']]
    v4 = filter_closest(4.0, threshold=2.0).rename(columns={'views':'v4', 'likes':'l4', 'comments':'c4'})[['video_id', 'v4', 'l4', 'c4']]
    v6 = filter_closest(6.0, threshold=2.0).rename(columns={'views':'v6', 'likes':'l6', 'comments':'c6'})[['video_id', 'v6', 'l6', 'c6']]
    v24 = filter_closest(24.0, threshold=6.0).rename(columns={'views':'v24'})[['video_id', 'v24']]
    v48 = filter_closest(48.0, threshold=12.0).rename(columns={'views':'v48'})[['video_id', 'v48']]
    v168 = filter_closest(168.0, threshold=48.0).rename(columns={'views':'v168'})[['video_id', 'v168']]

    # Triển khai gộp mốc
    df_merged = df_static
    df_merged = df_merged.merge(v0, on='video_id', how='left')
    df_merged = df_merged.merge(v2, on='video_id', how='left')
    df_merged = df_merged.merge(v4, on='video_id', how='left')
    df_merged = df_merged.merge(v6, on='video_id', how='left')
    df_merged = df_merged.merge(v24, on='video_id', how='left')
    df_merged = df_merged.merge(v48, on='video_id', how='left')
    df_merged = df_merged.merge(v168, on='video_id', how='left')
    
    df_merged = df_merged.fillna(0)

    # Backfill & Imputation
    df_merged['views_at_2h'] = df_merged['v2'].replace(0, np.nan).fillna(df_merged['v0'])
    df_merged['views_at_4h'] = df_merged['v4'].replace(0, np.nan).fillna(df_merged['views_at_2h'])
    df_merged['views_at_6h'] = df_merged['v6'].replace(0, np.nan).fillna(df_merged['views_at_4h'])
    df_merged['views_24h'] = df_merged['v24'].replace(0, np.nan).fillna(df_merged['views_at_6h'])
    df_merged['views_48h'] = df_merged['v48'].replace(0, np.nan).fillna(df_merged['views_24h'])
    df_merged['views_7d'] = df_merged['v168'].replace(0, np.nan).fillna(df_merged['views_48h'])

    df_merged['like_ratio_6h'] = (df_merged['l6'] / df_merged['views_at_6h']).fillna(0)
    df_merged['comment_ratio_6h'] = (df_merged['c6'] / df_merged['views_at_6h']).fillna(0)
    
    # Chỉ số Gia tốc theo mốc đề xuất
    df_merged['vel_2'] = df_merged['views_at_2h'] - df_merged['v0']
    df_merged['vel_4'] = df_merged['views_at_4h'] - df_merged['views_at_2h']
    df_merged['vel_6'] = df_merged['views_at_6h'] - df_merged['views_at_4h']

    # Chỉ số Gia tốc của Like & Comment (Engagements)
    df_merged['vel_likes_2'] = (df_merged['l2'] - df_merged['l0']).fillna(0)
    df_merged['vel_likes_4'] = (df_merged['l4'] - df_merged['l2']).fillna(0)
    df_merged['vel_likes_6'] = (df_merged['l6'] - df_merged['l4']).fillna(0)

    df_merged['vel_cmts_2'] = (df_merged['c2'] - df_merged['c0']).fillna(0)
    df_merged['vel_cmts_4'] = (df_merged['c4'] - df_merged['c2']).fillna(0)
    df_merged['vel_cmts_6'] = (df_merged['c6'] - df_merged['c4']).fillna(0)

    # Last view logic
    last_views = df_sorted.drop_duplicates('video_id', keep='last')[['video_id', 'views']].rename(columns={'views':'_last_views'})
    df_merged = df_merged.merge(last_views, on='video_id', how='left')

    ch_stats = df_merged.groupby('channel_id').agg(
        channel_avg_views=('_last_views', 'mean'),
        channel_video_count=('video_id', 'nunique'),
        channel_avg_vel_6h=('vel_6', 'mean')
    ).reset_index()
    
    df_merged['is_viral'] = df_merged['_last_views'] > df_merged.groupby('channel_id')['_last_views'].transform('median')
    ch_viral = df_merged.groupby('channel_id')['is_viral'].mean().rename('channel_viral_rate').reset_index()

    df_merged = df_merged.merge(ch_stats, on='channel_id', how='left')
    df_merged = df_merged.merge(ch_viral, on='channel_id', how='left')
    
    df_merged['is_viral_1m'] = (df_merged['_last_views'] >= 1000000).astype(int)

    # 🌟 Bổ sung Tính năng: Sức mạnh Kênh (Composite Score)
    max_avg_views = df_merged['channel_avg_views'].max() if df_merged['channel_avg_views'].max() > 0 else 1
    df_merged['channel_strength'] = (df_merged['channel_avg_views'] / max_avg_views) * 0.7 + df_merged['channel_viral_rate'] * 0.3

    df_merged = df_merged.drop(columns=['_last_views', 'is_viral', 'v0', 'v2', 'v4', 'v6', 'v24', 'v48', 'v168', 'l0', 'l2', 'l4', 'l6', 'c0', 'c2', 'c4', 'c6'], errors='ignore')

    # 🌟 GIAI ĐOẠN 1: Mã hóa Tiêu Đề (NLP Title Embedding)
    print("Bắt đầu Mã hóa Tiêu đề (Sentence-Transformers)...")
    from sentence_transformers import SentenceTransformer
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = nlp_model.encode(df_merged['title'].astype(str).tolist(), show_progress_bar=True)
    
    df_emb = pd.DataFrame(embeddings, columns=[f'title_emb_{i}' for i in range(embeddings.shape[1])])
    df_merged = pd.concat([df_merged.reset_index(drop=True), df_emb], axis=1)
    df_merged = df_merged.drop(columns=['title'])

    # 🟢 LOẠI BỎ VIDEO 0 VIEWS ĐỂ CHỐNG LOÃNG PHÂN PHỐI (Nâng cao Skewness accuracy)
    df_merged = df_merged[df_merged['views_24h'] > 0]

    return df_merged

if __name__ == "__main__":
    raw_dat = load_data_full(BASE_DIR, force_month=None)
    
    if raw_dat:
        df_clean = clean_and_impute_fast(raw_dat)
        if not df_clean.empty:
            print(f"Hoàn thành Căn chỉnh! Shape: {df_clean.shape}")
            
            channels = df_clean['channel_id'].unique()
            train_channels, test_channels = train_test_split(channels, test_size=0.2, random_state=42)
            
            df_train_chan = df_clean[df_clean['channel_id'].isin(train_channels)]
            df_test_chan = df_clean[df_clean['channel_id'].isin(test_channels)]
            
            df_train_chan.drop(columns=['published_at'], errors='ignore').to_csv(OUTPUT_TRAIN_CHAN, index=False)
            df_test_chan.drop(columns=['published_at'], errors='ignore').to_csv(OUTPUT_TEST_CHAN, index=False)
            
            # Chia thêm bộ dữ liệu theo Time-Series
            df_time = df_clean.sort_values('published_at').reset_index(drop=True)
            split_idx = int(len(df_time) * 0.8)
            df_train_time = df_time.iloc[:split_idx]
            df_test_time = df_time.iloc[split_idx:]
            
            df_train_time.drop(columns=['published_at'], errors='ignore').to_csv(OUTPUT_TRAIN_TIME, index=False)
            df_test_time.drop(columns=['published_at'], errors='ignore').to_csv(OUTPUT_TEST_TIME, index=False)
            
            print(f"--- XÁC MINH CHỐNG LEAKAGE ---")
            print(f"Tổng kênh: {len(channels)}")
            print(f"Train Channels: {len(train_channels)} | Test Channels: {len(test_channels)}")
            print(f"Đã xuất Channel Based: {OUTPUT_TRAIN_CHAN} và {OUTPUT_TEST_CHAN}")
            print(f"Đã xuất Time Based: {OUTPUT_TRAIN_TIME} và {OUTPUT_TEST_TIME}")
        else:
             print("Lọc rỗng.")
    else:
         print("Không nạp được raw_dat.")
