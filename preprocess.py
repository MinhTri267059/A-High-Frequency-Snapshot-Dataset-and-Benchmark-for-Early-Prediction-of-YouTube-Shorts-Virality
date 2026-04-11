import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def count_caps_ratio(title):
    if not title: return 0.0
    upper_chars = sum(1 for c in title if c.isupper())
    return upper_chars / len(title)

def count_emojis(title):
    # Đếm các ký tự nằm ngoài dải ASCII/Bán chuẩn dạng thô
    if not title: return 0
    # Một số dải mã unicode cho emoji
    emoji_count = 0
    for c in title:
        # Kiểm tra dải mã của emoji
        if ord(c) > 10000:  # Đơn giản hóa đếm emoji thô
            emoji_count += 1
    return emoji_count

def has_number(title):
    return 1 if re.search(r'\d+', title) else 0

def is_question_func(title):
    if not title: return 0
    title_clean = title.strip()
    return 1 if title_clean.endswith('?') or '?' in title_clean else 0

def get_cyclic_time(val, max_val):
    sin_val = np.sin(2 * np.pi * val / max_val)
    cos_val = np.cos(2 * np.pi * val / max_val)
    return sin_val, cos_val

def load_data(base_dir, max_files=1000):
    all_video_data = []
    count = 0
    
    # Duyệt file: raw/Năm/Tháng/Ngày/Giờ/Kênh/Video.json
    year = '2026'
    months = ['03'] # Test tháng 3 cho nhanh
    
    month_path = os.path.join(base_dir, year)
    if not os.path.exists(month_path):
        print(f"Không tìm thấy thư mục {month_path}")
        return []

    for m in months:
        m_path = os.path.join(month_path, m)
        if not os.path.exists(m_path): continue
        
        # Duyệt ngày
        days = sorted(os.listdir(m_path))
        for d in days:
            if d.startswith('.'): continue
            d_path = os.path.join(m_path, d)
            
            # Duyệt giờ
            hours = sorted(os.listdir(d_path))
            for h in hours:
                if h.startswith('.'): continue
                h_path = os.path.join(d_path, h)
                
                # Duyệt kênh
                channels = os.listdir(h_path)
                for ch in channels:
                    if ch.startswith('.'): continue
                    ch_path = os.path.join(h_path, ch)
                    
                    # Duyệt videos
                    try:
                        files = os.listdir(ch_path)
                    except:
                        continue
                    for f in files:
                        if not f.endswith('.json'): continue
                        f_path = os.path.join(ch_path, f)
                        
                        try:
                            with open(f_path, 'r', encoding='utf-8') as jf:
                                data = json.load(jf)
                                all_video_data.append(data)
                                count += 1
                        except:
                            continue
                        
                        if max_files and count >= max_files:
                            break
                    if max_files and count >= max_files: break
                if max_files and count >= max_files: break
            if max_files and count >= max_files: break
    
    print(f"Đã tải {len(all_video_data)} bản ghi video.")
    return all_video_data

def process_features(raw_records):
    df_raw = pd.DataFrame(raw_records)
    if df_raw.empty:
        print("Không có dữ liệu để xử lý.")
        return pd.DataFrame()

    # 1. Trích xuất metadata tĩnh (Static Features)
    records = []
    
    # Nhóm theo video_id để gom các điểm thời gian
    for vid, group in df_raw.groupby('video_id'):
        # Lấy bản ghi đầu tiên để tính feature tĩnh cho video đó
        first_row = group.iloc[0]
        title = first_row.get('title', '')
        desc = first_row.get('description', '')
        published_at = pd.to_datetime(first_row.get('published_at'), utc=True).tz_localize(None)
        
        # Thời gian
        pub_hour = published_at.hour
        pub_dow = published_at.weekday() # 0 = Monday, 6 = Sunday
        hour_sin, hour_cos = get_cyclic_time(pub_hour, 24)
        dow_sin, dow_cos = get_cyclic_time(pub_dow, 7)
        is_weekend = 1 if pub_dow >= 5 else 0
        
        # Text Metrics
        tags = first_row.get('tags', [])
        tags_count = len(tags) if isinstance(tags, list) else 0
        desc_len = len(desc) if desc else 0
        caps_ratio = count_caps_ratio(title)
        emoji_cnt = count_emojis(title)
        is_q = is_question_func(title)
        has_num = has_number(title)
        title_len = len(title) if title else 0
        hashtag_count = len(re.findall(r'#\w+', desc)) if desc else 0 # Chữa cháy từ mô tả
        
        item = {
            'video_id': vid,
            'channel_id': first_row.get('channel_id'),
            'duration_sec': first_row.get('duration_sec', 0),
            'tags_count': tags_count,
            'desc_length': desc_len,
            'caps_ratio': caps_ratio,
            'emoji_count': emoji_cnt,
            'is_question': is_q,
            'title_has_number': has_num,
            'title_len': title_len,
            'hashtag_count': hashtag_count,
            'publish_hour_sin': hour_sin,
            'publish_hour_cos': hour_cos,
            'publish_dow_sin': dow_sin,
            'publish_dow_cos': dow_cos,
            'is_weekend': is_weekend,
            'category_id': first_row.get('category_id', 'Unknown'),
            'published_date': published_at.strftime('%Y-%m-%d') if pd.notnull(published_at) else '1980-01-01'
        }
        
        # 2. Xử lý Time-Series snapshot (0h, 6h, 24h)
        # Tính tuổi thọ cho từng điểm crawl
        group['crawled_at'] = pd.to_datetime(group['crawled_at'], errors='coerce', utc=True).dt.tz_localize(None)
        group['published_at'] = published_at
        group['age_hours'] = (group['crawled_at'] - group['published_at']).dt.total_seconds() / 3600.0
        
        # Sắp xếp theo tuổi
        group_sorted = group.dropna(subset=['age_hours']).sort_values('age_hours')
        
        # Tìm điểm gần 0h, 6h, 24h nhất
        def get_stat_near_age(target_age, threshold=3.0):
            # Lọc điểm cách biệt hiệu ứng
            group_sorted['diff'] = abs(group_sorted['age_hours'] - target_age)
            best_match = group_sorted[group_sorted['diff'] <= threshold]
            if not best_match.empty:
                match_row = best_match.sort_values('diff').iloc[0]
                stats = match_row.get('statistics', {})
                return int(stats.get('viewCount', 0)), int(stats.get('likeCount', 0)), int(stats.get('commentCount', 0)), match_row['age_hours']
            return None, None, None, None

        v0, l0, c0, _ = get_stat_near_age(0.0)
        v1, _, _, _ = get_stat_near_age(1.0)
        v2, _, _, _ = get_stat_near_age(2.0)
        v6, l6, c6, _ = get_stat_near_age(6.0)
        v24, _, _, _ = get_stat_near_age(24.0)
        v72, _, _, _ = get_stat_near_age(72.0, threshold=12.0)
        v7d, _, _, _ = get_stat_near_age(168.0, threshold=24.0)

        # Gán view mốc
        item['views_at_0h'] = v0 if v0 is not None else 0
        item['views_at_1h'] = v1 if v1 is not None else item['views_at_0h']
        item['views_at_2h'] = v2 if v2 is not None else item['views_at_1h']
        item['views_at_6h'] = v6 if v6 is not None else item['views_at_2h']
        item['views_24h'] = v24 if v24 is not None else item['views_at_6h']
        item['views_72h'] = v72 if v72 is not None else item['views_24h']
        item['views_7d'] = v7d if v7d is not None else item['views_72h']
        
        # Tính Latios và Ratio
        item['like_ratio_6h'] = (l6 / v6) if (v6 and v6 > 0 and l6 is not None) else 0.0
        item['comment_ratio_6h'] = (c6 / v6) if (v6 and v6 > 0 and c6 is not None) else 0.0
        
        # Tính Velocity
        item['vel_1h'] = (item['views_at_1h'] - item['views_at_0h']) / 1.0
        item['vel_2h'] = (item['views_at_2h'] - item['views_at_1h']) / 1.0
        item['vel_6h'] = (item['views_at_6h'] - item['views_at_0h']) / 6.0 if v6 is not None else 0.0
        
        item['_last_views'] = int(df_raw[df_raw['video_id'] == vid].iloc[-1].get('statistics', {}).get('viewCount', 0))
        item['_like6'] = l6
        item['_all_views'] = group_sorted['statistics'].apply(lambda x: int(x.get('viewCount', 0))).tolist()
        
        records.append(item)

    df_feats = pd.DataFrame(records)
    if df_feats.empty: return df_feats

    # 3. Tính toán Feature cấp Kênh (Channel-level aggregates)
    ch_stats = df_feats.groupby('channel_id').agg(
        channel_avg_views=('_last_views', 'mean'),
        channel_video_count=('video_id', 'nunique'),
        channel_avg_vel_6h=('vel_6h', 'mean')
    ).reset_index()

    # Tính Viral Rate (ví dụ video đạt > 1000 view trung bình được tính là viral, tạm tính thô)
    df_feats['is_viral'] = df_feats['_last_views'] > df_feats.groupby('channel_id')['_last_views'].transform('median')
    ch_viral = df_feats.groupby('channel_id')['is_viral'].mean().rename('channel_viral_rate').reset_index()

    # Merge lại vào df_feats
    df_feats = df_feats.merge(ch_stats, on='channel_id', how='left')
    df_feats = df_feats.merge(ch_viral, on='channel_id', how='left')

    # Dropping helper columns
    df_feats = df_feats.drop(columns=['_last_views', '_like6', '_all_views', 'is_viral'], errors='ignore')

    return df_feats

if __name__ == "__main__":
    BASE_DIR = "/Users/minhtri86/Downloads/raw"
    OUTPUT_FILE = "/Users/minhtri86/Downloads/raw/processed_features.csv"
    
    print("Bắt đầu tải dữ liệu...")
    raw_dat = load_data(BASE_DIR, max_files=5000) # Lấy 5000 file cho có mẫu gộp
    
    if raw_dat:
        print("Bắt đầu xử lý Features...")
        final_df = process_features(raw_dat)
        
        if not final_df.empty:
            print(f"Hoàn thành! Shape: {final_df.shape}")
            # Ensure specified order
            cols_order = [
                'video_id', 'channel_id', 'published_date', 'channel_avg_views', 'channel_viral_rate', 
                'channel_video_count', 'channel_avg_vel_6h', 'vel_1h', 'vel_2h', 'vel_6h', 
                'duration_sec', 'tags_count', 'desc_length', 'caps_ratio', 'emoji_count', 
                'is_question', 'title_has_number', 'title_len', 'hashtag_count', 
                'publish_hour_sin', 'publish_hour_cos', 'publish_dow_sin', 'publish_dow_cos', 
                'is_weekend', 'category_id', 'like_ratio_6h', 'comment_ratio_6h', 
                'views_at_0h', 'views_at_1h', 'views_at_2h', 'views_at_6h', 'views_24h', 
                'views_72h', 'views_7d'
            ]
            # Lọc các cột hiện có trong cols_order để tránh lỗi KeyError nếu cột nào đó khuyết
            existing_cols = [c for c in cols_order if c in final_df.columns]
            final_df = final_df[existing_cols]
            
            final_df.to_csv(OUTPUT_FILE, index=False)
            print(f"Đã lưu kết quả tại {OUTPUT_FILE}")
        else:
            print("Xử lý thất bại.")
    else:
        print("Không có dữ liệu.")
