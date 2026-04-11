import pandas as pd
import numpy as np

CSV_PATH = '/Users/minhtri86/Downloads/raw/processed_features.csv'
df = pd.read_csv(CSV_PATH)

print('=== THỐNG KÊ QUY MÔ ===')
print(f'Tổng số dòng (Video Thống kê): {len(df)}')
print(f'Số lượng Kênh duy nhất: {df["channel_id"].nunique()}')

# Tính Viral Rate sơ bộ 
median_24h = df['views_24h'].median()
df['is_viral'] = df['views_24h'] > median_24h

print('\n=== PHÂN BỔ THEO CATEGORY ===')
cat = df['category_id'].value_counts()
for k, v in cat.items():
    print(f'- Category {k}: \t{v} video ({v/len(df)*100:.1f}%)')

print('\n=== PHÂN BỔ TARGETS (VIEWS) ===')
for t in ['views_24h', 'views_72h', 'views_7d']:
    if t in df.columns:
         series = df[t].fillna(0)
         n_zero = (series > 0).sum()
         print(f' * Mốc {t}:')
         print(f'   - Số video có data (>0): {n_zero} ({n_zero/len(df)*100:.1f}%)')
         print(f'   - View trung bình: \t{series.mean():.1f}')
         print(f'   - View lớn nhất: \t{series.max():.1f}')

print('\n=== ĐỘ PHÂN BỔ VIRAL (MÓC 24H) ===')
viral_counts = df['is_viral'].value_counts()
print(f' - Nhóm Viral (> Trung vị {median_24h:.1f} views): {viral_counts.get(True, 0)} video')
print(f' - Nhóm Chưa Viral: \t\t\t{viral_counts.get(False, 0)} video')

print('\n=== TẦN SUẤT VIDEO TRÊN KÊNH ===')
v_per_ch = df.groupby('channel_id').size()
print(f' - Trung bình: {v_per_ch.mean():.1f} video/kênh')
print(f' - Nhiều nhất: {df["channel_video_count"].max()} video/kênh')
