import pandas as pd

df = pd.read_csv('/Users/minhtri86/Downloads/raw/features_24h.csv')
print('=== THỐNG KÊ DATASET MỚI ===')
print(f'Tổng số dòng (Video): {len(df)}')
print(f'Số lượng Kênh: {df["channel_id"].nunique()}')

print('\n=== PHÂN BỔ CATEGORY ===')
cat = df['category_id'].value_counts()
for k, v in cat.items():
    print(f'- Category {k}: \t{v} video ({v/len(df)*100:.1f}%)')

print('\n=== ĐÁNH GIÁ TARGET (views_24h) ===')
series = df['views_24h'].fillna(0)
n_zero = (series > 0).sum()
print(f' - Số video có views_24h > 0: {n_zero} ({n_zero/len(df)*100:.1f}%)')
print(f' - View trung bình: \t{series.mean():.1f}')
print(f' - View lớn nhất: \t{series.max():.1f}')
print(f' - Median view: \t{series.median():.1f}')
