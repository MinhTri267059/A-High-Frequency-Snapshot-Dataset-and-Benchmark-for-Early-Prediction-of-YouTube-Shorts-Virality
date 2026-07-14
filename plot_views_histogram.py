import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Đọc dữ liệu
print("Loading data...")
df = pd.read_csv('clean_train_time.csv')

views_24h = df['views_24h'].dropna()

# Các thống kê cơ bản
print("\n--- Basic Statistics for views_24h ---")
print(views_24h.describe())

percentiles = [50, 75, 80, 85, 90, 95, 99]
print("\n--- Percentiles ---")
for p in percentiles:
    val = np.percentile(views_24h, p)
    print(f"{p}th percentile: {val:,.0f} views")

# Vẽ histogram
plt.figure(figsize=(12, 6))

# Lọc bỏ top 1% outliers để dễ nhìn histogram hơn
threshold_99 = np.percentile(views_24h, 99)
data_to_plot = views_24h[views_24h < threshold_99]

plt.hist(data_to_plot, bins=100, color='skyblue', edgecolor='black')
plt.title('Histogram of views_24h (excluding top 1%)')
plt.xlabel('Views after 24h')
plt.ylabel('Count of videos')
plt.grid(axis='y', alpha=0.75)

# Đánh dấu một số percentiles trên biểu đồ
for p, color in zip([75, 85, 90, 95], ['green', 'yellow', 'orange', 'red']):
    val = np.percentile(views_24h, p)
    plt.axvline(x=val, color=color, linestyle='--', label=f'{p}th ({val:,.0f})')

plt.legend()
plt.tight_layout()
plt.savefig('views_24h_histogram.png')
print("\nSaved histogram to views_24h_histogram.png")
