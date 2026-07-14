import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('evaluation_plots', exist_ok=True)

df_train = pd.read_csv('/Users/minhtri86/Downloads/raw/clean_train.csv')
df_test = pd.read_csv('/Users/minhtri86/Downloads/raw/clean_test.csv')

def profile_df(df, name):
    print(f"\n===== 📊 THỐNG KÊ BỘ DỮ LIỆU: {name.upper()} =====")
    print(f"-> Số lượng Video (Dòng): {df.shape[0]}")
    if 'channel_id' in df.columns:
        print(f"-> Số lượng Kênh (unique): {df['channel_id'].nunique()}")
    else:
        # Nếu column đã bị drop trong train data (Chỉ xem size)
        print("-> Không tìm thấy cột channel_id")

    targets = ['views_24h', 'views_48h']
    for target in targets:
        if target in df.columns:
            series = df[target]
            print(f"\n--- Thống kê mốc {target} (View tuyệt đối) ---")
            print(f"   * Mean:   {series.mean():.2f}")
            print(f"   * Median: {series.median():.2f}")
            print(f"   * Max:    {series.max():.2f}")
            print(f"   * Std:    {series.std():.2f}")
            print(f"   * Cực tiểu (0 views): {(series == 0).sum()} dòng ({(series == 0).sum()/len(series)*100:.2f}%)")
            
            # Tính Skewness
            skew = series.skew()
            print(f"   * Độ lệch (Skewness): {skew:.2f} " + ("(Lệch phải nặng - Extreme Right-Skewed)" if skew > 2 else ""))

print("--- PHÂN TÁCH TRAIN / TEST ---")
profile_df(df_train, 'Train')
profile_df(df_test, 'Test')

# Vẽ Biểu đồ Phân phối (Distribution) mẫu
target_visual = 'views_24h'
if target_visual in df_train.columns:
    plt.figure(figsize=(10, 5))
    plt.hist(np.log1p(df_train[target_visual]), bins=50, alpha=0.5, label='Train (Log)', color='blue')
    plt.hist(np.log1p(df_test[target_visual]), bins=50, alpha=0.5, label='Test (Log)', color='orange')
    plt.title(f'Phân phối Lượt Xem (Log Views) - {target_visual}')
    plt.xlabel('Log(Views + 1)')
    plt.ylabel('Tần suất')
    plt.legend()
    plt.tight_layout()
    plt.savefig('evaluation_plots/target_distribution.png')
    plt.close()
    print("\n✅ Đã lưu đồ thị phân phối tại 'evaluation_plots/target_distribution.png'")
