import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

df_train = pd.read_csv('/Users/minhtri86/Downloads/raw/clean_train.csv')
target = 'views_24h'
y_train = np.log1p(df_train[target])

drop_cols = ['video_id', 'channel_id', 'published_date', 'category_id', 'views_24h', 'views_48h']
X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors='ignore')
X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)

print(f"Features: {X_train.columns.tolist()}")

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

importances = model.feature_importances_
feature_names = X_train.columns

imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\n📊 ĐÁNH GIÁ ĐỘ QUAN TRỌNG TỈ TRỌNG (Importance) - RandomForest 📊")
print(imp_df.head(10).to_string(index=False))

vel_cols = ['vel_2', 'vel_4', 'vel_6']
print("\n--- Xét riêng 3 mốc Hầm Vận Tốc ---")
print(imp_df[imp_df['Feature'].isin(vel_cols)].to_string(index=False))
