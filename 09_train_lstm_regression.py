import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

os.makedirs('evaluation_plots', exist_ok=True)
device = torch.device('cpu') 

class YoutubeDataset(Dataset):
    def __init__(self, seq_data, static_data, y_data):
        self.seq_data = torch.tensor(seq_data, dtype=torch.float32)
        self.static_data = torch.tensor(static_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.seq_data[idx], self.static_data[idx], self.y[idx]

class HybridLSTM(nn.Module):
    def __init__(self, static_dim, hidden_size=64):
        super(HybridLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, seq, static):
        _, (h_n, _) = self.lstm(seq)
        lstm_out = h_n.squeeze(0)
        combined = torch.cat((lstm_out, static), dim=1)
        out = self.fc(combined)
        return out.squeeze(-1)

def train_model():
    print("🚀 Đang khởi động Chia Split 75-15-10 (Channel-Based Split)...")
    train_df = pd.read_csv('clean_train.csv').fillna(0).replace([np.inf, -np.inf], 0)
    test_df = pd.read_csv('clean_test.csv').fillna(0).replace([np.inf, -np.inf], 0)

    # 1. Thực hiện Split 75-15-10 trên Kênh (Channel)
    # df_train hiện tại giữ 80% kênh. Lấy 15% tổng kênh là lấy 18.75% của 80% này!
    channels_train_val = train_df['channel_id'].unique()
    train_sub_ch, val_sub_ch = train_test_split(channels_train_val, test_size=0.1875, random_state=42)
    
    sub_train_df = train_df[train_df['channel_id'].isin(train_sub_ch)]
    val_df = train_df[train_df['channel_id'].isin(val_sub_ch)]

    print(f"\n📊 THỐNG KÊ KÍCH THƯỚC PHÂN TÁCH BA VÙNG 📊")
    print(f" -> TRAIN (75% Kênh): {sub_train_df.shape[0]} Video | {len(train_sub_ch)} Kênh")
    print(f" -> VAL   (15% Kênh): {val_df.shape[0]} Video | {len(val_sub_ch)} Kênh")
    print(f" -> TEST  (10% Kênh): {test_df.shape[0]} Video | {test_df['channel_id'].nunique()} Kênh")

    targets = ['views_24h', 'views_48h']
    seq_cols = ['views_at_2h', 'views_at_4h', 'views_at_6h']
    
    results = []

    for target_col in targets:
        print(f"\n--- 🚀 ĐÀO TẠO LSTM: {target_col} ---")
        drop_cols = ['video_id', 'channel_id', 'published_date', 'category_id'] + targets + seq_cols

        # Biền mảng Numpy
        y_train = np.log1p(sub_train_df[target_col].values)
        y_val = np.log1p(val_df[target_col].values)
        y_test = np.log1p(test_df[target_col].values)

        seq_train = np.expand_dims(np.log1p(sub_train_df[seq_cols].values), axis=-1)
        seq_val = np.expand_dims(np.log1p(val_df[seq_cols].values), axis=-1)
        seq_test = np.expand_dims(np.log1p(test_df[seq_cols].values), axis=-1)

        static_train = sub_train_df.drop(columns=[c for c in drop_cols if c in sub_train_df.columns], errors='ignore').values
        static_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns], errors='ignore').values
        static_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors='ignore').values

        # Chuẩn hóa
        scaler = StandardScaler()
        static_train = scaler.fit_transform(static_train)
        static_val = scaler.transform(static_val)
        static_test = scaler.transform(static_test)

        train_set = YoutubeDataset(seq_train, static_train, y_train)
        val_set = YoutubeDataset(seq_val, static_val, y_val)
        test_set = YoutubeDataset(seq_test, static_test, y_test)

        train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

        model = HybridLSTM(static_train.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Early stopping dải dừng
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = 5

        epochs = 100 # Cho phép chạy dài để test Early Stopping
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for s_b, st_b, y_b in train_loader:
                optimizer.zero_grad()
                outputs = model(s_b, st_b)
                loss = criterion(outputs, y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Kiểm định Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for s_b, st_b, y_b in val_loader:
                    outputs = model(s_b, st_b)
                    val_loss += criterion(outputs, y_b).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'best_lstm_{target_col}.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f" -> Dừng sớm (Early Stopping) tại epoch {epoch+1}")
                    break

        # Load best model
        model.load_state_dict(torch.load(f'best_lstm_{target_col}.pth'))
        model.eval()
        
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for s_b, st_b, y_b in test_loader:
                outputs = model(s_b, st_b)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(y_b.cpu().numpy())

        r2 = r2_score(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        print(f" -> Kết quả Test R2: {r2:.4f} | MAE = {mae:.4f}")
        results.append({'Target': target_col, 'R2': r2, 'MAE': mae})

    print("\n📊 TỔNG KẾT KẾT QUẢ LSTM (Split 75-15-10) 📊")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    train_model()
