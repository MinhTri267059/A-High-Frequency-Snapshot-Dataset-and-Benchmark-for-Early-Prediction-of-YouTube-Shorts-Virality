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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

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

def run_lstm(train_df, test_df, split_mode, target_col):
    # Tạo Validation Set
    if split_mode == "Channel-Based":
        channels = train_df['channel_id'].unique()
        train_ch, val_ch = train_test_split(channels, test_size=0.1875, random_state=42)
        sub_train_df = train_df[train_df['channel_id'].isin(train_ch)]
        val_df = train_df[train_df['channel_id'].isin(val_ch)]
    else: # Time-Based
        # Chia tiếp top 85% train và 15% validation
        split_idx = int(len(train_df) * 0.85)
        sub_train_df = train_df.iloc[:split_idx]
        val_df = train_df.iloc[split_idx:]

    seq_cols = ['views_at_2h', 'views_at_4h', 'views_at_6h']
    drop_cols = ['video_id', 'channel_id', 'published_date', 'category_id', 'is_viral_1m', 'views_24h', 'views_48h', 'views_7d'] + seq_cols

    y_train = np.log1p(sub_train_df[target_col].clip(lower=0).values)
    y_val = np.log1p(val_df[target_col].clip(lower=0).values)
    y_test = np.log1p(test_df[target_col].clip(lower=0).values)

    seq_train = np.expand_dims(np.log1p(sub_train_df[seq_cols].clip(lower=0).values), axis=-1)
    seq_val = np.expand_dims(np.log1p(val_df[seq_cols].clip(lower=0).values), axis=-1)
    seq_test = np.expand_dims(np.log1p(test_df[seq_cols].clip(lower=0).values), axis=-1)

    static_train = sub_train_df.drop(columns=[c for c in drop_cols if c in sub_train_df.columns], errors='ignore')
    static_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns], errors='ignore')
    static_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors='ignore')

    # Xử lý Inf
    static_train = static_train.replace([np.inf, -np.inf], 0).fillna(0).values
    static_val = static_val.replace([np.inf, -np.inf], 0).fillna(0).values
    static_test = static_test.replace([np.inf, -np.inf], 0).fillna(0).values

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

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5

    print(f" -> Đang đào tạo LSTM cho {target_col} ({split_mode})...", end=" ")
    for epoch in range(100):
        model.train()
        for s_b, st_b, y_b in train_loader:
            s_b, st_b, y_b = s_b.to(device), st_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(s_b, st_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for s_b, st_b, y_b in val_loader:
                s_b, st_b, y_b = s_b.to(device), st_b.to(device), y_b.to(device)
                outputs = model(s_b, st_b)
                val_loss += criterion(outputs, y_b).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'best_lstm_tmp.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(torch.load(f'best_lstm_tmp.pth'))
    model.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for s_b, st_b, y_b in test_loader:
            s_b, st_b = s_b.to(device), st_b.to(device)
            outputs = model(s_b, st_b)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_b.numpy())

    r2 = r2_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    print(f"OK. R2: {r2:.4f} | MAE: {mae:.4f}")
    return r2, mae

if __name__ == "__main__":
    targets = ['views_24h', 'views_48h']
    splits = {
        'Channel-Based': ('clean_train_channel.csv', 'clean_test_channel.csv'),
        'Time-Based': ('clean_train_time.csv', 'clean_test_time.csv')
    }
    
    results = []
    for split_name, (tr_path, te_path) in splits.items():
        try:
            train_df = pd.read_csv(tr_path).fillna(0)
            test_df = pd.read_csv(te_path).fillna(0)
            for target in targets:
                r2, mae = run_lstm(train_df, test_df, split_name, target)
                results.append({'Target': target, 'Split': split_name, 'R2': r2, 'MAE': mae})
        except Exception as e:
            print(f"Error {split_name}: {e}")
            
    print("\n--- KẾT QUẢ SO SÁNH SỬ DỤNG MẠNG TÍN HIỆU SÂU (LSTM) ---")
    df_res = pd.DataFrame(results)
    for target in targets:
        print(f"\nTarget: {target}")
        sub = df_res[df_res['Target'] == target]
        if not sub.empty:
            print(sub[['Split', 'R2', 'MAE']].to_string(index=False))
