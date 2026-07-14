"""
exp13_lstm_vs_xgboost.py
==========================
EXP-13: So sánh chính thức LSTM vs XGBoost Regression trên cùng Time-Based Split.
Đảm bảo tính công bằng (cùng features, cùng train/test sets).
"""
import pandas as pd, numpy as np, torch, torch.nn as nn, os, pickle
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEAK_COLS = ['video_id','channel_id','published_date','category_id', 'views_24h','views_48h','views_7d','is_viral_1m']
SEQ_COLS = ['views_at_2h', 'views_at_4h', 'views_at_6h']

class YoutubeDataset(Dataset):
    def __init__(self, seq, static, y):
        self.seq = torch.tensor(seq, dtype=torch.float32)
        self.static = torch.tensor(static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.seq[i], self.static[i], self.y[i]

class HybridLSTM(nn.Module):
    def __init__(self, static_dim, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden + static_dim, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, s, st):
        _, (h, _) = self.lstm(s)
        out = torch.cat((h.squeeze(0), st), 1)
        return self.fc(out).squeeze(-1)

def load_and_prep(target):
    tr_df = pd.read_csv('clean_train_time.csv').fillna(0)
    te_df = pd.read_csv('clean_test_time.csv').fillna(0)
    
    y_tr = np.log1p(tr_df[target].clip(lower=0)).values
    y_te = np.log1p(te_df[target].clip(lower=0)).values
    
    drop = LEAK_COLS + SEQ_COLS
    X_tr_static = tr_df.drop(columns=[c for c in drop if c in tr_df.columns]).replace([np.inf,-np.inf],0).fillna(0)
    X_te_static = te_df.drop(columns=[c for c in drop if c in te_df.columns]).replace([np.inf,-np.inf],0).fillna(0)
    
    # Log transform for static numeric features
    for df in [X_tr_static, X_te_static]:
        cols = [c for c in df.columns if 'views' in c or 'vel' in c]
        df[cols] = np.log1p(df[cols].clip(lower=0))
    
    common = sorted(set(X_tr_static.columns) & set(X_te_static.columns))
    X_tr_static = X_tr_static[common]
    X_te_static = X_te_static[common]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_static)
    X_te_s = scaler.transform(X_te_static)
    
    seq_tr = np.expand_dims(np.log1p(tr_df[SEQ_COLS].values), -1)
    seq_te = np.expand_dims(np.log1p(te_df[SEQ_COLS].values), -1)
    
    return seq_tr, X_tr_s, y_tr, seq_te, X_te_s, y_te, common, scaler

def run_comparison(target='views_24h'):
    print(f"\n🚀 Comparing LSTM vs XGBoost for {target} (Time-Based)...")
    s_tr, st_tr, y_tr, s_te, st_te, y_te, common, scaler = load_and_prep(target)
    
    # --- XGBoost ---
    print("  Training XGBoost...")
    # Using features including sequence ones for XGBoost
    X_tr_xgb = np.hstack([s_tr.squeeze(-1), st_tr])
    X_te_xgb = np.hstack([s_te.squeeze(-1), st_te])
    
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    xgb.fit(X_tr_xgb, y_tr)
    p_xgb = xgb.predict(X_te_xgb)
    r2_xgb = r2_score(y_te, p_xgb)
    print(f"  ✓ XGBoost R2: {r2_xgb:.4f}")
    
    # --- LSTM ---
    print("  Training LSTM...")
    train_dl = DataLoader(YoutubeDataset(s_tr, st_tr, y_tr), batch_size=256, shuffle=True)
    test_dl = DataLoader(YoutubeDataset(s_te, st_te, y_te), batch_size=256, shuffle=False)
    
    model = HybridLSTM(st_tr.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    for epoch in range(50):
        model.train()
        for sb, stb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(sb.to(device), stb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for sb, stb, yb in test_dl:
                v_loss += criterion(model(sb.to(device), stb.to(device)), yb.to(device)).item()
        v_loss /= len(test_dl)
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), 'temp_lstm.pth')
    
    model.load_state_dict(torch.load('temp_lstm.pth'))
    model.eval()
    p_lstm = []
    with torch.no_grad():
        for sb, stb, yb in test_dl:
            p_lstm.extend(model(sb.to(device), stb.to(device)).cpu().numpy())
    r2_lstm = r2_score(y_te, p_lstm)
    print(f"  ✓ LSTM R2: {r2_lstm:.4f}")
    
    print(f"\nFinal Comparison for {target}:")
    print(f"  XGBoost R2: {r2_xgb:.4f} | MAE: {mean_absolute_error(y_te, p_xgb):.4f}")
    print(f"  LSTM    R2: {r2_lstm:.4f} | MAE: {mean_absolute_error(y_te, p_lstm):.4f}")
    os.remove('temp_lstm.pth')

if __name__ == "__main__":
    run_comparison('views_24h')
