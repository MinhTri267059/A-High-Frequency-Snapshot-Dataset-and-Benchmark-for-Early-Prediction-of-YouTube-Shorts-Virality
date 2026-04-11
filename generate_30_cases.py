import pandas as pd
import numpy as np
import pickle
import json
import os
import glob

# Models
MODEL_24H = 'models/best_xgboost_views_24h.pkl'
MODEL_48H = 'models/best_xgboost_views_48h.pkl'
MODEL_VIRAL_CHAN = 'models/best_xgb_viral_channel_based.pkl'
MODEL_VIRAL_TIME = 'models/best_xgb_viral_time_based.pkl'
FEATURES_LIST = 'models/features_list.pkl'

def load_pkl(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def find_title_and_thumb(video_id, channel_id):
    # Try searching 2026 folders based on IDs
    # Path pattern: 2026/{month}/{day}/{hour}/{channel_id}/{video_id}.json
    pattern = f"2026/*/*/*/{channel_id}/{video_id}.json"
    matches = glob.glob(pattern)
    if matches:
        try:
            with open(matches[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('title', 'Unknown'), f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg"
        except:
            pass
    
    # Backup: check common names in snapshots? (Too slow)
    return f"Video {video_id}", f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg"

def main():
    m24 = load_pkl(MODEL_24H)
    m48 = load_pkl(MODEL_48H)
    v_chan = load_pkl(MODEL_VIRAL_CHAN)
    v_time = load_pkl(MODEL_VIRAL_TIME)
    features = load_pkl(FEATURES_LIST)
    
    if not (m24 and m48 and features):
        print("Required models or feature list missing.")
        return

    # Load test sets
    test_chan_df = pd.read_csv('clean_test_channel.csv').fillna(0)
    test_time_df = pd.read_csv('clean_test_time.csv').fillna(0)
    
    all_cases = []
    
    for df, group_name in [(test_chan_df, "Channel-Based"), (test_time_df, "Time-Based")]:
        # Sample 15
        samples = df.sample(n=min(15, len(df)), random_state=42)
        
        # Feature engineering (Regression)
        X = samples[features].copy()
        cols_to_log = [c for c in X.columns if 'views' in c or 'vel' in c]
        X[cols_to_log] = np.log1p(X[cols_to_log].clip(lower=0))
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Predict Views
        pred_24_log = m24.predict(X)
        pred_48_log = m48.predict(X)
        pred_24 = np.expm1(pred_24_log)
        pred_48 = np.expm1(pred_48_log)
        
        # Viral prediction
        v_model = v_time if group_name == "Time-Based" else v_chan
        drop_cols_c = ['video_id', 'channel_id', 'published_date', 'category_id', 'views_24h', 'views_48h', 'views_7d', 'is_viral_1m']
        X_c = samples.drop(columns=[c for c in drop_cols_c if c in samples.columns], errors='ignore')
        X_c = X_c.replace([np.inf, -np.inf], 0).fillna(0)
        
        pred_viral = v_model.predict(X_c)
        pred_viral_prob = v_model.predict_proba(X_c)[:, 1]

        for i in range(len(samples)):
            row = samples.iloc[i]
            vid_id = row['video_id']
            chan_id = row['channel_id']
            
            title, thumb = find_title_and_thumb(vid_id, chan_id)
            
            # Calculate Deterministic Status
            actual_24h = row.get('views_24h', 0)
            pred_24h_val = pred_24[i]
            
            # Tiered rule
            def get_label(views):
                if views >= 500000: return "VIRAL"
                if views >= 50000: return "POTENTIAL"
                return "NORMAL"
            
            actual_label = get_label(actual_24h)
            pred_label = get_label(pred_24h_val)
            
            status = "SUCCESS"
            if actual_label != "NORMAL" and pred_label == "NORMAL":
                status = "MISS"
            elif actual_label == "NORMAL" and pred_label != "NORMAL":
                status = "FALSE_ALARM"

            case = {
                "video_id": vid_id,
                "title": title,
                "thumbnail": thumb,
                "channel": chan_id,
                "split": group_name,
                "actual_24h": int(actual_24h),
                "pred_24h": int(pred_24h_val),
                "actual_48h": int(row.get('views_48h', 0)),
                "pred_48h": int(pred_48[i]),
                "actual_label": actual_label,
                "pred_label": pred_label,
                "error_24h_pct": float(abs(pred_24h_val - actual_24h) / max(1, actual_24h) * 100),
                "status": status
            }
            all_cases.append(case)
            
    output_path = 'yt-dashboard/public/data/prediction_cases.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    
    print(f"Repaired 30 test cases at {output_path}")

if __name__ == "__main__":
    main()
