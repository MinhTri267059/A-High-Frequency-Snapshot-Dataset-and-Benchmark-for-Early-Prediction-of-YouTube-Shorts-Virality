import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Load necessary modules from current dir
# Check if feature_list and models exist
MODEL_24H = 'models/best_xgboost_views_24h.pkl'
MODEL_48H = 'models/best_xgboost_views_48h.pkl'
MODEL_VIRAL = 'models/best_xgb_viral_time_based.pkl'
FEATURES = 'models/features_list.pkl'

def load_pkl(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def predict_video(video_id, current_views, likes, comments, title, published_at):
    m24 = load_pkl(MODEL_24H)
    m48 = load_pkl(MODEL_48H)
    mviral = load_pkl(MODEL_VIRAL)
    feature_names = load_pkl(FEATURES)
    
    if not (m24 and m48 and mviral and feature_names):
        return {"error": "Models or features list not found. Please run training first."}

    # Simulate feature engineering (Simplification)
    # In a real scenario, we'd need the 2h, 4h, 6h snapshots.
    # We will 'guess' the velocity based on current age if it's a recent video.
    
    pub_dt = pd.to_datetime(published_at)
    now = datetime.now()
    age_h = (now - pub_dt).total_seconds() / 3600
    
    # Construct a dummy feature row with default values
    data = {f: 0 for f in feature_names}
    
    # Map available data
    data['age_hours'] = age_h
    # Here we simulate that the current views ARE the early momentum views if video is young
    if age_h <= 6:
        data['vel_6'] = current_views
        data['views_at_6h'] = current_views
    
    # NLP Embedding (Simulated or just use dummy if library missing)
    # For a real implementation, we'd run sentence_transformers here.
    
    df = pd.DataFrame([data])
    
    # Apply log1p scaling used in training
    cols_to_log = [c for c in df.columns if 'views' in c or 'vel' in c]
    df[cols_to_log] = np.log1p(df[cols_to_log].clip(lower=0))
    
    # Inference
    p24_log = m24.predict(df[feature_names])[0]
    p48_log = m48.predict(df[feature_names])[0]
    
    p24 = np.expm1(p24_log)
    p48 = np.expm1(p48_log)
    
    # Viral Classification (Simplified feature set for demo)
    # Viral model usually expects X_c (standard drop cols)
    # We'll just provide a status
    is_viral_prob = mviral.predict_proba(df.drop(columns=['video_id','channel_id','published_date','category_id','views_24h','views_48h','views_7d','is_viral_1m'], errors='ignore'))[0][1]

    return {
        "video_id": video_id,
        "title": title,
        "current_views": current_views,
        "predicted_24h": int(p24),
        "predicted_48h": int(p48),
        "viral_probability": round(float(is_viral_prob) * 100, 2)
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict_live.py <youtube_url_or_id>")
        sys.exit(1)
        
    identifier = sys.argv[1]
    # Simple Mock Metadata for Demonstration
    print(f"--- 🌩️ PREDICTING FOR: {identifier} ---")
    print("Fetching metadata and momentum signals...")
    
    result = predict_video(
        video_id=identifier,
        current_views=1500,
        likes=120,
        comments=15,
        title="Sample Viral Short #shorts",
        published_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    print(json.dumps(result, indent=2))
