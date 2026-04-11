from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime
from googleapiclient.discovery import build
import isodate
from dotenv import load_dotenv

# --- Cấu hình API ---
load_dotenv() # Load biến môi trường từ .env
app = FastAPI(title="YouTube Real-Time Deep Inference API")

# API KEY từ .env
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Hỗ trợ kết nối từ React (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Tải Model ---
MODEL_24H_PATH = "best_xgboost_views_24h.pkl"
MODEL_48H_PATH = "best_xgboost_views_48h.pkl"
FEATURES_PATH = "features_list.pkl"

try:
    with open(MODEL_24H_PATH, "rb") as f:
        model_24h = pickle.load(f)
    with open(MODEL_48H_PATH, "rb") as f:
        model_48h = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        features_list = pickle.load(f)
    
    # Load NLP Model
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("🚀 Model Chạy Thực Tế V5 đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi nạp model: {e}")

class VideoInput(BaseModel):
    title: str
    current_views: float
    published_at: str
    channel_avg_views: float = 0
    duration_sec: float = 60

@app.get("/fetch_metadata/{video_id}")
async def fetch_metadata(video_id: str):
    try:
        # 1. Fetch Video Details
        vid_req = youtube.videos().list(part="snippet,statistics,contentDetails", id=video_id)
        vid_res = vid_req.execute()
        
        if not vid_res['items']:
            raise HTTPException(status_code=404, detail="Video not found")
            
        v_item = vid_res['items'][0]
        snippet = v_item.get('snippet', {})
        stats = v_item.get('statistics', {})
        content = v_item.get('contentDetails', {})
        
        channel_id = snippet.get('channelId')
        
        # 2. Fetch Channel Stats (FOR DYNAMIC AVG VIEWS)
        chan_req = youtube.channels().list(part="statistics", id=channel_id)
        chan_res = chan_req.execute()
        
        channel_avg = 1000 # Default
        if chan_res['items']:
            c_stats = chan_res['items'][0]['statistics']
            total_views = int(c_stats.get('viewCount', 0))
            total_vids = int(c_stats.get('videoCount', 1))
            channel_avg = total_views / max(1, total_vids)

        dur_str = content.get('duration', 'PT0S')
        dur_sec = isodate.parse_duration(dur_str).total_seconds()
        
        thumbs = snippet.get('thumbnails', {})
        thumb_url = (thumbs.get('high') or thumbs.get('medium') or thumbs.get('default', {})).get('url', '')
        
        return {
            "title": snippet.get('title', 'Unknown'),
            "current_views": int(stats.get('viewCount', 0)),
            "published_at": snippet.get('publishedAt', '').replace('Z', '').replace('T', ' '),
            "thumbnail": thumb_url,
            "duration_sec": dur_sec,
            "channel_id": channel_id,
            "channel_avg_views": channel_avg
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fetch_channel_latest/{channel_id}")
async def fetch_channel_latest(channel_id: str):
    try:
        req = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            order="date",
            maxResults=8,
            type="video"
        )
        res = req.execute()
        
        videos = []
        for item in res.get('items', []):
            v_id = item['id']['videoId']
            # Chú thích: Để tiết kiệm quota, ta chỉ lấy metadata cơ bản, 
            # không gọi video.list riêng lẻ cho từng cái ở đây
            videos.append({
                "video_id": v_id,
                "title": item['snippet']['title'],
                "thumbnail": item['snippet']['thumbnails']['medium']['url'],
                "published_at": item['snippet']['publishedAt']
            })
            
        return videos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(data: VideoInput):
    try:
        pub_dt = pd.to_datetime(data.published_at)
        now = datetime.now()
        age_h = (now - pub_dt).total_seconds() / 3600
        
        input_dict = {f: 0 for f in features_list}
        input_dict['channel_avg_views'] = data.channel_avg_views
        input_dict['duration_sec'] = data.duration_sec
        input_dict['age_hours'] = age_h
        
        if age_h < 24:
            if age_h >= 6: input_dict['views_at_6h'] = data.current_views
            elif age_h >= 4: input_dict['views_at_4h'] = data.current_views
            elif age_h >= 2: input_dict['views_at_2h'] = data.current_views
            else: input_dict['views_at_0h'] = data.current_views
        else:
            input_dict['views_at_6h'] = data.current_views

        df = pd.DataFrame([input_dict])

        embeddings = nlp_model.encode([data.title])[0]
        for i, val in enumerate(embeddings):
            df[f'title_emb_{i}'] = val

        cols_to_log = [c for c in df.columns if 'views' in c or 'vel' in c]
        for col in cols_to_log:
            df[col] = np.log1p(df[col].clip(0))

        df_final = df.reindex(columns=features_list, fill_value=0)

        p24_log = model_24h.predict(df_final)[0]
        p48_log = model_48h.predict(df_final)[0]
        
        res_24h = int(np.expm1(p24_log))
        res_48h = int(np.expm1(p48_log))
        
        # --- LOGIC CẢI TIẾN: Dự báo tăng trưởng ---
        # Nếu model dự báo quá thấp (do thiếu dữ liệu lịch sử momentum), 
        # ta áp dụng một baseline growth giả định dựa trên channel_avg_views
        baseline_growth = (data.channel_avg_views / 24) * max(0, 24 - age_h)
        final_24h = int(max(data.current_views + baseline_growth, res_24h))
        final_48h = int(max(final_24h + (data.channel_avg_views / 24) * 24, res_48h))

        viral_label = "NORMAL"
        if final_24h >= 500000: viral_label = "VIRAL"
        elif final_24h >= 50000: viral_label = "POTENTIAL"

        return {
            "title": data.title,
            "prediction_24h": final_24h,
            "prediction_48h": final_48h,
            "viral_status": viral_label,
            "age_hours": round(age_h, 1),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
