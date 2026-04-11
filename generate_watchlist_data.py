import pandas as pd
import json
import os
import glob

# For demo, we keep these channels in focus
CHANNELS = ['UC2rxmP4xIG8aPArxvx5fXxQ', 'UCDpeY-XhvHTW-vW2BfB__mw', 'UC6T9BjlemmwDVeCPko1OzaQ']

def main():
    watchlist_data = {}
    
    for ch_id in CHANNELS:
        # Find videos for this channel
        pattern = f"2026/*/*/*/{ch_id}/*.json"
        files = glob.glob(pattern)
        
        videos = []
        # Take up to 5 unique videos
        unique_vids = set()
        for f in sorted(files, reverse=True): # Newest firstish
            if len(unique_vids) >= 5: break
            try:
                with open(f, 'r') as jf:
                    data = json.load(jf)
                    vid_id = data.get('video_id')
                    if vid_id and vid_id not in unique_vids:
                        unique_vids.add(vid_id)
                        videos.append({
                            "video_id": vid_id,
                            "title": data.get('title', f"Video {vid_id}"),
                            "thumbnail": f"https://i.ytimg.com/vi/{vid_id}/mqdefault.jpg",
                            "published_at": data.get('published_at'),
                            "views": data.get('statistics', {}).get('viewCount', 0),
                            "predicted_viral": 1 if int(data.get('statistics', {}).get('viewCount', 0)) > 500000 else 0
                        })
            except:
                continue
        
        watchlist_data[ch_id] = videos

    output_path = 'yt-dashboard/public/data/watchlist_history.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(watchlist_data, f, ensure_ascii=False, indent=2)
    
    print(f"Generated watchlist history at {output_path}")

if __name__ == "__main__":
    main()
