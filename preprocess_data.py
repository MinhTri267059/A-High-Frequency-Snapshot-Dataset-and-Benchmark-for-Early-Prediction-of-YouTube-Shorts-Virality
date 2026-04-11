import os
import json
import datetime
from collections import defaultdict
import re

base_path = "/Users/minhtri86/Downloads/raw/2026"
output_dir = "/Users/minhtri86/Downloads/raw/yt-dashboard/public/data"
os.makedirs(output_dir, exist_ok=True)

# Heuristic category mapping
CATEGORY_KEYWORDS = {
    "Entertainment": ["entertainment", "show", "viral", "trending", "shorts"],
    "Comedy": ["funny", "comedy", "joke", "lol", "laugh", "humor"],
    "Music": ["music", "song", "dance", "remix", "beat", "singer", "artist"],
    "Gaming": ["gaming", "game", "minecraft", "roblox", "pubg", "freefire", "gamer"],
    "Education": ["education", "learn", "how to", "tutorial", "science", "facts"],
    "Sports": ["sports", "football", "cricket", "soccer", "basketball", "workout", "gym"],
    "People & Blogs": ["vlog", "life", "blog", "family", "friends"]
}

def guess_category(title, hashtags):
    title_lower = title.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return cat
        if any(tag.lower() in keywords for tag in hashtags):
            return cat
    return "Others"

def guess_country(title):
    # Rough Unicode heuristic
    has_hindi = any('\u0900' <= char <= '\u097F' for char in title)
    has_bengali = any('\u0980' <= char <= '\u09FF' for char in title)
    if has_hindi or has_bengali: return "India"
    
    has_thai = any('\u0e00' <= char <= '\u0e7f' for char in title)
    if has_thai: return "Thailand"
    
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in title)
    if has_chinese: return "China"
    
    has_japanese = any('\u3040' <= char <= '\u30ff' for char in title)
    if has_japanese: return "Japan"
    
    has_korean = any('\uac00' <= char <= '\ud7af' for char in title)
    if has_korean: return "South Korea"
    
    has_viet = any(char in "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ" for char in title.lower())
    if has_viet: return "Vietnam"
    
    return "Global/English"

# Data structures
video_latest_stats = {}
video_snapshots = defaultdict(list)
hashtag_counts = defaultdict(int)
hashtag_views = defaultdict(int)
category_stats = defaultdict(lambda: {"count": 0, "views": 0})
country_stats = defaultdict(lambda: {"count": 0, "views": 0})

hashtag_regex = re.compile(r'#(\w+)')

# Parsing limit
max_files = 200000
count = 0

print("Scanning files and enriching data...")
for root, _, files in os.walk(base_path):
    for filename in files:
        if filename.endswith(".json"):
            count += 1
            if count > max_files: break
            
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                vid_id = data.get('video_id')
                if not vid_id: continue
                
                crawled_at = data.get('crawled_at', '')
                view_count = int(data.get('statistics', {}).get('viewCount', 0))
                like_count = int(data.get('statistics', {}).get('likeCount', 0))
                
                # Snapshot for timeseries
                video_snapshots[vid_id].append({"timestamp": crawled_at, "views": view_count, "likes": like_count})
                
                if vid_id not in video_latest_stats:
                    title = data.get('title', '')
                    hashtags = hashtag_regex.findall(title)
                    category = guess_category(title, hashtags)
                    country = guess_country(title)
                    
                    video_latest_stats[vid_id] = {
                        "video_id": vid_id,
                        "channel_id": data.get('channel_id'),
                        "title": title,
                        "published_at": data.get('published_at', ''),
                        "latest_timestamp": crawled_at,
                        "views": view_count,
                        "likes": like_count,
                        "category": category,
                        "country": country,
                        "hashtags": hashtags
                    }
                else:
                    if crawled_at > video_latest_stats[vid_id]["latest_timestamp"]:
                        video_latest_stats[vid_id]["latest_timestamp"] = crawled_at
                        video_latest_stats[vid_id]["views"] = view_count
                        video_latest_stats[vid_id]["likes"] = like_count

            except Exception: pass
    if count > max_files: break

# Aggregations
for vid, info in video_latest_stats.items():
    v = info['views']
    # Hashtags
    tags = list(set([t.lower() for t in info['hashtags']]))
    for tag in tags:
        hashtag_counts[tag] += 1
        hashtag_views[tag] += v
    
    # Categories & Countries
    category_stats[info['category']]["count"] += 1
    category_stats[info['category']]["views"] += v
    country_stats[info['country']]["count"] += 1
    country_stats[info['country']]["views"] += v

# Summaries
top_videos = sorted(list(video_latest_stats.values()), key=lambda x: x['views'], reverse=True)[:500]
top_hashtags = sorted([{"tag": tag, "count": count, "views": hashtag_views[tag]} for tag, count in hashtag_counts.items()], key=lambda x: x['views'], reverse=True)[:50]

summary_data = {
    "top_videos": top_videos,
    "top_hashtags": top_hashtags,
    "category_analytics": [{"name": k, "count": v["count"], "views": v["views"]} for k, v in category_stats.items()],
    "country_analytics": [{"name": k, "count": v["count"], "views": v["views"]} for k, v in country_stats.items()],
    "total_videos_analyzed": len(video_latest_stats)
}

with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary_data, f, ensure_ascii=False)

# Timeseries
os.makedirs(os.path.join(output_dir, 'timeseries'), exist_ok=True)
for info in top_videos[:200]: # Limit to top 200 for faster timeseries loading
    vid = info['video_id']
    with open(os.path.join(output_dir, 'timeseries', f"{vid}.json"), 'w', encoding='utf-8') as f:
        json.dump({"video_id": vid, "title": info['title'], "history": sorted(video_snapshots[vid], key=lambda x: x['timestamp'])}, f, ensure_ascii=False)

print("Enriched data processing complete.")
