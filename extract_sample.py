import os
import json
import glob
from collections import defaultdict
import datetime

# Find a handful of JSON files to inspect
base_path = "/Users/minhtri86/Downloads/raw/2026"
files = []
# we use glob to get files
for root, dirs, _files in os.walk(base_path):
    for f in _files:
        if f.endswith('.json'):
            files.append(os.path.join(root, f))
            if len(files) >= 5000:
                break
    if len(files) >= 5000:
        break

print(f"Found {len(files)} files to sample.")

# Structure of our consolidated data
# We'll group by video_id to see growth.
videos = defaultdict(list)
for f in files:
    try:
        with open(f, 'r') as fp:
            data = json.load(fp)
            videos[data['video_id']].append(data)
    except Exception as e:
        pass

print(f"Total unique videos in sample: {len(videos)}")

# Let's save a summary of what we found to a temp file so we can view it
sample_videos = list(videos.values())[:10]
with open("/Users/minhtri86/Downloads/raw/data_sample.json", "w") as fp:
    json.dump(sample_videos, fp, indent=2)

print("Sample saved to /Users/minhtri86/Downloads/raw/data_sample.json")
