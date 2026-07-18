import json

with open('.bot_state.json') as f:
    state = json.load(f)

seen = state.get('channel_job_seen', {}).get('1513654492539785406', [])
print("channel_job_seen type:", type(seen), "length:", len(seen))

if isinstance(seen, list):
    daanaa_seen = [x for x in seen if 'daanaa' in str(x).lower()]
elif isinstance(seen, dict):
    daanaa_seen = {k: v for k, v in seen.items() if 'daanaa' in str(k).lower() or 'daanaa' in str(v).lower()}
else:
    daanaa_seen = []

print("Daanaa entries:", daanaa_seen)

# Also check a sample of what seen entries look like
sample = seen[:3] if isinstance(seen, list) else list(seen.items())[:3]
print("Sample seen entries:", sample)
