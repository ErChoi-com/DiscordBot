import os

import requests

token = os.getenv('discordtoken')
if not token:
    raise SystemExit("Missing env var 'discordtoken'.")
headers = {'Authorization': 'Bot ' + token}
allahu_id = '1513654492539785406'
ece_id = '1482891850439458857'

print("=== #allahu-akbar last 20 ===")
r = requests.get('https://discord.com/api/v10/channels/' + allahu_id + '/messages?limit=20', headers=headers)
print('status:', r.status_code)
if r.ok:
    for m in r.json():
        content = m['content'][:200].replace('\n', ' ')
        print('  [' + m['timestamp'] + '] ' + content)
else:
    print('  error:', r.text)

print()
print("=== #ece-job-goon last 20 ===")
r2 = requests.get('https://discord.com/api/v10/channels/' + ece_id + '/messages?limit=20', headers=headers)
print('status:', r2.status_code)
if r2.ok:
    for m in r2.json():
        content = m['content'][:200].replace('\n', ' ')
        print('  [' + m['timestamp'] + '] ' + content)
else:
    print('  error:', r2.text)
