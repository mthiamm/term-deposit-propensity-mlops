"""Send random customers from the raw dataset to the running API.

python scripts/send_traffic.py --n 300 --delay 0.1
"""

import argparse
import json
import time
import urllib.error
import urllib.request

import pandas as pd

FIELDS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=300)
parser.add_argument("--delay", type=float, default=0.1)
parser.add_argument("--url", default="http://localhost:8000/predict")
args = parser.parse_args()

df = pd.read_csv("data/raw/bank-full.csv", sep=";")[FIELDS].sample(
    args.n, random_state=None
)
rows = json.loads(df.to_json(orient="records"))  # plain Python types

ok = bad = 0
for row in rows:
    req = urllib.request.Request(
        args.url, json.dumps(row).encode(), {"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req)
        ok += 1
    except urllib.error.HTTPError:
        bad += 1  # e.g. 422 if a raw value isn't allowed by the schema
    time.sleep(args.delay)
print(f"sent {ok + bad}: {ok} ok, {bad} rejected")
