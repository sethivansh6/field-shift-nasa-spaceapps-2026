"""
Step 3a: request ECOSTRESS data for the demo fields through the AppEEARS API.

Login: your Earthdata username/password. Set them once per terminal session:
    export EARTHDATA_USER=your_username
    export EARTHDATA_PASS=your_password
(or the script will ask for them; nothing is saved to disk except the task id)

Commands (run from the FARMS folder):
    python src/appeears.py list       # show your existing AppEEARS tasks (did the website one go through?)
    python src/appeears.py layers     # check which layers we will request
    python src/appeears.py submit     # submit the point request
    python src/appeears.py status     # check progress
    python src/appeears.py download   # download CSVs into data/raw/ecostress/
"""
import getpass
import json
import os
import sys
import time

import requests

from config import FIELDS, RAW_ECOSTRESS

API = "https://appeears.earthdatacloud.nasa.gov/api"
TASK_FILE = RAW_ECOSTRESS / "appeears_task.json"
TASK_NAME = "farms_lusatia_demo"
START, END = "07-09-2018", "11-01-2026"      # MM-DD-YYYY

# product -> layers we want (matched case-insensitively against what AppEEARS offers)
WANTED = {
    "ECO_L4T_ESI.002":  ["ESI", "PET"],
    "ECO_L3T_JET.002":  ["ETdaily"],
    "ECO_L2T_LSTE.002": ["LST", "QC", "cloud", "water"],
    "ECO_L3T_SM.002":   ["SM"],
}


def login():
    user = os.environ.get("EARTHDATA_USER") or input("Earthdata username: ")
    pw = os.environ.get("EARTHDATA_PASS") or getpass.getpass("Earthdata password: ")
    r = requests.post(f"{API}/login", auth=(user, pw), timeout=60)
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['token']}"}


def available_layers(product):
    r = requests.get(f"{API}/product/{product}", timeout=60)
    r.raise_for_status()
    return list(r.json().keys())


def pick_layers():
    chosen, problems = [], []
    for product, wanted in WANTED.items():
        try:
            avail = available_layers(product)
        except Exception as e:
            problems.append(f"{product}: cannot read layer list ({e})")
            continue
        lower = {a.lower(): a for a in avail}
        for w in wanted:
            match = lower.get(w.lower()) or next((a for a in avail if w.lower() in a.lower()), None)
            if match:
                chosen.append({"product": product, "layer": match})
            else:
                problems.append(f"{product}: no layer like '{w}'. Available: {avail}")
    return chosen, problems


def cmd_layers():
    chosen, problems = pick_layers()
    for c in chosen:
        print(f"  {c['product']:18s} {c['layer']}")
    for p in problems:
        print("  WARN", p)


def cmd_list():
    h = login()
    r = requests.get(f"{API}/task", headers=h, timeout=60)
    r.raise_for_status()
    tasks = r.json()
    if not tasks:
        print("No AppEEARS tasks found for this account.")
    for t in tasks:
        print(f"  {t.get('task_id')}  {t.get('task_name')}  status={t.get('status')}  created={t.get('created')}")


def cmd_submit():
    chosen, problems = pick_layers()
    for p in problems:
        print("  WARN", p)
    if not chosen:
        sys.exit("No layers found, not submitting.")
    task = {
        "task_type": "point",
        "task_name": TASK_NAME,
        "params": {
            "dates": [{"startDate": START, "endDate": END}],
            "layers": chosen,
            "coordinates": [{"id": f["id"], "category": "demo",
                             "latitude": f["lat"], "longitude": f["lon"]} for f in FIELDS],
        },
    }
    h = login()
    r = requests.post(f"{API}/task", json=task, headers=h, timeout=120)
    if r.status_code >= 400:
        sys.exit(f"Submit failed ({r.status_code}): {r.text}")
    info = r.json()
    TASK_FILE.write_text(json.dumps({"task_id": info["task_id"], "layers": chosen}, indent=2))
    print(f"Submitted. task_id={info['task_id']} status={info.get('status')}")
    print("Check with: python src/appeears.py status")


def task_id():
    if not TASK_FILE.exists():
        sys.exit("No task recorded. Run: python src/appeears.py submit")
    return json.loads(TASK_FILE.read_text())["task_id"]


def cmd_status():
    h = login()
    r = requests.get(f"{API}/task/{task_id()}", headers=h, timeout=60)
    r.raise_for_status()
    t = r.json()
    print(f"status: {t.get('status')}  progress: {t.get('progress', {})}")


def cmd_download():
    h = login()
    tid = task_id()
    r = requests.get(f"{API}/bundle/{tid}", headers=h, timeout=60)
    if r.status_code == 404:
        sys.exit("Bundle not ready yet. Check: python src/appeears.py status")
    r.raise_for_status()
    files = r.json()["files"]
    n = 0
    for f in files:
        name = f["file_name"].split("/")[-1]
        if not name.endswith(".csv"):
            continue
        url = f"{API}/bundle/{tid}/{f['file_id']}"
        with requests.get(url, headers=h, stream=True, allow_redirects=True, timeout=300) as d:
            d.raise_for_status()
            with open(RAW_ECOSTRESS / name, "wb") as out:
                for chunk in d.iter_content(1 << 16):
                    out.write(chunk)
        n += 1
        print(f"  saved {name}")
        time.sleep(0.5)
    print(f"Downloaded {n} CSV files to {RAW_ECOSTRESS}")


COMMANDS = {"list": cmd_list, "layers": cmd_layers, "submit": cmd_submit,
            "status": cmd_status, "download": cmd_download}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()