import json
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
path = "toxic_test.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data))