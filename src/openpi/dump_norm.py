from openpi.training.config import get_config
import json
import os

cfg = get_config("pi05_realman_finetune")
dataset = cfg.data.create()
stats = dataset.compute_norm_stats()

out_path = "./outputs/pi05_realman_finetune/norm_stats.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path,"w",encoding="utf-8") as f:
    json.dump(stats,f,indent=2)
print("Saved norm stats ->", out_path)