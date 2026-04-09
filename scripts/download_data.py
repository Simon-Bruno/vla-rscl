import os
import time
from huggingface_hub import snapshot_download

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DATA_DIR = "data/libero"

os.makedirs(DATA_DIR, exist_ok=True)

for suite in SUITES:
    name = f"IPEC-COMMUNITY/{suite}_no_noops_1.0.0_lerobot"
    local = f"{DATA_DIR}/{suite}_no_noops_1.0.0_lerobot"
    print(f"downloading {suite}...")

    while True:
        try:
            snapshot_download(name, repo_type="dataset", local_dir=local)
            print(f"{suite} done")
            break
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"rate limited, waiting 5 min...")
                time.sleep(300)
            else:
                raise

    # copy modality config
    meta_dir = os.path.join(local, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    modality_src = "Isaac-GR00T/examples/Libero/modality.json"
    modality_dst = os.path.join(meta_dir, "modality.json")
    if os.path.exists(modality_src):
        import shutil
        shutil.copy2(modality_src, modality_dst)
        print(f"copied modality.json to {meta_dir}")

    time.sleep(30)

print("all datasets downloaded")
