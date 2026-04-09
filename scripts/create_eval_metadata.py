import json
import os
import sys

sys.path.insert(0, "Isaac-GR00T")

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config

# load the dataset to get its metadata in the exact format groot expects
data_config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
ds = LeRobotSingleDataset(
    dataset_path="data/libero/libero_spatial_no_noops_1.0.0_lerobot",
    modality_configs=data_config.modality_config(),
    transforms=data_config.transform(),
    embodiment_tag="new_embodiment",
    video_backend="torchvision_av",
)

# dump metadata in the exact format TrainRunner uses
metadata_json = {ds.tag: ds.metadata.model_dump(mode="json")}

for exp in ["baseline", "vanilla_cl", "rscl"]:
    cfg_dir = f"outputs/{exp}/checkpoint-60000/experiment_cfg"
    os.makedirs(cfg_dir, exist_ok=True)
    with open(f"{cfg_dir}/metadata.json", "w") as f:
        json.dump(metadata_json, f, indent=2)
    print(f"created {cfg_dir}/metadata.json")
