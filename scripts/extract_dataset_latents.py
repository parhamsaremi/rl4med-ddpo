from collections import defaultdict
import contextlib
import os
import json
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import torch
import wandb
from functools import partial
import tqdm
from filelock import FileLock

import csv
from PIL import Image
import torchvision.transforms as transforms
from safetensors.torch import load_file
import torch.nn as nn
import torchvision.models as models

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

CLASS_NAMES = ['hair', 'gel_bubble', 'ruler', 'ink', 'MEL', 'NV']


# Make tqdm bar nice
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

def build_prompt_from_row(row):
    """
    Given a CSV row with columns:
      - MEL, NV, (or none)  -> disease
      - hair, gel_bubble, ruler, ink -> artifacts
    Return a text string like:
      "a dermoscopic image with melanoma (MEL) showing hair, ruler"
    or "a dermoscopic image of a normal skin with no visible artifacts"
    """
    mel = float(row.get("MEL", 0))
    nv  = float(row.get("NV", 0))
    if mel == 1.0:
        prefix = "a dermoscopic image with melanoma (MEL)"
    elif nv == 1.0:
        prefix = "a dermoscopic image with melanocytic nevus (NV)"
    else:
        prefix = "a dermoscopic image of a normal skin"
    
    artifact_columns = ["hair", "gel_bubble", "ruler", "ink"]
    artifacts_present = []
    for artifact in artifact_columns:
        val = row.get(artifact, "0")
        if float(val) == 1.0:
            artifact_name = artifact.replace("_", " ")
            artifacts_present.append(artifact_name)
    
    if len(artifacts_present) == 0:
        suffix = "with no visible artifacts"
    else:
        suffix = "showing " + ", ".join(artifacts_present)
    
    prompt = prefix + " " + suffix
    return prompt


def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id

    os.makedirs(config.logdir, exist_ok=True)

    json_file_path = os.path.join(config.logdir, "image_log.json")

    images_dir = os.path.join(config.logdir, "images")
    os.makedirs(images_dir, exist_ok=True)

    csv_path = config.dataset.csv_path 
    with open(csv_path, "r") as f:
        csv_reader = csv.DictReader(f)
        rows = list(csv_reader)

    all_logs = []
    for idx, row in tqdm(
        enumerate(rows),
        desc="Saving images",
        total=len(rows),
    ):
        prompt = build_prompt_from_row(row)
        folder_safe_prompt = prompt.replace("/", "_").replace(" ", "_")
        image_basename = row["image"]
        
        img_path_on_disk = os.path.join(config.dataset.image_dir, image_basename + ".jpg")
        if not os.path.exists(img_path_on_disk):
            img_path_on_disk = os.path.join(config.dataset.image_dir, image_basename + "_downsampled.jpg")
            if not os.path.exists(img_path_on_disk):
                logger.warning(f"Image not found: {img_path_on_disk}")
                continue

        pil_image = Image.open(img_path_on_disk).convert("RGB")


        images_subdir = os.path.join(config.logdir, "images", folder_safe_prompt)
        os.makedirs(images_subdir, exist_ok=True)

        out_img_filename = f"{idx}.jpg"

        out_img_path = os.path.join(images_subdir, out_img_filename)

        pil_image = pil_image.resize((512, 512))
        pil_image.save(out_img_path)


        image_data = {
            "sampleID": idx,
            "prompt": prompt,
            "path_to_img": out_img_path,
        }
        for col_name, col_value in row.items():
            image_data[col_name] = col_value
        all_logs.append(image_data)

    with open(json_file_path, "w") as jf:
        json.dump(all_logs, jf, indent=2)

    logger.info("All done! Encoded latents and updated JSON log.")


if __name__ == "__main__":
    app.run(main)