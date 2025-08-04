from accelerate import Accelerator
from accelerate.logging import get_logger
from absl import app, flags
import lpips
import numpy as np
import os
import json
from ml_collections import config_flags
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.distributions import Normal
from tqdm import tqdm

from artifact_classifier import ArtifactClassifier 


NUM_CLASSES = 4
CLASS_NAMES =  ['hair', 'gel_bubble', 'ruler', 'ink']
    
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/dgx.py", "Metrics evaluation.")

logger = get_logger(__name__)


class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset to load images and corresponding latent tensors.
    """
    def __init__(self, input_folder: str, prompts: list[str] = None, transform=None):
        self.image_folder = os.path.join(input_folder, "images")
        self.transform = transform if transform else transforms.ToTensor()

        if prompts is None or len(prompts)==0:
            prompts = [d for d in os.listdir(self.image_folder) if os.path.isdir(os.path.join(self.image_folder, d))]
        self.prompts = prompts

        self.samples = []
        for prompt in prompts:
            prompt_image_folder = os.path.join(self.image_folder, prompt)
            if not os.path.exists(prompt_image_folder):
                continue

            image_files = sorted([f for f in os.listdir(prompt_image_folder) if f.endswith(".jpg") or f.endswith(".png")])
            for img_file in image_files:
                self.samples.append(os.path.join(prompt_image_folder, img_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")          # image is of size 3,512,512
        image = self.transform(image)     
        true_labels = torch.tensor(
            [c in img_path.split("/")[-2] for c in CLASS_NAMES], dtype=torch.int, device=torch.device("cpu")
        ) # Tensor of shape (num_classes,)
        return {'img': image, 'true_labels': true_labels}


def get_dataloader(input_folder: str, prompts: list[str] = None, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4, transform=None):
    """
    Returns a DataLoader for the ImageDataset.
    """
    dataset = ImageDataset(input_folder, prompts, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


@torch.no_grad
def artifact_prelevance_rate(config, fake_loader, accelerator=None):
    """
    Computes the prevalence rates of artifacts in generated images.

    - `true_label_ratio`: The fraction of images where all true labels are present in the predicted labels.
    - `other_artifact_ratio`: The fraction of images where extra artifacts are predicted beyond the true labels.

    Returns:
        (true_label_ratio, other_artifact_ratio)
    """

    checkpoint_path = config.classifier_h4_path
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

    model = ArtifactClassifier(num_classes=NUM_CLASSES).to(accelerator.device)
    model = accelerator.prepare(model)
    print("epoch", checkpoint_data["epoch"])
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    model.requires_grad_(False)

    total_samples = 0

    total_apr_samples = 0

    for batch in (tqdm(fake_loader, desc="Processing APR") if accelerator.is_main_process else fake_loader):
        images = batch["img"]  # (B, 3, 512, 512)
        true_labels = batch["true_labels"]  # (B, 4)

        logits = model(images)  # (B, 4)
        preds = (logits.sigmoid() > 0.5).int() 

        apr = (preds == true_labels).all(dim=1)  # Boolean tensor (B,) where True means all true labels are present
        total_apr_samples += apr.sum().item()  # Count how many in (B,) are True

        total_samples += images.shape[0]

    return total_apr_samples / total_samples


@torch.no_grad
def calculate_fid(real_loader, fake_loader, feature_dim=2048, key='img', accelerator=None):
    """
    Computes the FID score between two datasets.
        feature_dim should be on of 64, 192, 768, 2048
    """
    fid = FrechetInceptionDistance(feature=feature_dim).to(accelerator.device)
    
    for batch in (tqdm(real_loader, desc="Processing FID real") if accelerator.is_main_process else real_loader):
        fid.update((batch[key]*256).to(torch.uint8), real=True)
    for batch in (tqdm(fake_loader, desc="Processing FID fake") if accelerator.is_main_process else fake_loader):
        fid.update((batch[key]*256).to(torch.uint8), real=False)
    
    return fid.compute().item()

@torch.no_grad
def calculate_lpips(real_loader, fake_loader, key='img', model="alex", accelerator=None):
    """
    Computes the LPIPS score between two datasets for consecutive pairs: O(N)
    """

    loss_fn = lpips.LPIPS(net=model).to(accelerator.device)
    total_lpips, total_samples = 0.0, 0

    for real_batch, fake_batch in (tqdm(zip(real_loader, fake_loader), desc="Processing lpips sequential") if accelerator.is_main_process else zip(real_loader, fake_loader)):
        real_images = real_batch[key]
        fake_images = fake_batch[key]

        B1, C, H, W = real_images.shape
        B2, _, _, _ = fake_images.shape

        real_images, fake_images = real_images[:min(B1,B2)], fake_images[:min(B1,B2)]

        batch_size = real_images.shape[0]
        total_samples += batch_size

        # Normalize images to [-1, 1] if they are in [0, 1]
        real_images = real_images * 2 - 1.0
        fake_images = fake_images * 2 - 1.0

        # Compute LPIPS
        lpips_loss = loss_fn(real_images, fake_images) # 4,1,1,1  -> mean() -> []  -> item() -> lpips_loss
        total_lpips += lpips_loss.mean().item() * batch_size

    return total_lpips / total_samples, total_samples


def main(_):
    config = FLAGS.config
    logger.info(f"\n{config}")

    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="rl4med",
            config=config.to_dict(),
        )
        
    real_loader = get_dataloader(config.real_path, batch_size=config.batch_size, prompts=config.prompts)
    fake_loader = get_dataloader(config.fake_path, batch_size=config.batch_size, prompts=config.prompts)

    print(config.fake_path, config.real_path)
        
    metrics = {}

    apr_loader = get_dataloader(config.fake_path, batch_size=config.batch_size, prompts=config.prompts, transform=
                                transforms.Compose([
                                    transforms.Resize((512, 512)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))
    real_loader, fake_loader, apr_loader = accelerator.prepare(real_loader, fake_loader, apr_loader)

    # Compute Artifact Prevalence Rate
    logger.info(f"Starting Artifact Prevalence Rate evaluation..")
    apr = artifact_prelevance_rate(config, apr_loader, accelerator=accelerator)
    apr = accelerator.gather(torch.tensor([apr], device=accelerator.device)).mean().item()

    logger.info(f"Artifact Prelevance Ratio Mine: {apr}")
    metrics["APR_m"]=apr

    restricted_prompts = [
        "a_dermoscopic_image_with_melanoma_(MEL)_showing_hair",
        "a_dermoscopic_image_with_melanoma_(MEL)_showing_gel_bubble",
        "a_dermoscopic_image_with_melanoma_(MEL)_showing_ruler",
        "a_dermoscopic_image_with_melanoma_(MEL)_showing_ink",
        "a_dermoscopic_image_with_melanocytic_nevus_(NV)_showing_hair",
        "a_dermoscopic_image_with_melanocytic_nevus_(NV)_showing_gel_bubble",
        "a_dermoscopic_image_with_melanocytic_nevus_(NV)_showing_ruler",
        "a_dermoscopic_image_with_melanocytic_nevus_(NV)_showing_ink",
    ]

    for restricted_prompt in restricted_prompts:
        apr_loader = get_dataloader(
            config.fake_path, 
            batch_size=config.batch_size, 
            prompts=[restricted_prompt], 
            transform=transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )

        apr_loader = accelerator.prepare(apr_loader)
        logger.info(f"Starting Artifact Prevalence Rate evaluation for {restricted_prompt}..")
        APR_rest = artifact_prelevance_rate(config, apr_loader, accelerator=accelerator)
        APR_rest = accelerator.gather(torch.tensor([APR_rest], device=accelerator.device)).mean().item()
        logger.info(f"Artifact Prelevance Ratio for {restricted_prompt}: {APR_rest}")


    real_prompts = real_loader.dataset.prompts

    lpips_images_all = 0.0
    num_samples_all = 0

    for prompt in real_prompts:
        real_loader_temp = get_dataloader(config.real_path, batch_size=config.batch_size, prompts=[prompt])
        fake_loader_temp = get_dataloader(config.fake_path, batch_size=config.batch_size, prompts=[prompt])

        real_loader_temp, fake_loader_temp = accelerator.prepare(real_loader_temp, fake_loader_temp)

        lpips_images, num_samples = calculate_lpips(real_loader_temp, fake_loader_temp, key='img', accelerator=accelerator)
        lpips_images = accelerator.gather(torch.tensor([lpips_images], device=accelerator.device)).mean().item()
        num_samples = accelerator.gather(torch.tensor([num_samples], device=accelerator.device)).sum().item()
        lpips_images_all += lpips_images * num_samples
        num_samples_all += num_samples
    metrics["LPIPS"] = lpips_images_all / num_samples_all
    logger.info(f"LPIPS for images: {metrics['LPIPS']}")

    # Compute FID for images
    logger.info(f"Starting FID evaluation on images..")
    fid_images = calculate_fid(real_loader, fake_loader, key='img', feature_dim=2048, accelerator=accelerator)
    fid_images_all = accelerator.gather(torch.tensor([fid_images], device=accelerator.device)).mean().item()
    logger.info(f"FID for images: {fid_images_all}")
    metrics["FID"] = fid_images_all

    if accelerator.is_main_process:
        log_file = os.path.join(config.logdir, "metrics_log.jsonl")  # Append metrics_log.jsonl to log_path
        os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure the directory exists

        # Open the file in append mode and write a new JSON object as a separate line
        with open(log_file, "a") as f:
            f.write(json.dumps({"metrics": metrics}) + "\n")  # Append JSONL format

        logger.info(f"Metrics appended to {log_file}")

if __name__ == "__main__":
    app.run(main)
