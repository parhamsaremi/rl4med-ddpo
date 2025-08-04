from PIL import Image
import io
import numpy as np
import torch

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F_vision
from torchvision import transforms
from torch import nn
from torch.cuda.amp import autocast
from torchvision import models


NUM_CLASSES = 6

def isic_classifier_reward(model, image_size=512):
    """
    Creates a function `_fn(images, prompts, metadata)` that:
      - Uses the *already-initialized* classifier `model` (which you load and prepare externally).
      - Resizes the images with `F_vision.resize` and normalizes with `F_vision.normalize`.
      - Runs the classifier to get multi-label predictions.
      - Compares those predictions with the ground-truth labels in `metadata` to produce a scalar reward.

    Args:
        model: An ArtifactClassifier (or similar) that has been loaded from a checkpoint and placed on
               the correct device (e.g., via accelerate.prepare(model)).
        image_size: Desired image size for resizing (default 512).

    Returns:
        A callable `_fn(images, prompts, metadata)` that computes rewards.
    """

    # Make sure the model is in eval mode and not tracking gradients
    # model.eval()
    # model.requires_grad_(False)

    # Hard-coded normalization stats (e.g. ImageNet/EfficientNet)
    mean_list = [0.485, 0.456, 0.406]
    std_list = [0.229, 0.224, 0.225]

    def _fn(images, prompts, metadata):
        """
        Args:
            images: A batch of images in either [0,1] float or [0..255] uint8, shape NCHW or NHWC.
            prompts: (Unused here, can be ignored or removed)
            metadata: A list of dicts, each with "label" -> [hair, gel_bubble, ruler, ink, MEL, NV].

        Returns:
            rewards: np.array of shape (N,) with scalar reward per image
            info: dict with auxiliary data (e.g. predictions, labels).
        """
        del prompts  # Not needed by this reward function

        # 1) If `images` is NumPy or on some other device, convert to tensor
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)
        # We do NOT specify .to(device) here. We assume "accelerator" or the caller
        # has placed images on the same device as the model. Or you can do:
        # images = images.to(next(model.parameters()).device)

        # 2) If it's uint8 [0..255], convert to float [0..1]
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
        else:
            # Assume already in [0..1]
            images = images.float()

        # 3) Ensure NCHW format
        if images.dim() == 4 and images.shape[1] not in (1, 3):
            images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # 4) Resize to (image_size, image_size)
        images = F_vision.resize(
            images,
            [image_size, image_size],
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        )

        # 5) Normalize in-place
        images = F_vision.normalize(images, mean_list, std_list, inplace=True)

        # 6) Gather labels: each metadata[i]["label"] -> [hair, gel_bubble, ruler, ink, MEL, NV]
        labels_list = [m["label"] for m in metadata]
        labels = torch.tensor(labels_list, dtype=torch.float32, device=images.device)

        # 7) Forward pass (mixed precision optional)
        with torch.no_grad(), autocast():
            outputs = model(images)  # shape (N, 6)

        # 8) Multi-label predictions
        preds = (torch.sigmoid(outputs) > 0.5).float()  # shape (N, 6)

        # 9) Calculate reward: fraction of correct labels
        correctness = (preds == labels).float()  # shape (N, 6)
        reward_per_image = correctness.mean(dim=1)  # shape (N,)

        # Return as numpy
        reward_per_image_np = reward_per_image.cpu().numpy()

        info = {
            "predictions": preds.cpu().numpy(),
            "labels": labels.cpu().numpy(),
        }

        return reward_per_image_np, info

    return _fn