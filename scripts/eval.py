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
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from safetensors.torch import load_file
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from filelock import FileLock

from artifact_classifier import ArtifactClassifier 

CLASS_NAMES = ['hair', 'gel_bubble', 'ruler', 'ink', 'MEL', 'NV']


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

    if not config.run_name:
        config.run_name = unique_id

    if config.load_from:
        config.load_from = os.path.normpath(os.path.expanduser(config.load_from))
        if "checkpoint_" not in os.path.basename(config.load_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.load_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.load_from}")
            config.load_from = os.path.join(
                config.load_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    print("logging to directory:",os.path.join(config.logdir))
    os.makedirs(os.path.join(config.logdir), exist_ok=True)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir),
        automatic_checkpoint_naming=False,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
    )
    # Create json file
    json_file_path = os.path.join(config.logdir, "image_log.json")
    lock_file = json_file_path + ".lock"

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="rl4med",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )

        # Initialize the JSON file
        if not os.path.exists(json_file_path):
            with open(json_file_path, 'w') as f:
                json.dump([], f)  # Empty list to start the log

    # All processes wait until the main process has initialized the JSON file
    accelerator.wait_for_everyone()
    
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )

    state_dict = load_file(config.unet_path)
    pipeline.unet.load_state_dict(state_dict)

    # freeze parameters of models
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
        pipeline.unet.set_attn_processor(lora_attn_procs)

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)

        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        unet = pipeline.unet

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model,
                revision=config.pretrained.revision,
                subfolder="unet",
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(
                AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            )
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            state_dict = load_file(config.unet_path)
            pipeline.unet.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)

    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.per_prompt_stat_tracking.buffer_size,
            config.per_prompt_stat_tracking.min_count,
        )

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    unet = accelerator.prepare(unet)

    executor = futures.ThreadPoolExecutor(max_workers=2)

    logger.info("***** Sampling *****")
    logger.info(f"  Total number of samples = {config.sample.num_samples}")
    logger.info(f"  Batch size per gpu = {config.sample.batch_size}")

    if config.load_from:
        accelerator.load_state(config.load_from) # once the model state dict is loaded, register_load_state_pre_hook is called
        epochs_trained = int(config.load_from.split("_")[-1]) + 1
        logger.info(f"Loading model from {config.load_from} that was trained for {epochs_trained} epochs")
    else:
        print("Using Amar's pretrained diffusion model")

    # Classifier model
    checkpoint_path = config.classifier_h6_path
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

    model = ArtifactClassifier(num_classes=6).to(accelerator.device)
    model.load_state_dict(checkpoint_data["model_state_dict"])

    model.eval()
    model.requires_grad_(False)

    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)

    if config.reward_fn == "isic_classifier_reward":
        reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)(model, image_size=512)
    else:
        reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()
    
    assert (config.sample.num_samples % (accelerator.num_processes * config.sample.batch_size)) == 0, "Ensure num_samples is evenly divisible by batch_size*num_gpus."
    step_size = accelerator.num_processes * config.sample.batch_size 
    print("Accelerator num_processes:",accelerator.num_processes,"sample batch_size:",config.sample.batch_size)
    prompt_index = 0

    all_logs = []

    for batch in tqdm(
        range(config.skip_idx * step_size, config.sample.num_samples, step_size), 
        disable=not accelerator.is_local_main_process,
        desc="Processing batches on GPU"):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        prompts = []

        """
        GPU = 4
        batch_size = 2
        step_size = 8
        num_samples = 32 * 48 (per_prompt * num_prompts)
        
        batch = 0, 8, 16, 24, 32, 40, 48, ...
        prompt_index = 0, 0, 0, 0, 1, 1, 1, ...
        
        """

        prompt_index = batch // config.sample.num_samples_per_prompt
        # generate prompts
        # print(prormpt_metadata)
        prompts, prompt_metadata = zip(
            *[
                prompt_fn(prompt_index)
                for _ in range(config.sample.batch_size)
            ]
        )

        # encode prompts
        prompt_ids = pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

        # sample
        with autocast():
            images, _, latents, log_probs = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
            )

        latents = torch.stack(
            latents, dim=1
        )  # (batch_size, num_steps + 1, 4, 64, 64)

        log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
        timesteps = pipeline.scheduler.timesteps.repeat(
            config.sample.batch_size, 1
        )  # (batch_size, num_steps)

        # compute rewards asynchronously
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
        # yield to to make sure reward computation starts
        time.sleep(0)

        samples.append(
            {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "timesteps": timesteps,
                "latents": latents[
                    :, :-1
                ],  # each entry is the latent before timestep t
                "next_latents": latents[
                    :, 1:
                ],  # each entry is the latent after timestep t
                "log_probs": log_probs,
                "rewards": rewards,
            }
        )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}


        for i, (image, prompt, reward) in enumerate(zip(images, prompts, rewards)):

            sample_id = (batch) + (accelerator.local_process_index) +i*accelerator.num_processes   
            
            image_filename = f"{sample_id}.jpg"

            image_folder = os.path.join(config.logdir, "images", prompt.replace(" ", "_"))
            os.makedirs(image_folder, exist_ok=True)
            image_path = os.path.join(image_folder, image_filename)

            pil = Image.fromarray(
                (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            pil.save(image_path)  # Save the image as JPG

            image_data = {
                "sampleID": sample_id,
                "prompt": prompt,
                "reward": "{:.4f}".format(reward),
                "path_to_img": image_path,
            }

            for key, value in reward_metadata.items():
                if isinstance(value, np.ndarray):
                    value = list(map(lambda x: "{:.3f}".format(x), value[i].tolist()))  # Convert each element to a string with 3 decimal places
                image_data[key] = value  # Store the converted value in image_data
            all_logs.append(image_data)

    # Append to the JSON log file
    with FileLock(lock_file):
        with open(json_file_path, 'r+') as f:
            data = json.load(f)
            data.extend(all_logs)
            f.seek(0)
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    app.run(main)
