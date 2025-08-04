import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def model_and_dataset_configs(config):
    # Model configs
    config.base_log_path = "/network/scratch/p/parham.saremi/rl4med-out"
    config.pretrained.model = "runwayml/stable-diffusion-v1-5"
    config.unet_path = f"{config.base_log_path}/published/pretrained_unet.safetensors"
    config.classifier_h6_path = f"{config.base_log_path}/published/artifact_classifier_train_H6.pth"
    config.classifier_h4_path = f"{config.base_log_path}/published/artifact_classifier_test_H4.pth"
    
    config.dataset = ml_collections.ConfigDict()
    config.dataset.csv_path = 'splits/isic2019_test.csv'
    config.dataset.image_dir = '/home/mila/p/parham.saremi/scratch/rl4med-models/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input'

    # For eval.py
    config.load_from = f"{config.base_log_path}/published/rl4med-checkpoint_50"

    return config

def prompt_image_alignment_classifier_isic():

    config = base.get_config()

    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    config.num_epochs = 200


    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }

    config = model_and_dataset_configs(config)

    # for this experiment, I reserved 2 GPUs for LLaVA inference so only 6 could be used for DDPO. the total number of
    # samples per epoch is 8 * 8 * 4 = 256.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 8

    # again, this one is harder to optimize, so I used (8 * 8) / (4 * 8) = 2 gradient updates per epoch.
    config.train.batch_size = 4 # 64 / 4 = 16 steps per epoch (per GPU)
    config.train.gradient_accumulation_steps = 8

    # prompting
    config.prompt_fn = "isic_all"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "isic_classifier_reward"

    config.run_name = "isic_classifier_reward_main"
    config.logdir = f"{config.base_log_path}/logs"
    config.resume_from = f"{config.base_log_path}/logs/isic_classifier_reward_main/"

    return config

def prompt_image_alignment_classifier_isic_eval():

    # TODO: changed the config to the base config and test
    config = prompt_image_alignment_classifier_isic()

    # True to use finetuned lora weights, False to use initial stable diffusion. 
    config.use_lora = False

    # Fixed seed for reproducibility. (Each process has a different seed)
    config.seed = 88 # mohamed will use 78, 88

    # mixed precision. options are "fp16", "bf16", and "no". half-precision speeds up inference significantly.
    config.mixed_precision = "fp16"

    # Num_samples should be a multiple of num_gpus*batch_size*num_processes
    # num_samples  -> total amount of samples you want to create
    num_samples_per_prompt = 960 # has to be a multiple of num_processes * batch_size
    config.sample.num_samples_per_prompt = num_samples_per_prompt
    config.sample.num_samples = num_samples_per_prompt * 32 # 32 is the number of prompts
    # batch size per model replica
    config.sample.batch_size = 20

    config.skip_idx = 0 # skip the first n steps: useful for resuming sampling. This will be multiplied by step_size (which is batch_size * num_processes)

    # prompting
    config.prompt_fn = "isic_all_idx"
    # Directory to log, images, prompts rewards, etc, make sure the folder name is informative of the model checkpoint you used
    config.logdir = f"{config.base_log_path}/samples/SD_samples"

    return config

def prompt_image_alignment_classifier_isic_eval_rl():
    # TODO: changed the config to the base config and test
    config = prompt_image_alignment_classifier_isic_eval()

    config.use_lora = True
    # Directory to log, images, prompts rewards, etc, make sure the folder name is informative of the model checkpoint you used
    config.logdir = f"{config.base_log_path}/samples/RL_samples"

    return config

def preprocess_isic_real_data():
    config = prompt_image_alignment_classifier_isic_eval()
    config.logdir = f'{config.base_log_path}/real-images-test'
    return config


def evaluate_metric_sd():
    config = base.get_config()
    config = model_and_dataset_configs(config)
    config.logdir = f"{config.base_log_path}/metric_logs"
    config.batch_size = 256 
    config.real_path = f"{config.base_log_path}/real-images-test"
    config.fake_path = f"{config.base_log_path}/published/script-samples/checkpoint_SD_1000_seed88"
    config.prompts = []
    return config

def evaluate_metric_rl():
    config = base.get_config()
    config = model_and_dataset_configs(config)
    config.logdir = f"{config.base_log_path}/metric_logs"
    config.batch_size = 256 
    config.real_path = f"{config.base_log_path}/real-images-test"
    config.fake_path = f"{config.base_log_path}/published/script-samples/checkpoint_RL_1000_seed88"
    config.prompts = []
    return config

def get_config(name):
    return globals()[name]()
