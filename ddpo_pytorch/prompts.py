from importlib import resources
import os
import functools
import random
import inflect

IE = inflect.engine()
ASSETS_PATH = resources.files("ddpo_pytorch.assets")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `ddpo_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or ddpo_pytorch.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}

def convert_prompt_to_metadata_isic(prompt):
    """
    Convert a single caption/prompt (e.g. 
    "a dermoscopic image with melanoma (MEL) showing hair, gel bubble")
    back into the metadata labels: [hair, gel_bubble, ruler, ink, MEL, NV].

    Returns:
        dict: {
            "label": [hair, gel_bubble, ruler, ink, MEL, NV]
        }
    """
    
    # Check disease
    MEL = 1 if "melanoma (MEL)" in prompt else 0
    NV = 1 if "melanocytic nevus (NV)" in prompt else 0
    
    # Check artifacts (note underscores become spaces, e.g. 'gel bubble')
    hair = 1 if "hair" in prompt else 0
    gel_bubble = 1 if "gel bubble" in prompt else 0
    ruler = 1 if "ruler" in prompt else 0
    ink = 1 if "ink" in prompt else 0
    
    return {
        "label": [hair, gel_bubble, ruler, ink, MEL, NV]
    }

def isic_all():
    prompt = from_file("isic_prompts.txt")[0]
    return prompt, convert_prompt_to_metadata_isic(prompt)

def isic_all_idx(selected_idx):
    prompts = _load_lines("isic_prompts.txt")
    prompt = prompts[selected_idx]
    return prompt, convert_prompt_to_metadata_isic(prompt)

def counting(nouns_file, low, high):
    nouns = _load_lines(nouns_file)
    number = IE.number_to_words(random.randint(low, high))
    noun = random.choice(nouns)
    plural_noun = IE.plural(noun)
    prompt = f"{number} {plural_noun}"
    metadata = {
        "questions": [
            f"How many {plural_noun} are there in this image?",
            f"What animal is in this image?",
        ],
        "answers": [
            number,
            noun,
        ],
    }
    return prompt, metadata
