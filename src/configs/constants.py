import re
from typing import Literal

### Main arg MODE types
Mode = Literal["pretrain", "downstream_train", "downstream_eval"]

### Datasets
BASE_DATASETS = {"ptb_xl", "mimic_iv", "code15", "cpsc", "csn",}

# All allowed datasets, just add to the union
ALLOWED_DATA = set().union(BASE_DATASETS)

### Configs
CONFIG_FOLDER = "src/configs"

### Runs
RUNS_FOLDER = "src/runs"

DATA_DIR = "../data"

PTB_ORDER = ["I", "II", "III", "aVL", "aVR", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PTB_INDEPENDENT_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
PTB_INDEPENDENT_IDX = [0,1,6,7,8,9,10,11]

### Models

TRANSFORMER_MODELS = {
    "trans_discrete_decoder": {
        "find_unused_parameters": False,
    },
    "trans_discrete_encoder": {
        "find_unused_parameters": False,
    },
    "trans_discrete_seq2seq": {
        "find_unused_parameters": False,
    },
    "trans_continuous_nepa": {
        "find_unused_parameters": False,
    },
    "trans_continuous_dit": {
        "find_unused_parameters": False,
    },
    "trans_discrete_decoder_fm": {
        "find_unused_parameters": False,
    },
}

MAE_MODELS = {
    "mae_vit": {
        "find_unused_parameters": False,
    },
}

MERL_MODEL = {
    "merl": {
        "find_unused_parameters": False,
    },
}

MLAE_MODELS = {
    "mlae": {
        "find_unused_parameters": False,
    },
}

MTAE_MODELS = {
    "mtae": {
        "find_unused_parameters": False,
    },
}

ST_MEM_MODELS = {
    "st_mem": {
        "find_unused_parameters": False,
    },
}

VISION_ENCODERS = {
    "clip-vit-base-patch32": {
        "model": "openai/clip-vit-base-patch32",
        "tokenizer": "openai/clip-vit-base-patch32",
        "find_unused_parameters": False,
        "strict": True,
        "model_hidden_size": None,
        "projection_dim": None,
        "encoder_input_len": 77,
    },
    "siglip-base-patch16-224": {
        "model": "google/siglip-base-patch16-224",
        "tokenizer": "google/siglip-base-patch16-224",
        "find_unused_parameters": False,
        "strict": True,
        "model_hidden_size": None,
        "projection_dim": None,
        "encoder_input_len": 64,
    },
    "vit-base-patch16-224-in21k": {
        "model": "google/vit-base-patch16-224-in21k",
        "tokenizer": "google/vit-base-patch16-224-in21k",
        "find_unused_parameters": False,
        "strict": True,
        "model_hidden_size": None,
        "projection_dim": None,
        "num_patches": None,
        "encoder_input_len": None,
    },
}


# Encoders
ECG_ENCODERS = {
    "st_mem": {
        "find_unused_parameters": False,
        "strict": False,
        "model_hidden_size": 768,
        "projection_dim": 256,
        "encoder_input_len": None,
    },
    "mtae": {
        "find_unused_parameters": False,
        "strict": False,
        "model_hidden_size": 768,
        "projection_dim": 256,
        "encoder_input_len": None,
    },
    "mlae": {
        "find_unused_parameters": False,
        "strict": False,
        "model_hidden_size": 768,
        "projection_dim": 256,
        "encoder_input_len": None,
    },
}


# TEXT CLEANER
LEADING_PREFIX_RE = re.compile(
    r"^\s*(?:[:：]\s*|(?:user|assistant|human|gpt|model|system|q|a)\s*:\s*)+",
    flags=re.IGNORECASE,
)
TAG_RE = re.compile(r"<\s*(?:ecg|image)\s*>\s*\n?", flags=re.IGNORECASE)
IMAGE_WORD_RE = re.compile(r"\b(image)\b", flags=re.IGNORECASE)


def case_preserving_signal(m: re.Match) -> str:
    w = m.group(1)
    return "Signal" if w[0].isupper() else "signal"