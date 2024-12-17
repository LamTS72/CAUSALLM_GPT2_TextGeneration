from configparser import ConfigParser

config_parser = ConfigParser()
config_parser.read("../config.ini")
parser = config_parser["API_KEY"]


class ConfigDataset():
    PATH_DATASET = "huggingface-course/codeparrot-ds-"
    REVISION = None


class ConfigModel():
    BATCH_SIZE = 8
    CHUNK_SIZE = 128
    CONTEXT_LENGTH = 128
    VOCAB_SIZE = 50000
    MODEL_NAME = 'gpt2'
    MODEL_TOKENIZER = "huggingface-course/code-search-net-tokenizer"
    TRAIN_SIZE = 10000
    RATIO = 0.1
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.1
    EPOCHS = 8
    GRADIENT_ACCUMULATION_STEPS = 8
    EVAL_STEPS = 5_000
    METRICs = ""
    PATH_TENSORBOARD = "runs/data_run"
    PATH_SAVE = "causal_lm"
    NUM_WARMUP_STEPS = 1_000

class ConfigHelper():
    TOKEN_HF = parser["HF_KEY"]
    AUTHOR = "Chessmen"