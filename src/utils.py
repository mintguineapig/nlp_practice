import omegaconf
from omegaconf import OmegaConf

def load_config(config_path="/root/venv39/nlp_practice/configs/configs.yaml") -> omegaconf.DictConfig:
    return OmegaConf.load(config_path)