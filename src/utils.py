import omegaconf
from omegaconf import OmegaConf

def load_config(config_path="/workspace/Experiment 1-2025/exp_1/configs/configs.yaml") -> omegaconf.DictConfig:
    return OmegaConf.load(config_path)