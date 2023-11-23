import gc
import wandb
import multiprocessing

from constants.wandb import WANDB_VERBOSE
from utils.data_reader import read_target_set_names
from utils.core import chunks, combinations 
from utils.debugging import logger

from ..constants import MODES, SAMPLING_METHODS
from .agent import Agent

def define_config(target_set_name: str, mode: str):
    return {
        "dataset": "galaxies",
        "workers": multiprocessing.cpu_count(),
        "target_set": target_set_name,
        "mode": mode,
        "gamma": 0.99,
        "update_interval": 20,
        "actor_lr": 0.00003,
        "critic_lr": 0.00003,
        "icm_lr": 0.05,
        "lstm_steps": 3,
        "eval_interval": 10,
        "curiosity_ratio": 0.0,
        "counter_curiosity_ratio": 0.0,
        "operators": ["by_facet", "by_superset", "by_neighbors", "by_distribution"],
        "utility_mode": None,
        "utility_weights": [0.333, 0.333, 0.334],
    }

def train_policies_from_target_sets(episodes: int, batch_size: int = 3):
    for sampling_method in SAMPLING_METHODS:
        target_set_names = read_target_set_names(sampling_method)
        for combination_chunk in chunks(combinations([MODES, target_set_names]), batch_size):
            for mode, target_set_name in combination_chunk:
                logger.info(f"Training policy [target={target_set_name}, mode={mode}]")
                config = define_config(target_set_name, mode)
                agent_name = f"{target_set_name}/{mode}"
                if WANDB_VERBOSE:
                    wandb.init(
                        project="xeda",
                        name=agent_name,
                        id=wandb.util.generate_id(),
                        config=config,
                    )
                agent = Agent(
                    "pipeline",
                    agent_name,
                    target_set_name,
                    mode, 
                    config
                )
                agent.train(episodes)
                logger.info(f"Trained policy [target={target_set_name}, mode={mode}]")
                del agent
                gc.collect()
