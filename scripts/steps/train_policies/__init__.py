import gc

from utils.data_reader import read_target_set_names
from utils.core import chunks, combinations 
from utils.debugging import logger

from ..constants import MODES, SAMPLING_METHODS
from .agent import Agent


def train_policies_from_target_sets(episodes: int, batch_size: int = 3):
    for sampling_method in SAMPLING_METHODS:
        target_set_names = read_target_set_names(sampling_method)
        for combination_chunk in chunks(combinations([MODES,  target_set_names]), batch_size):
            for mode, target_set_name in combination_chunk:
                logger.info(f"Training policy [target={target_set_name}, mode={mode}]")
                agent = Agent(
                    env_name="pipeline", target_set_name=target_set_name, mode=mode
                )
                agent.train(episodes)
                logger.info(f"Trained policy [target={target_set_name}, mode={mode}]")
                del agent
                gc.collect()
