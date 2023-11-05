import pathlib
import typing as t

from tqdm import tqdm

from scripts.agent import Agent
from utils.debugging import logger

root = pathlib.Path.cwd()
target_set_dir = root / "rl" / "targets" / "sdss"

def read_target_sets(target_set_type: str) -> t.Iterator[str]:
    logger.info(f"Started reading target set dir: {target_set_dir}")
    for target_set_path in tqdm((target_set_dir / target_set_type).rglob("*.json")):
        target_set_file_path = "/".join(str(target_set_path).split("/")[-3:]).split(
            "."
        )[0]
        logger.info(f"Found target set: {target_set_file_path}")
        yield target_set_file_path
    logger.info(f"Finished reading target set dir: {target_set_dir}")


modes = ["scattered", "concentrated"]
target_set_types = ["node_sampling"]

for mode in modes:
    for target_set_type in target_set_types:
        for target_set_name in read_target_sets(target_set_type):
            logger.info(
                f"Started training a policy [target_set_name={target_set_name}, target_set_type={target_set_type}, mode={mode}]"
            )
            agent = Agent(env_name="pipeline", target_set_name=target_set_name, mode=mode)
            agent.train()
            logger.info(
                f"Finished training the policy [target_set_name={target_set_name}, target_set_type={target_set_type}, mode={mode}]"
            )
