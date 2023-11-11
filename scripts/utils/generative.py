import gc
import itertools
import json
import pathlib
import tensorflow as tf
import typing as t

from tqdm import tqdm

from app.model_manager import ModelManager
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from scripts.utils.agent import Agent
from utils.debugging import logger


root = pathlib.Path.cwd()

target_set_folder = root / "rl" / "targets" / "sdss"
model_folder = root / "saved_models" / "sdss"


def combinations(lst):
    return list(itertools.product(*lst))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_target_sets(sampling_folder: str = "sampling") -> t.Iterator[str]:

    logger.info(f"Started reading target set folder: {target_set_folder}")
    for target_set_path in tqdm((target_set_folder / sampling_folder).rglob("**/*.json")):
        target_set_file_path = "/".join(str(target_set_path).split("/")[-3:]).split(
            "."
        )[0]
        logger.info(f"Found target set: {target_set_file_path}")
        yield target_set_file_path
    logger.info(f"Finished reading target set folder: {target_set_folder}")


def train_policies(episodes: int, batch_size: int = 3):

    modes = ["scattered", "concentrated"]
    target_set_names = read_target_sets()

    for combination_chunk in chunks(combinations([modes, target_set_names]), batch_size):
        for mode, target_set_name in combination_chunk:
            logger.info(f"Training policy [target={target_set_name}, mode={mode}]")
            agent = Agent(env_name="pipeline", target_set_name=target_set_name, mode=mode)
            agent.train(episodes)
            logger.info(f"Trained policy [target={target_set_name}, mode={mode}]")
            del agent
            gc.collect()


def scan_folders(path: pathlib.Path):
    return list(filter(lambda f: f.is_dir(), path.glob("*")))


def generate_pipeline(database_pipeline_cache, model_manager, prev_request):
    """
    return {
        "predictedOperation": operation,
        "predictedDimension": dimension,
        "predictedSetId": set_id,
        "foundItemsWithRatio": state_encoder.found_items_with_ratio,
        "setStates": new_set_states,
        "operationStates": new_operation_states,
        "reward": reward
    }
    """
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
    if len(prev_request.dataset_ids) == 0:
        datasets = [pipeline.get_dataset()]
    else:
        datasets = pipeline.get_groups_as_datasets(prev_request.dataset_ids)
    next_request = model_manager.get_prediction(
        datasets,
        prev_request.weights_mode,
        prev_request.target_items,
        prev_request.found_items_with_ratio,
        previous_set_states=prev_request.previous_set_states,
        previous_operation_states=prev_request.previous_operation_states,
    )
    return next_request


def generate_pipelines():
    modes = ["scattered", "concentrated"]
    target_set_types = ["node_sampling"]

    logger.info(f"Started gathering trained policies")
    trained_policies = []
    for target_set_type in tqdm(scan_folders(model_folder)):
        for target_set in scan_folders(target_set_type):
            logger.info(f"Started gathering target set policy {target_set}")
            for mode in modes:
                for model in scan_folders(target_set / mode):
                    logger.info(f"Started gathering policy {model}")
                    if (model / "set_op_counters.json").exists():
                        with (model / "set_op_counters.json").open() as f:
                            set_op_counters = json.load(f)
                    else:
                        set_op_counters = {}
                    trained_policies.append(
                        {
                            "set": tf.keras.models.load_model(f"{model}/set_actor"),
                            "operation": tf.keras.models.load_model(f"{model}/operation_actor"),
                            "set_op_counters": set_op_counters,
                        }
                    )
                    logger.info(f"Finished gathering policy {model}")
            logger.info(f"Finished gathering target set policy {target_set}")
    logger.info(f"Finished gathering trained policies")

    logger.info(f"Started reading precalculated dataset of pipelines")
    data_folder = "./app/data"
    database_pipeline_cache = {}
    database_pipeline_cache["galaxies"] = PipelineWithPrecalculatedSets(
        database_name="sdss",
        initial_collection_names=["galaxies"],
        data_folder=data_folder,
        discrete_categories_count=10,
        min_set_size=10,
        exploration_columns=[
            "galaxies.u",
            "galaxies.g",
            "galaxies.r",
            "galaxies.i",
            "galaxies.z",
            "galaxies.petroRad_r",
            "galaxies.redshift",
        ],
        id_column="galaxies.objID",
    )
    logger.info(f"Finished reading precalculated dataset of pipelines")

    model_manager = ModelManager(database_pipeline_cache["galaxies"])
    
    pipeline = generate_pipeline(database_pipeline_cache, model_manager, {})
    print(pipeline)
