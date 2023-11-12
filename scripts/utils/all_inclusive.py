import gc
import itertools
import json
import pathlib
import typing as t

from random import randrange
from tqdm import tqdm

from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from data_types.pipeline import RequestData
from scripts.utils.model_manager import ModelManager
from scripts.utils.agent import Agent
from utils.debugging import logger
from utils.data_writer import write_pipeline


root = pathlib.Path.cwd()

target_set_folder = root / "rl" / "targets" / "sdss"
model_folder = root / "saved_models"


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


def generate_pipeline(database_pipeline_cache, model_manager: ModelManager, prev_request: RequestData):
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
    next_request = RequestData.parse_obj(prev_request.dict())
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
    if prev_request.dataset_ids is None or len(prev_request.dataset_ids) == 0:
        datasets = [pipeline.get_dataset()]
    else:
        datasets = pipeline.get_groups_as_datasets(prev_request.dataset_ids)
    prediction = model_manager.get_prediction(
        datasets,
        prev_request.weights_mode,
        prev_request.target_items,
        prev_request.found_items_with_ratio,
        previous_set_states=prev_request.previous_set_states,
        previous_operation_states=prev_request.previous_operation_states,
    )
    next_request.dimensions.append(prediction.get("predictedDimension"))
    next_request.found_items_with_ratio = prediction.get("foundItemsWithRatio")
    next_request.previous_operations.append(prediction.get("predictedOperation"))
    next_request.previous_set_states = prediction.get("setStates")
    next_request.previous_operation_states = prediction.get("operationStates")
    if prediction.get("predictedSetId") is not None:
        next_request.input_set_id = prediction.get("predictedSetId")
        next_request.seen_sets.append(prediction.get("predictedSetId"))
    return next_request


def scan_folders(path: pathlib.Path):
    return list(filter(lambda f: f.is_dir(), path.glob("*")))


def generate_node_description(database_pipeline_cache, request_data: RequestData):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[request_data.dataset_to_explore]
    greedy_summarizer = GreedySummarizer(pipeline)
    result_sets = greedy_summarizer.get_summary(
        min_set_size, 
        min_uniformity_target, 
        result_set_count
    )
    result = get_items_sets(result_sets, 
                            pipeline, 
                            True, 
                            False,
                            None, 
                            seen_sets=set(),
                            previous_dataset_ids=set(), 
                            utility_weights=[0.5, 0.5, 0], 
                            previous_operations=[])
    return result


def generate_pipelines_from_policies(min_pipeline_size: int, max_pipeline_size: int):
    logger.info(f"Reading precalculated dataset of pipelines")
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
    logger.info(f"Read precalculated dataset of pipelines")

    logger.info(f"Started generating trained pipelines")
    modes = ["scattered", "concentrated"]
    sampling_methods = ["node_sampling"]
    for mode, sampling_method in combinations([modes, sampling_methods]):
        for target_set_name in scan_folders(model_folder / "sampling" / sampling_method):
            with (target_set_name / mode / "info.json").open("r") as f:
                info = json.load(f)
            for model_path in scan_folders(target_set_name / mode):
                if "final" in str(model_path):
                    continue
                pipeline_size = randrange(min_pipeline_size, max_pipeline_size)
                pipeline, request_data = ([], RequestData(dataset_to_explore="galaxies",
                                                          utility_weights=info.get("utility_weights"),
                                                          decreasing_gamma=False,
                                                          galaxy_class_scores=None,
                                                          dimensions=[],
                                                          seen_sets=[],
                                                          previous_operations=[],
                                                          previous_operation_states=None,
                                                          previous_set_states=None))
                logger.info(f"Generating pipeline with {model_path} of size {pipeline_size}")
                for _ in range(pipeline_size):
                    model_manager = ModelManager(database_pipeline_cache["galaxies"], model_path)
                    request_data = generate_pipeline(database_pipeline_cache, 
                                                     model_manager, 
                                                     request_data)
                    pipeline.append(request_data.dict())
                pipeline_filename = f"{target_set_name.stem}_{mode}.json"
                write_pipeline(pipeline_filename, pipeline, sampling_method)
                logger.info(f"Finished gathering pipeline with {model_path}")
    logger.info(f"Finished generating trained pipelines")
