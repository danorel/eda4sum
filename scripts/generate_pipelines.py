import json
import pathlib
import tensorflow as tf

from tqdm import tqdm

from app.model_manager import ModelManager
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from utils.debugging import logger


def scan_directories(path: pathlib.Path):
    return list(filter(lambda f: f.is_dir(), path.glob("*")))


root = pathlib.Path.cwd()
model_folder = root / "saved_models" / "sdss"
modes = ["scattered", "concentrated"]
target_set_types = ["node_sampling"]

logger.info(f"Started gathering trained policies")
trained_policies = []
for target_set_type in tqdm(scan_directories(model_folder)):
    for target_set in scan_directories(target_set_type):
        logger.info(f"Started gathering target set policy {target_set}")
        for mode in modes:
            for model in scan_directories(target_set / mode):
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

print(trained_policies)


def iterate(prev_request):
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
