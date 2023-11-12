import json

from random import randrange

from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from data_types.pipeline import RequestData
from utils.core import combinations, scan_folders
from utils.debugging import logger
from utils.data_writer import write_pipeline

from ..constants import MODEL_FOLDER, MODES, SAMPLING_METHODS
from .model_manager import ModelManager
from .operators import by_distribution, by_facet, by_neighbors, by_superset

def _get_initial_request_data(database_pipeline_cache, info, dataset: str = "galaxies"):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[dataset]
    request_data = RequestData(
        input_set_id=-1,
        dataset_to_explore=dataset,
        dataset_ids=[],
        weights_mode="custom",
        utility_weights=info.get("utility_weights"),
        found_items_with_ratio={},
        decreasing_gamma=False,
        galaxy_class_scores=None,
        dimensions=list(pipeline.ordered_dimensions.keys()),
        seen_sets=[],
        previous_operations=[],
        previous_operation_states=None,
        previous_set_states=None,
        get_scores=True,
        get_predicted_scores=True
    )
    return request_data


def next_pipeline_iter(
    database_pipeline_cache, model_manager, prev_request: RequestData
):
    if len(prev_request.previous_operations):
        operator = prev_request.previous_operations[-1]
    else:
        operator = "by_facet"

    """
    prediction contains:
    {
        "predictedOperation": operation,
        "predictedDimension": dimension,
        "predictedSetId": set_id,
        "foundItemsWithRatio": state_encoder.found_items_with_ratio,
        "setStates": new_set_states,
        "operationStates": new_operation_states,
        "reward": reward
    }
    """
    if operator == "by_distribution":
        prediction = by_distribution(
            database_pipeline_cache, model_manager, prev_request
        )
    elif operator == "by_facet":
        prediction = by_facet(database_pipeline_cache, model_manager, prev_request)
    elif operator == "by_neighbors":
        prediction = by_neighbors(database_pipeline_cache, model_manager, prev_request)
    elif operator == "by_superset":
        prediction = by_superset(database_pipeline_cache, model_manager, prev_request)
    else:
        raise Exception("Operator not implemented")

    if not prediction:
        raise ValueError("Prediction failed")

    next_request = RequestData.parse_obj(prev_request.dict())
    next_set_id = prediction.get("predictedSetId")
    if next_set_id is not None:
        next_request.input_set_id = next_set_id
        next_request.seen_sets.append(next_set_id)
    next_request.found_items_with_ratio = prediction.get("foundItemsWithRatio")
    next_request.previous_operations.append(prediction.get("predictedOperation"))
    next_request.previous_set_states = prediction.get("setStates")
    next_request.previous_operation_states = prediction.get("operationStates")
    next_request.dataset_ids = list(map(lambda x: x.get("id"), prediction.get("sets")))

    next_node = {
        "selectedSetId": next_set_id,
        "operator": operator,
        "checkedDimension": prediction.get("predictedDimension"),
        "inputSet": prev_request.input_set_id,
        "reward": prediction.get("reward", 0),
        "requestData": next_request.dict(),
        "curiosityReward": prediction.get("curiosityReward"),
        "utility": prediction.get("utility"),
        "uniformity": prediction.get("uniformity"),
        "novelty": prediction.get("novelty"),
        "distance": prediction.get("distance"),
        "utilityWeights": prediction.get("utility_weights"),
        "galaxy_class_score": prediction.get("galaxy_class_score"),
        "class_score_found_12": prediction.get("class_score_found_12"),
        "class_score_found_15": prediction.get("class_score_found_15"),
        "class_score_found_18": prediction.get("class_score_found_18"),
        "class_score_found_21": prediction.get("class_score_found_21"),
    }

    return next_node, next_request


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
    for mode, sampling_method in combinations([MODES, SAMPLING_METHODS]):
        for target_set_name in scan_folders(
            MODEL_FOLDER / "sampling" / sampling_method
        ):
            with (target_set_name / mode / "info.json").open("r") as f:
                info = json.load(f)
            for model_path in scan_folders(target_set_name / mode):
                model_type = model_path.stem
                pipeline, size = [], randrange(min_pipeline_size, max_pipeline_size)
                request_data = _get_initial_request_data(database_pipeline_cache, info)
                logger.info(f"Generating pipeline using {model_path} of size {size}")
                for i in range(1, size + 1):
                    model_manager = ModelManager(
                        database_pipeline_cache["galaxies"], 
                        model_path
                    )
                    try:
                        node, request_data = next_pipeline_iter(
                            database_pipeline_cache, model_manager, request_data
                        )
                        pipeline.append(node)
                    except ValueError:
                        print(f"Unexpectedly exited from pipeline generation on step {i}. Saving pipeline as it is...")
                        break
                pipeline_filename = f"{target_set_name.stem}_{mode}_{model_type}.json"
                write_pipeline(pipeline_filename, pipeline, sampling_method)
                logger.info(f"Finished gathering pipeline with {model_path}")
    logger.info(f"Finished generating trained pipelines")
