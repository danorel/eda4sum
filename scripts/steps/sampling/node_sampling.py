import pandas as pd
import typing as t
import typing_extensions as te
import random
import uuid

from data_types.pipeline import Pipeline
from utils.data_reader import read_members, read_pipelines
from utils.data_writer import write_target_set
from utils.debugging import logger


def make_item_set_sampler(sampling_rate: float):

    def sample(item_set: set):
        k = round(len(item_set) * sampling_rate)
        return random.choices(list(item_set), k=k)

    return sample


def make_item_set_validator(min_item_set_nodes: int, max_item_set_nodes: int):

    def satisfies_conditions(item_set: set):
        return len(item_set) >= min_item_set_nodes and len(item_set) <= max_item_set_nodes

    return satisfies_conditions


def node_sampling_pipeline(
    members: pd.DataFrame,
    pipeline: Pipeline,
    sample,
    satisfies_conditions
) -> t.Set[str]:
    logger.info("Sampling by pipelines...")

    leaf = random.randint(0, len(pipeline) - 1)
    pipeline_leaf = pipeline[leaf]

    input_set_id = pipeline_leaf["inputSet"]["id"]
    members = members.loc[members["id"] == input_set_id]["members"]

    if members.empty:
        logger.warn(f"Node[id={input_set_id}] is missing in .csv")
        return None

    item_set = set([int(id) for id in members.iloc[0][1:-1].split(", ")])

    if not satisfies_conditions(item_set):
        logger.warn(f"Node[id={input_set_id}] not satisfies conditions")
        return None

    return sample(item_set)


def node_sampling_csv(
    members: pd.DataFrame,
    sample,
    satisfies_conditions,
    attempts: int = 1000
) -> t.Set[str]:
    logger.info("Sampling by csv...")

    item_set = set()
    for attempt in range(attempts):
        input_set_id = random.randint(0, members.shape[0] - 1)
        item_set = set(
            [int(id) for id in members.iloc[input_set_id, 2][1:-1].split(", ")])
        if satisfies_conditions(item_set):
            logger.info(
                f"Node[id={input_set_id}] satisfies the conditions (size = {len(item_set)})")
            break
        else:
            logger.info(
                f"Node[id={input_set_id}] violates the conditions (size = {len(item_set)}). Attempt #{attempt}...")

    if not len(item_set):
        logger.warn(
            f"Not found node satisfying conditions ({attempts} attemts done)")
        return None

    return sample(item_set)


def node_sampling(
    sample_amount: int,
    sampling_rate: float,
    min_item_set_nodes: int,
    max_item_set_nodes: int,
    sample_by: te.Literal['pipeline', 'csv'] = 'csv'
):
    logger.info("Node sampling started...")

    members = read_members()

    sample, satisfies_conditions = (
        make_item_set_sampler(sampling_rate),
        make_item_set_validator(min_item_set_nodes, max_item_set_nodes)
    )

    if sample_by == 'pipeline':
        for filename, pipeline in read_pipelines("eda4sum", "raw"):
            target_set = node_sampling_pipeline(
                members,
                pipeline,
                sample,
                satisfies_conditions
            )
            if target_set:
                write_target_set(
                    filename,
                    target_set=target_set,
                    sampling_method="node_sampling",
                )
    elif sample_by == 'csv':
        for _ in range(sample_amount):
            target_set = node_sampling_csv(
                members,
                sample,
                satisfies_conditions
            )
            if target_set:
                write_target_set(
                    filename=f"{str(uuid.uuid4())}.json",
                    target_set=target_set,
                    sampling_method="node_sampling",
                )
    else:
        raise NotImplementedError()

    logger.info("Node sampling is done and saved!")


if __name__ == "__main__":
    members = node_sampling(
        sample_amount=100,
        sampling_rate=0.5,
        min_item_set_nodes=100,
        max_item_set_nodes=1000,
        sample_by='csv'
    )
