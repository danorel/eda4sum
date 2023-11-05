import pandas as pd
import typing as t
import os
import random
import uuid

from data_types.pipeline import PipelineBodyItem
from utils.data_reader import read_members, read_pipelines
from utils.data_writer import write_target_set
from utils.debugging import logger


NODE_SAMPLING_RATE = float(os.environ.get("NODE_SAMPLING_RATE", "0.5"))
SAMPLE_AMOUNT = int(os.environ.get("SAMPLE_AMOUNT", "5"))
SAMPLE_BY = os.environ.get("SAMPLE_BY", "csv")


def node_sampling_pipeline(members: pd.DataFrame, pipeline_body: t.List[PipelineBodyItem]) -> t.Set[str]:
    logger.info("Sampling by pipelines...")

    leaf = random.randint(0, len(pipeline_body) - 1)
    pipeline_body_item = pipeline_body[leaf]

    input_set_id = pipeline_body_item["inputSet"]["id"]
    members = members.loc[members["id"] == input_set_id]["members"]
    if members.empty:
        return set([])
    item_set = set([int(member)
                    for member in members.iloc[0][1:-1].split(", ")])

    sampled_item_set = random.choices(list(item_set), k=(
        round(len(item_set) * NODE_SAMPLING_RATE)))
    return sampled_item_set


def node_sampling_csv(members: pd.DataFrame) -> t.Set[str]:
    logger.info("Sampling by csv...")

    input_set_id = random.randint(0, members.shape[0] - 1)
    item_set = set([int(member)
                    for member in members.iloc[input_set_id, 2][1:-1].split(", ")])

    sampled_item_set = random.choices(list(item_set), k=(
        round(len(item_set) * NODE_SAMPLING_RATE)))
    return sampled_item_set


if __name__ == "__main__":
    logger.info("Node sampling started...")

    members = read_members()

    if SAMPLE_BY == 'pipeline':
        for filename, pipeline_head, pipeline_body in read_pipelines():
            target_set = node_sampling_pipeline(members, pipeline_body)
            write_target_set(
                filename,
                target_set=target_set,
                sampling_method="node_sampling",
            )
    elif SAMPLE_BY == 'csv':
        for _ in range(SAMPLE_AMOUNT):
            target_set = node_sampling_csv(members)
            write_target_set(
                filename=f"{str(uuid.uuid4())}.json",
                target_set=target_set,
                sampling_method="node_sampling",
            )

    logger.info("Node sampling is done and saved!")
