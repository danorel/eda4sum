import json
import pandas as pd
import pathlib
import typing as t

from tqdm import tqdm

from data_types.pipeline import PipelineBodyItem, PipelineHead
from data_types.sampling import SamplingMethod
from utils.debugging import logger


def read_pipelines() -> t.Iterator[
    t.Tuple[str, PipelineHead, t.List[PipelineBodyItem]]
]:
    root = pathlib.Path.cwd()
    pipeline_dir = root / "rl" / "pipelines"
    logger.info("Starting reading pipelines...")
    for pipeline_path in tqdm(pipeline_dir.rglob("*.json")):
        pipeline_file = pathlib.Path(pipeline_path)
        filename, pipeline = (
            pipeline_file.name.split("_")[1],
            json.loads(pipeline_file.read_text()),
        )
        pipeline_head, *pipeline_body = pipeline
        yield filename, pipeline_head, pipeline_body
    logger.info("Finished reading pipelines...")


def read_target_sets(sampling_method: SamplingMethod) -> t.Iterator[t.Set[str]]:
    root = pathlib.Path.cwd()
    target_set_dir = root / "rl" / "targets" / sampling_method
    logger.info("Starting reading target sets...")
    for target_set_path in tqdm(target_set_dir.rglob("*.json")):
        target_set_file = pathlib.Path(target_set_path)
        yield set(json.loads(target_set_file.read_text()))
    logger.info("Finished reading target sets...")


def read_definitions() -> pd.DataFrame:
    root = pathlib.Path.cwd()
    definitions_path = root / "app" / "data" / "sdss" / "galaxies_index" / "groups.csv"
    logger.info("Starting reading definitions...")
    definitions_df = pd.read_csv(definitions_path)
    logger.info("Finished reading definitions...")
    return definitions_df


def read_members() -> pd.DataFrame:
    root = pathlib.Path.cwd()
    members = root / "app" / "data" / "sdss" / "galaxies_100000_index" / "groups.csv"
    logger.info("Starting reading members...")
    members_df = pd.read_csv(members)
    logger.info("Finished reading members...")
    return members_df
