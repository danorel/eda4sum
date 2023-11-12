import json
import pandas as pd
import pathlib
import typing as t

from tqdm import tqdm

from data_types.pipeline import Pipeline, PipelineType, PipelineKind
from data_types.sampling import SamplingMethod
from utils.debugging import logger


def read_pipelines(
    sampling_method: SamplingMethod, 
    pipeline_type: PipelineType = "eda4sum", 
    pipeline_kind: PipelineKind = "raw"
) -> t.Iterator[t.Tuple[str, Pipeline]]:
    root = pathlib.Path.cwd()
    pipeline_dir = (
        root 
        / "rl" 
        / "pipelines"
        / "sdss"
        / "sampling"
        / sampling_method
        / pipeline_type
        / pipeline_kind
    )
    logger.info(f"starting reading pipelines folder: {pipeline_dir}")
    for pipeline_path in tqdm(pipeline_dir.rglob("*.json")):
        filename, pipeline = (
            '_'.join(pipeline_path.name.split("_")[1:]),
            json.loads(pipeline_path.read_text()),
        )
        yield filename, pipeline
    logger.info(f"finished reading pipelines folder: {pipeline_dir}")


def read_target_set_names(sampling_method: SamplingMethod) -> t.Iterator[t.Set[str]]:
    root = pathlib.Path.cwd()
    target_set_dir = (
        root 
        / "rl" 
        / "targets"
        / "sdss"
        / "sampling"
        / sampling_method
    )
    logger.info(f"started reading target set folder: {target_set_dir}")
    for target_set_path in tqdm(target_set_dir.rglob("**/*.json")):
        target_set_subpath = str(target_set_path).split("/")[-3:]
        target_set_file_path = "/".join(target_set_subpath).split(".")[0]
        logger.info(f"found target set: {target_set_file_path}")
        yield target_set_file_path
    logger.info(f"finished reading target set folder: {target_set_dir}")


def read_definitions() -> pd.DataFrame:
    root = pathlib.Path.cwd()
    definitions_path = root / "app" / "data" / "sdss" / "galaxies_index" / "groups.csv"
    logger.info("Starting reading definitions.csv...")
    definitions_df = pd.read_csv(definitions_path)
    logger.info("Finished reading definitions.csv...")
    return definitions_df


def read_members() -> pd.DataFrame:
    root = pathlib.Path.cwd()
    members = root / "app" / "data" / "sdss" / "galaxies_100000_index" / "groups.csv"
    logger.info("Starting reading members.csv...")
    members_df = pd.read_csv(members)
    logger.info("Finished reading members.csv...")
    return members_df
