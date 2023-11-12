from data_types.pipeline import PipelineType
from utils.data_reader import read_pipelines
from utils.data_writer import write_pipeline
from utils.debugging import logger

from ..constants import SAMPLING_METHODS
from .annotate import annotate_pipeline


def annotate_pipelines(pipeline_type: PipelineType = "eda4sum"):
    for sampling_method in SAMPLING_METHODS:
        logger.info(f"{pipeline_type} annotation has been started for {sampling_method}")

        for filename, pipeline in read_pipelines(sampling_method, pipeline_type, "raw"):
            annotated_pipeline = annotate_pipeline(pipeline)
            write_pipeline(filename, annotated_pipeline, sampling_method, pipeline_type, "annotated")

        logger.info(f"{pipeline_type} annotation is done and saved for {sampling_method}!")
