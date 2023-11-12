import pandas as pd
import typing as t

from data_types.pipeline import PipelineEda4Sum, PipelineItemEda4Sum, AnnotatedPipelineEda4Sum, AnnotatedPipelineItemEda4Sum
from data_types.annotation import Annotation


def find_item_set(
        members: pd.DataFrame, pipeline_body_item: PipelineItemEda4Sum
) -> t.Set[str]:
    input_set_id = pipeline_body_item["inputSet"]["id"]
    members = members.loc[members["id"] == input_set_id]["members"].iloc[0]
    input_set = set(members[1:-1].split(", "))
    return input_set


def _find_remaining_operators(pipeline: PipelineEda4Sum) -> dict:
    operators = {}
    for pipeline_item in pipeline:
        current_operator = pipeline_item["operator"]
        if current_operator in operators.keys():
            operators[current_operator] += 1
        else:
            operators[current_operator] = 1

    return operators


def _find_delta_uniformity(pipeline_item_current: PipelineItemEda4Sum,
                           pipeline_item_next: PipelineItemEda4Sum) -> float:
    return pipeline_item_next["uniformity"] - pipeline_item_current["uniformity"]


def _find_delta_novelty(pipeline_item_current: PipelineItemEda4Sum,
                        pipeline_item_next: PipelineItemEda4Sum) -> float:
    return pipeline_item_next["novelty"] - pipeline_item_current["novelty"]


def _find_delta_diversity(pipeline_item_current: PipelineEda4Sum,
                          pipeline_item_next: PipelineEda4Sum) -> float:
    return pipeline_item_next["distance"] - pipeline_item_current["distance"]


def annotate_pipeline(pipeline: PipelineEda4Sum) -> AnnotatedPipelineEda4Sum:
    length = len(pipeline)
    annotated_pipeline: AnnotatedPipelineEda4Sum = []

    for item in range(length):
        operators = _find_remaining_operators(pipeline[item:])
        if item is not length - 1:
            delta_uniformity = _find_delta_uniformity(
                pipeline[item], pipeline[item + 1])
            delta_novelty = _find_delta_novelty(
                pipeline[item], pipeline[item + 1])
            delta_diversity = _find_delta_diversity(
                pipeline[item], pipeline[item + 1])
        else:
            delta_uniformity = 0
            delta_novelty = 0
            delta_diversity = 0
        annotation = Annotation(total_length=length, remaining_operators=operators,
                                delta_uniformity=delta_uniformity, delta_novelty=delta_novelty,
                                delta_diversity=delta_diversity)
        annotated_pipeline_item = AnnotatedPipelineItemEda4Sum(
            **pipeline[item], annotation=annotation)
        annotated_pipeline.append(annotated_pipeline_item)

    return annotated_pipeline
