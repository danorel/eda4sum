import typing as t
import typing_extensions as te

from pydantic import BaseModel

from data_types.annotation import PipelineAnnotation

ID: te.TypeAlias = str
Operator = te.Literal["by_facet", "by_neighbors"]
Dimension = te.Literal["i", "r", "z"]
TargetSet = te.Literal["Scattered"]


class Predicate(te.TypedDict):
    dimension: Dimension
    value: str


class InputSet(te.TypedDict):
    length: int
    id: int
    predicate: t.List[Predicate]
    silhouette: t.Optional[t.Any]
    novelty: t.Optional[t.Any]


class RequestData(BaseModel):
    dataset_to_explore: str
    input_set_id: t.Optional[int] = None
    dimensions: t.Optional[t.List[str]] = None
    get_scores: t.Optional[bool] = False
    get_predicted_scores: t.Optional[bool] = False
    target_set: t.Optional[str] = None
    curiosity_weight: t.Optional[float] = None
    found_items_with_ratio: t.Optional[t.Dict[str, float]] = None
    target_items: t.Optional[t.List[str]] = None
    previous_set_states: t.Optional[t.List[t.List[float]]] = None
    previous_operation_states: t.Optional[t.List[t.List[float]]] = None
    seen_predicates: t.Optional[t.List[str]] = []
    dataset_ids: t.Optional[t.List[int]] = None
    seen_sets: t.Optional[t.List[int]] = [],
    utility_weights: t.Optional[t.List[float]] = [0.333, 0.333, 0.334],
    previous_operations: t.Optional[t.List[str]] = [],
    decreasing_gamma: t.Optional[bool] = False,
    galaxy_class_scores: t.Optional[t.Dict[str, float]] = None,
    weights_mode: t.Optional[str] = None


class PipelineHead(te.TypedDict):
    selectedSetId: t.Optional[str]
    operator: str
    checkedDimension: str
    url: str
    inputSet: t.Optional[InputSet]
    reward: float
    curiosityReward: float


class PipelineBodyItem(PipelineHead):
    requestData: RequestData


class AnnotatedPipelineBodyItem(PipelineBodyItem):
    annotation: PipelineAnnotation


Pipeline = t.List[PipelineHead]
PipelineType = te.Literal["dora", "eda4sum"]
