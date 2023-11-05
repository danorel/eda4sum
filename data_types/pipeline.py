import typing as t
import typing_extensions as te

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


class RequestData(te.TypedDict):
    get_scores: bool
    get_predicted_scores: bool
    seen_predicates: t.List[str]
    input_set_id: int
    dimensions: t.List[str]
    target_set: str
    curiosity_weight: float
    target_items: t.List[ID]
    found_items_with_ratio: t.Dict[ID, float]
    previous_set_states: t.List[t.List[float]]
    previous_operation_states: t.List[t.List[float]]


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
