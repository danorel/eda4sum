import typing_extensions as te


class PipelineBodyAnnotation(te.TypedDict):
    final_curiosity: float
    average_curiosity: float
    final_familiarity: float
    average_familiarity: float
    total_length_of_pipeline: int


class PipelineBodyItemAnnotation(te.TypedDict):
    target_set_rate: float
    delta_curisity: float
    delta_familiarity: float
    remaining_operators: str


class PipelineAnnotation(PipelineBodyAnnotation, PipelineBodyItemAnnotation):
    pass
