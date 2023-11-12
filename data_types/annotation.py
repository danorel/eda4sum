import typing_extensions as te


"""
TODO: find whether these properties could be derrived from pipeline/model:

    final_familiarity: float
    final_curisity: float
    delta_familiarity: float
    delta_curisity: float
"""


class Annotation(te.TypedDict):
    total_length: int
    remaining_operators: dict
    delta_novelty: float
    delta_uniformity: float
    delta_diversity: float
