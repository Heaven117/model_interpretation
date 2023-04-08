from enum import Enum, auto


class Tasktype(Enum):
    """
    Type of data that is going to be explained by the
    anchor.
    """
    TABULAR = auto()
    IMAGE = auto()
    TEXT = auto()
