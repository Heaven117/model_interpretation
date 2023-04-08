from dataclasses import dataclass


@dataclass()
class AnchorCandidate:
    """
    Reprensents a possible candidate in the process of finding the best anchor.
    """

    feature_mask: list
    precision: float = 0
    n_samples: int = 0
    positive_samples: int = 0
    coverage: float = -1

    def __init__(self, feature_mask):
        self.feature_mask = feature_mask
        self.covered_true = []
        self.covered_false = []

    def update_precision(self, positives: int, n_samples: int, covered_true, covered_false):
        """Updatest the precision of this AnchorCandidate.

        Args:
            positives (int): Number of correct predictions 正确预测次数
            n_samples (int): Number of predictions 预测次数
        """
        self.n_samples += n_samples
        self.positive_samples += positives
        self.precision = self.positive_samples / self.n_samples
        self.covered_true.extend(covered_true)
        self.covered_false.extend(covered_false)

    def append_feature(self, feature: int):
        """Appends feature index to feature mask.

        Args:
            feature (int): Index of the feature
        """
        self.feature_mask.append(feature)
