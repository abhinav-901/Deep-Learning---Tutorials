import numpy as np
from typing import List


def softmax(*scores: List[float]) -> List[float]:
    """ takes list of float scores(probablity) and calculate softmax

    Returns:
        List[float]: list of individual softmax scores
    """
    exp_sum: float = np.sum(np.array([np.exp(score) for score in scores]))
    softmax_score: List[float] = [
        np.divide(np.exp(score), exp_sum) for score in scores
    ]
    return softmax_score


if __name__ == "__main__":
    softmax(.8, -.9, .2, .1)
