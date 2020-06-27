import numpy as np
from typing import List


def crossentropy(Y: List[int], P: List[float]) -> float:
    """ calculate crossentropy loss

    Args:
        Y (List[int]): event occured the 1 else 0
        P (List[float]): probablities associated with occurence of the event

    Returns:
        float: [description]
    """
    cross_entropy = -sum([(y * np.log(p) + (1 - y) * np.log(1 - p))
                          for y, p in zip(Y, P)])
    return cross_entropy


if __name__ == '__main__':
    crossentropy([0, 0, 1], [.7, 0.7, 0.1])
