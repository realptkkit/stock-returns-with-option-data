from typing import Tuple
import numpy as np
import tensorflow as tf


def r_squared(target_returns, predictions) -> Tuple[float, float, float]:
    """_summary_

    Args:
        target_returns (_type_): _description_
        predictions (_type_): _description_

    Returns:
        Tuple[float, float, float]: oss, rss, tss
    """
    rss = np.sum((target_returns-predictions)**2)
    tss = np.sum(target_returns**2)

    oos = 1 - rss/tss

    return oos, rss, tss


def r_squared_fn(y_true, y_pred):
    rss = tf.reduce_sum((y_true-y_pred)**2)
    tss = tf.reduce_sum(y_true**2)

    oos = 1 - rss/tss

    return oos
