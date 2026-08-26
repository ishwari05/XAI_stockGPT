from typing import List, Sequence
import numpy as np
import pandas as pd


def compute_lagged_returns(
    close: pd.Series, 
    horizons: Sequence[int] = (1, 5, 10, 20), 
    log_returns: bool = False
) -> pd.DataFrame:
    """
    Computes trailing/lagged returns as features.
    These use only historical information available at time t:
    return_hd = (Close[t] - Close[t-h]) / Close[t-h]
    
    NEVER uses forward shift.
    """
    res = pd.DataFrame(index=close.index)
    for h in horizons:
        col_name = f"return_{h}d"
        if log_returns:
            res[col_name] = np.log(close / close.shift(h))
        else:
            res[col_name] = close.pct_change(h)
    return res
