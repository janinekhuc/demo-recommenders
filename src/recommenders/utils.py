import pandas as pd
import redshift_connector
from typing import Any


def exclusive_or(left: Any, right: Any) -> bool:
    """Test if either left or right is true, but not both"""
    return (left or right) and not (left and right)
