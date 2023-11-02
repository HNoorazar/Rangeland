import numpy as np
import pandas as pd

import time, datetime
from pprint import pprint
import os, os.path, sys


def clean_census(df, col_):
    """
    Census data is weird;
        - Column can have ' (D)' or ' (Z)' in it.
        - Numbers are as strings.
    """
    df = df[df[col_] != " (D)"]
    df = df[df[col_] != " (Z)"]
    if type(df[col_][0]) == str:
        df[col_] = df[col_].str.replace(",", "")
        df[col_] = df[col_].astype(float)
    return df
