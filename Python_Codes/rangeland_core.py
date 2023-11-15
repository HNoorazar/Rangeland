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
    df.reset_index(drop=True, inplace=True)
    df = df[df[col_] != " (D)"]
    df = df[df[col_] != " (Z)"]
    df.reset_index(drop=True, inplace=True)
    if type(df[col_][0]) == str:
        df[col_] = df[col_].str.replace(",", "")
        df[col_] = df[col_].astype(float)
    return df


def correct_Mins_FIPS(df, col_):
    """
    Min has added a leading 1 to FIPS
    since some FIPs starts with 0.

    Get rid of 1 and convert to strings.
    """
    df[col_] = df[col_].astype("str")
    df[col_] = df[col_].str.slice(1)
    return df
