import numpy as np
import pandas as pd

import time, datetime
from pprint import pprint
import os, os.path, sys


def covert_unitNPP_2_toal(NPP_df, npp_col_, area_col_, new_col_):
    meterSq_2_acr = 0.000247105
    NPP_df[new_col_] = (NPP_df[npp_col_] * NPP_df[area_col_]) / meterSq_2_acr
    return NPP_df


def census_stateCntyAnsi_2_countyFips(df):
    df["state_ansi"] = df["state_ansi"].astype("int32")
    df["county_ansi"] = df["county_ansi"].astype("int32")

    df["state_ansi"] = df["state_ansi"].astype("str")
    df["county_ansi"] = df["county_ansi"].astype("str")

    for idx in df.index:
        if len(df.loc[idx, "state_ansi"]) == 1:
            df.loc[idx, "state_ansi"] = "0" + df.loc[idx, "state_ansi"]

        if len(df.loc[idx, "county_ansi"]) == 1:
            df.loc[idx, "county_ansi"] = "00" + df.loc[idx, "county_ansi"]
        elif len(df.loc[idx, "county_ansi"]) == 2:
            df.loc[idx, "county_ansi"] = "0" + df.loc[idx, "county_ansi"]

    df["county_fips"] = df["state_ansi"] + df["county_ansi"]
    return df


def clean_census(df, col_):
    """
    Census data is weird;
        - Column can have ' (D)' or ' (Z)' in it.
        - Numbers are as strings.
    """
    df.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
    if "state" in df.columns:
        df.state = df.state.str.title()
    if "county" in df.columns:
        df.county = df.county.str.title()

    df.reset_index(drop=True, inplace=True)
    col_ = col_.lower()
    df = df[df[col_] != " (D)"]
    df = df[df[col_] != " (Z)"]
    df.reset_index(drop=True, inplace=True)
    if type(df[col_][0]) == str:
        df[col_] = df[col_].str.replace(",", "")
        df[col_] = df[col_].astype(float)

    if (
        ("state_ansi" in df.columns)
        and ("county_ansi" in df.columns)
        and not ("county_fips" in df.columns)
    ):
        df = census_stateCntyAnsi_2_countyFips(df)

    return df


def correct_Mins_FIPS(df, col_):
    """
    Min has added a leading 1 to FIPS
    since some FIPs starts with 0.

    Get rid of 1 and convert to strings.
    """
    df[col_] = df[col_].astype("str")
    df[col_] = df[col_].str.slice(1)

    ## if county name is missing, that is for
    ## all of state. or sth. drop them. They have ' ' in them, no NA!
    if "county_name" in df.columns:
        df = df[df.county_name != " "].copy()
        df.reset_index(drop=True, inplace=True)
    return df



def correct_4digitFips(df, col_):
    df[col_] = df[col_].astype('str')
    for idx in df.index:
        if len(df.loc[idx, col_]) == 4:
            df.loc[idx, col_] = "0" + df.loc[idx, col_]
    return df




def correct_3digitStateFips_Min(df, col_):
    # Min has an extra 1 in his data. just get rid of it.
    df[col_] = df[col_].astype('str')
    df[col_] = df[col_].str.slice(1, 3)
    return df



