# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Wed. 29th or Tues 28 of Nov
#
# Mike and I met in my office. Looked at annual plots of inventories.
# Then saw a big dip in annual data that will be missed on cencus data.
# So, here we model annual data

# %%
import shutup
shutup.please()

import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

sys.path.append("/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/")
import rangeland_core as rc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %% [markdown]
# # Read

# %%
SoI = ["Alabama", "Arkansas", "California", 
       "Colorado", "Florida", "Georgia", 
       "Idaho", "Illinois", "Iowa", 
       "Kansas", "Kentucky", "Louisiana", 
       "Mississippi", "Missouri", "Montana", 
       "Nebraska", "New Mexico", "North Dakota", 
       "Oklahoma", "Oregon", "South Dakota", 
       "Tennessee", "Texas", "Virginia", 
       "Wyoming"]

abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI_abb = []
for x in SoI:
    SoI_abb = SoI_abb + [abb_dict["full_2_abb"][x]]

# %%
county_id_name_fips = pd.read_csv(Min_data_base + "county_id_name_fips.csv")
county_id_name_fips.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)

county_id_name_fips = county_id_name_fips[county_id_name_fips.state.isin(SoI_abb)].copy()

county_id_name_fips.sort_values(by=["state", "county"], inplace=True)

county_id_name_fips = rc.correct_Mins_FIPS(df=county_id_name_fips, col_="county")
county_id_name_fips.rename(columns={"county": "county_fips"}, inplace=True)

county_id_name_fips.reset_index(drop=True, inplace=True)
print (len(county_id_name_fips.state.unique()))
county_id_name_fips.head(2)

# %%
county_id_name_fips["state_fip"] = county_id_name_fips.county_fips.str.slice(0, 2)
county_id_name_fips.head(2)

# %%

# %%
herbRatio = pd.read_csv(data_dir_base + "Supriya/Nov30_HerbRatio/herbRatio.csv")
herbRatio.dropna(how="any", inplace=True)
herbRatio = rc.correct_4digitFips(df=herbRatio, col_="county_fips")
herbRatio.head(2)

# %%
NPP = pd.read_csv(Min_data_base + "statefips_annual_MODIS_NPP.csv")
NPP.rename(columns={"NPP": "modis_npp",
                    "statefips90m": "state_fip"}, 
           inplace=True)
NPP = rc.correct_3digitStateFips_Min(NPP, "state_fip")

NPP = NPP[NPP.state_fip.isin(county_id_name_fips.state_fip)]
NPP.reset_index(drop=True, inplace=True)

NPP.head(2)

# %%
# Rangeland area and Total area:
county_RA_and_TA_fraction = pd.read_csv(reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv")
county_RA_and_TA_fraction.rename(columns={"fips_id": "county_fips"}, inplace=True)

county_RA_and_TA_fraction = rc.correct_Mins_FIPS(df=county_RA_and_TA_fraction, col_="county_fips")
L = len(county_RA_and_TA_fraction.county_fips.unique())
print ("number of counties are {}.".format(L))
print (county_RA_and_TA_fraction.shape)
county_RA_and_TA_fraction.head(2)

# %%
county_RA_and_TA_fraction["state_fip"] = county_RA_and_TA_fraction.county_fips.str.slice(0, 2)
county_RA_and_TA_fraction.head(2)

# %%
state_RA = county_RA_and_TA_fraction[["state_fip", "rangeland_acre"]].groupby("state_fip").sum()
state_RA.reset_index(drop=False, inplace=True)
state_RA = state_RA[state_RA.state_fip.isin(county_id_name_fips.state_fip)]
state_RA.reset_index(drop=True, inplace=True)
state_RA.head(2)

# %%
print (len(NPP.state_fip.unique()))
print (len(state_RA.state_fip.unique()))

# %%
NPP.head(2)

# %%
state_RA.head(2)

# %%
state_annual_NPP_Ra = pd.merge(NPP, state_RA, on = ["state_fip"], how = "left")
state_annual_NPP_Ra.head(2)

# %%
shannon_annual = pd.read_csv(reOrganized_dir + "Beef_Cows_fromCATINV.csv")
shannon_annual = shannon_annual[shannon_annual.state.isin(SoI_abb)]
shannon_annual.head(2)

# %%
wanted_years = np.arange(state_annual_NPP_Ra.year.min(), state_annual_NPP_Ra.year.max())
print (wanted_years[:3])
print (wanted_years[-3:])
cols_ = ["state"] + list(wanted_years.astype(str))
shannon_annual = shannon_annual[cols_]


shannon_annual = shannon_annual[shannon_annual.state.isin(list(county_id_name_fips.state.unique()))]
shannon_annual.sort_values(by=["state"], inplace=True)
shannon_annual.reset_index(drop=True, inplace=True)
shannon_annual.head(2)

# %%
shannon_annual.transpose.size()

# %%
