# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Nov 7.
#
# On Nov. 6 Mike wanted to model cattle inventory using only ```NPP``` and rangeland area for one year.

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

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
seasonal_dir = reOrganized_dir + "seasonal_variables/02_merged_mean_over_county/"

# %%

# %%
Bhupi = pd.read_csv(param_dir + "Bhupi_25states_clean.csv")
Bhupi["SC"] = Bhupi.state + "-" + Bhupi.county

print (f"{len(Bhupi.state.unique()) = }")
print (f"{len(Bhupi.county_fips.unique()) = }")
Bhupi.head(2)

# %%

# %%
SoI = ["Alabama", "Arkansas", "California", 
       "Colorado", "Florida", "Georgia", "Idaho",
       "Illinois", "Iowa", "Kansas", "Kentucky",
       "Louisiana", "Mississippi", "Missouri", "Montana", 
       "Nebraska", "New Mexico", "North Dakota", 
       "Oklahoma", "Oregon", "South Dakota", "Tennessee",
       "Texas", "Virginia", "Wyoming"]

abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")

SoI_abb = []
for x in SoI:
    SoI_abb = SoI_abb + [abb_dict["full_2_abb"][x]]

# %%
USDA_data = pickle.load(open(reOrganized_dir + "USDA_data.sav", "rb"))

cattle_inventory = USDA_data["cattle_inventory"]

# pick only 25 states we want
cattle_inventory = cattle_inventory[cattle_inventory.state.isin(SoI)].copy()

print (cattle_inventory.data_item.unique())
print (cattle_inventory.commodity.unique())
print (cattle_inventory.year.unique())

census_years = list(cattle_inventory.year.unique())
# pick only useful columns
cattle_inventory = cattle_inventory[["year", "county_fips", "cattle_cow_inventory"]]

print (f"{len(cattle_inventory.county_fips.unique()) = }")
cattle_inventory.head(2)

# %% [markdown]
# ### Min has an extra "1" as leading digit in FIPS!!

# %%
# county_annual_GPP_NPP_productivity = pd.read_csv(reOrganized_dir + "county_annual_GPP_NPP_productivity.csv")
# county_annual_GPP_NPP_productivity.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
# county_annual_GPP_NPP_productivity.head(2)

# NPP = pd.read_csv(reOrganized_dir + "county_annual_GPP_NPP_productivity.csv")
NPP = pd.read_csv(Min_data_base + "county_annual_MODIS_NPP.csv")
NPP.head(2)

# %%
# pick only census years
NPP = NPP[NPP.year.isin(census_years)]
NPP.reset_index(drop=True, inplace=True)
NPP.head(2)

# %%
county_id_name_fips = pd.read_csv(Min_data_base + "county_id_name_fips.csv")
county_id_name_fips = county_id_name_fips[county_id_name_fips.STATE.isin(SoI_abb)].copy()

county_id_name_fips.sort_values(by=["STATE", "county"], inplace=True)
county_id_name_fips.reset_index(drop=True, inplace=True)
county_id_name_fips.head(2)

# %%
print (f"{len(NPP.county.unique()) = }")

# %%
print (NPP.shape)
NPP = NPP[NPP.county.isin(list(county_id_name_fips.county.unique()))].copy()
print (NPP.shape)
NPP.head(2)

# %%
print (f"{(NPP.year.unique()) = }")

# %%
print (f"{len(NPP.county.unique()) = }")
print (f"{len(cattle_inventory.county_fips.unique()) = }")

# %%
for a_year in NPP.year.unique():
    df = NPP[NPP.year == a_year]
    print (f"{len(df.county.unique()) = }")

# %%
NPP.head(2)

# %%
NPP = rc.correct_Mins_FIPS(df=NPP, col_="county")
NPP.rename(columns={"county": "county_fips"}, inplace=True)
NPP.head(2)

# %%
# Rangeland area and Total area:
county_RA_and_TA_fraction = pd.read_csv(reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv")
print (county_RA_and_TA_fraction.shape)
county_RA_and_TA_fraction.head(5)

# %%
county_RA_and_TA_fraction.rename(columns={"fips_id": "county_fips"}, inplace=True)
county_RA_and_TA_fraction = rc.correct_Mins_FIPS(df=county_RA_and_TA_fraction, col_="county_fips")
county_RA_and_TA_fraction.head(2)

# %%
county_annual_NPP_Ra = pd.merge(NPP, county_RA_and_TA_fraction,
                                on=["county_fips"],
                                how="left")
county_annual_NPP_Ra.head(2)

# %%
cattle_inventory = cattle_inventory[cattle_inventory.year.isin(list(county_annual_NPP_Ra.year.unique()))]
cattle_inventory.year.unique()

# %%
print (len(cattle_inventory.county_fips.unique()))
print (len(county_annual_NPP_Ra.county_fips.unique()))

# %%
cattle_inventory_cnty_missing_from_NPP = [x for x in cattle_inventory.county_fips.unique()\
                                          if not(x in county_annual_NPP_Ra.county_fips.unique())]
len(cattle_inventory_cnty_missing_from_NPP)

# %%
NPP_cnty_missing_from_cattle = [x for x in county_annual_NPP_Ra.county_fips.unique()\
                                          if not(x in cattle_inventory.county_fips.unique())]
len(NPP_cnty_missing_from_cattle)

# %%
print ("01001" in list(county_annual_NPP_Ra.county_fips.unique()))
print ("01001" in list(cattle_inventory.county_fips.unique()))

# %%

# %%
county_annual_MODIS_NPP_Ra_cattleInv = pd.merge(county_annual_NPP_Ra, cattle_inventory,
                                                on=["county_fips", "year"],
                                                how="left")

print (f"{cattle_inventory.shape = }")
print (f"{county_annual_NPP_Ra.shape = }")
print (f"{county_annual_MODIS_NPP_Ra_cattleInv.shape = }")
county_annual_MODIS_NPP_Ra_cattleInv.head(2)

# %%
