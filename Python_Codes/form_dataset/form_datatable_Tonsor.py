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
# I want to have a direct stab at forming the datatables so they are usable for training the model.
# Let's just do it.

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os, pickle

# %%
# import geopandas
# A = geopandas.read_file("/Users/hn/Desktop/amin/shapefile.shp")
# A.head(2)

# %% [markdown]
# According to Google Map the coordinate ```30.59375N 88.40625W``` is in Alabama. According to Bhupi's file it
# is in Alabama by state but in Mississipi by ```fips```. That place is on the border of the two states. I will go by fips.

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_dir = data_dir_base + "census/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_dir_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
# file_name = "countyMean_seasonalVars_wFips.csv"
# countyMean_seasonalVars = pd.read_csv(reOrganized_dir + 
#                                        "seasonal_variables/02_merged_mean_over_county/" + file_name)
# print (f"{len(countyMean_seasonalVars.state.unique())=}")

# # round numbers
#  = countyMean_seasonalVars.round(decimals=2)
# countyMean_seasonalVars.head(2)

# %%
# file_name = "countyMean_seasonalVars_wFips.sav"
# countyMean_seasonalVars = pickle.load(open(reOrganized_dir + \
#                                            "seasonal_variables/02_merged_mean_over_county/" \
#                                            + file_name, "rb"))
# del(file_name)
# countyMean_seasonalVars = countyMean_seasonalVars["countyMean_seasonalVars_wFips"]
# countyMean_seasonalVars.head(2)

# %%
FIPS = pd.read_csv("/Users/hn/Documents/01_research_data/Sid/Analog/parameters/Min_counties.csv")
FIPS = FIPS[["state", "county", "fips"]]
FIPS.drop_duplicates(inplace=True)
FIPS.reset_index(drop=True, inplace=True)
FIPS.head(2)

# %%
SoI = ["Alabama", "Arkansas", 
       "California", "Colorado", 
       "Florida", "Georgia",
       "Idaho", "Illinois", 
       "Iowa", "Kansas", 
       "Kentucky", "Louisiana", 
       "Mississippi", "Missouri", 
       "Montana", "Nebraska", 
       "New Mexico", "North Dakota", 
       "Oklahoma", "Oregon", 
       "South Dakota", "Tennessee", 
       "Texas", "Virginia", "Wyoming"]

# %%
Bhupi = pd.read_csv(param_dir + "Bhupi_25states_clean.csv")
Bhupi["SC"] = Bhupi.state + "-" + Bhupi.county
Bhupi.head(2)

# %%

# %%
cntyMean_seasonVars_wide = pickle.load(open(reOrganized_dir + \
                                          "seasonal_variables/02_merged_mean_over_county/" + \
                                          "wide_seasonal_vars_cntyMean_wFips.sav", "rb"))
cntyMean_seasonVars_wide = cntyMean_seasonVars_wide["wide_seasonal_vars_cntyMean_wFips"]

# %%
# cntyMean_seasonVars_wide = pd.read_csv(reOrganized_dir + "wide_seasonal_vars_cntyMean.csv")
# cntyMean_seasonVars_wide.head(2)

# %%
USDA_data = pickle.load(open(reOrganized_dir + "USDA_data.sav", "rb"))

# %%
feed_expense = USDA_data["feed_expense"]
AgLand = USDA_data["AgLand"]
wetLand_area = USDA_data["wetLand_area"]

# FarmOperation = USDA_data["FarmOperation"] # not needed. create by NASS guy.

# %%
feed_expense.head(2)

# %%
# feed_expense = pd.read_csv(reOrganized_dir + "USDA_feed_expense_cleaned_01.csv")
# feed_expense.head(3)

# %%
feed_expense[(feed_expense.state=="Alabama") & (feed_expense.county=="Baldwin")]

# %%
feed_expense[(feed_expense.state=="Alabama") & (feed_expense.county=="Washington")]

# %%
print (f"{len(feed_expense.state.unique())=}")
feed_expense.data_item.unique()

# %%
feed_expense.rename(columns={"value":"feed_expense", 
                             "cv_(%)":"feed_expense_cv_(%)"}, inplace=True)
feed_expense.head(2)

# %%
feed_expense = feed_expense[feed_expense.state.isin(SoI)].copy()

# %%
### Subset seasonal vars to every 5 years that is in the USDA NASS
USDA_years = list(feed_expense.year.unique())
seasonal_5yearLapse = cntyMean_seasonVars_wide[cntyMean_seasonVars_wide.year.isin(USDA_years)].copy()

# %%
seasonal_5yearLapse.head(2)

# %%
feed_expense.head(2)

# %%
#
# Merge seasonal variables and feed expenses.
#
need_cols = ["year", "county_fips", "feed_expense", "feed_expense_cv_(%)"]
season_Feed = pd.merge(seasonal_5yearLapse, 
                       feed_expense[need_cols].drop_duplicates(), 
                       on=["year", "county_fips"], how='left')

print (f"{seasonal_5yearLapse.shape = }")
print (f"{feed_expense.shape = }")
print (f"{season_Feed.shape = }")
season_Feed.head(2)

# %%
feed_expense["FIPS_yr"] = feed_expense["county_fips"].astype(str)  + "_" + \
                          feed_expense["year"].astype(str)

# %%
seasonal_5yearLapse["FIPS_yr"] = seasonal_5yearLapse["county_fips"].astype(str)  + "_" + \
                                 seasonal_5yearLapse["year"].astype(str)

# %%
seasonal_5yearLapse

# %%
A = [x for x in list(seasonal_5yearLapse.FIPS_yr) if not (x in list(feed_expense.FIPS_yr))]
len(A)

# %%
A[:10]

# %%
feed_expense[feed_expense.county_fips==6075]

# %%
seasonal_5yearLapse[seasonal_5yearLapse.county_fips==6075]

# %%
season_Feed[season_Feed.county_fips==6075]

# %%
wetLand_area = pd.read_csv(reOrganized_dir  + "USDA_wetLand_area_cleaned_01.csv")
print (wetLand_area.data_item.unique())
wetLand_area.head(5)

# %%

# %%

# %%
