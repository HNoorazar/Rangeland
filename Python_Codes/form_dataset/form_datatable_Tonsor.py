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
import os, os.path, pickle, sys

import matplotlib
import matplotlib.pyplot as plt

sys.path.append('/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/')
import rangeland_core as rc

# import geopandas
# A = geopandas.read_file("/Users/hn/Desktop/amin/shapefile.shp")
# A.head(2)

# %% [markdown]
# According to Google Map the coordinate ```30.59375N 88.40625W``` is in Alabama. According to Bhupi's file it
# is in Alabama by state but in Mississipi by ```fips```. That place is on the border of the two states. I will go by fips.

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
# Min_data_dir_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"
seasonal_dir = reOrganized_dir + "seasonal_variables/02_merged_mean_over_county/" 

# %%
# file_name = "countyMean_seasonalVars_wFips.csv"
# countyMean_seasonalVars = pd.read_csv(seasonal_dir + file_name)
# print (f"{len(countyMean_seasonalVars.state.unique())=}")

# # round numbers
#  countyMean_seasonalVars = countyMean_seasonalVars.round(decimals=2)
# countyMean_seasonalVars.head(2)

# %%
# file_name = "countyMean_seasonalVars_wFips.sav"
# countyMean_seasonalVars = pickle.load(open(seasonal_dir + file_name, "rb"))
# del(file_name)
# countyMean_seasonalVars = countyMean_seasonalVars["countyMean_seasonalVars_wFips"]
# countyMean_seasonalVars.head(2)

# %%
# Min_FIPS = pd.read_csv("/Users/hn/Documents/01_research_data/Sid/Analog/parameters/Min_counties.csv")
# Min_FIPS = Min_FIPS[["state", "county", "fips"]]
# Min_FIPS.drop_duplicates(inplace=True)
# Min_FIPS.reset_index(drop=True, inplace=True)
# Min_FIPS.head(2)

# %%
# Min_FIPS[Min_FIPS.state == "AL"].sort_values(by=['county'])

# %%
Bhupi = pd.read_csv(param_dir + "Bhupi_25states_clean.csv")
Bhupi["SC"] = Bhupi.state + "-" + Bhupi.county
Bhupi.head(2)

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
cntyMean_seasonVars_wide = pickle.load(open(seasonal_dir + "wide_seasonal_vars_cntyMean_wFips.sav", "rb"))
cntyMean_seasonVars_wide = cntyMean_seasonVars_wide["wide_seasonal_vars_cntyMean_wFips"]

cntyMean_seasonVars_wide.sort_values(by=['state', 'county', 'year'], inplace=True)
cntyMean_seasonVars_wide.reset_index(drop=True, inplace=True)
cntyMean_seasonVars_wide = cntyMean_seasonVars_wide.round(decimals=2)
cntyMean_seasonVars_wide.head(2)

# %%
len(cntyMean_seasonVars_wide.county_fips.unique())

# %%
USDA_data = pickle.load(open(reOrganized_dir + "USDA_data.sav", "rb"))
print (USDA_data.keys())
feed_expense = USDA_data["feed_expense"]
AgLand = USDA_data["AgLand"]
wetLand_area = USDA_data["wetLand_area"]
cattle_inventory = USDA_data["cattle_inventory"]
# FarmOperation = USDA_data["FarmOperation"] # not needed. create by NASS guy.

# %%
feed_expense.rename(columns={"value":"feed_expense", 
                             "cv_(%)":"feed_expense_cv_(%)"}, inplace=True)

wetLand_area.rename(columns={"value":"CRP_wetland_acr", 
                             "cv_(%)":"CRP_wetland_acr_cv_(%)"}, inplace=True)

cattle_inventory.rename(columns={"value":"cattle_cow_inventory", 
                             "cv_(%)":"cattle_cow_inventory_cv_(%)"}, inplace=True)

# %%
print (f"{AgLand.shape = }")
AgLand = rc.clean_census(df=AgLand, col_="value")
print (f"{AgLand.shape = }")
print ()

print (f"{feed_expense.shape = }")
feed_expense = rc.clean_census(df=feed_expense, col_="feed_expense")
print (f"{feed_expense.shape = }")
print ()

print (f"{wetLand_area.shape = }")
wetLand_area = rc.clean_census(df=wetLand_area, col_="CRP_wetland_acr")
print (f"{wetLand_area.shape = }")
print ()

print (f"{cattle_inventory.shape = }")
cattle_inventory = rc.clean_census(df=cattle_inventory, col_="cattle_cow_inventory")
print (f"{cattle_inventory.shape = }")

# %%
cattle_inventory.head(2)

# %%
feed_expense[(feed_expense.state=="Alabama") & (feed_expense.county=="Baldwin")]

# %%
feed_expense[(feed_expense.state=="Alabama") & (feed_expense.county=="Washington")]

# %%
print (f"{len(feed_expense.state.unique())=}")
feed_expense.data_item.unique()

# %%
feed_expense.head(2)

# %%
feed_expense = feed_expense[feed_expense.state.isin(SoI)].copy()

print (feed_expense.shape)
print(len(feed_expense.state.unique()))
print(len(feed_expense.county.unique()))
print(len(feed_expense.year.unique()))

feed_expense.head(2)

# %%
print (cattle_inventory.shape)
cattle_inventory = cattle_inventory[cattle_inventory.state.isin(SoI)].copy()

print (cattle_inventory.shape)
print(len(cattle_inventory.state.unique()))
print(len(cattle_inventory.county.unique()))
print(len(cattle_inventory.year.unique()))

# %% [markdown]
# ### List all counties and years we want so we can fill the damn gaps

# %%
season_counties = cntyMean_seasonVars_wide.county_fips.unique()
print (f"{len(season_counties) = }")


# %%
### Subset seasonal vars to every 5 years that is in the USDA NASS

USDA_years = list(feed_expense.year.unique())
seasonal_5yearLapse = cntyMean_seasonVars_wide[cntyMean_seasonVars_wide.year.isin(USDA_years)].copy()
seasonal_5yearLapse.reset_index(drop=True, inplace=True)
seasonal_5yearLapse.head(2)

# %%

# %% [markdown]
# ### Fill the gaps 
#
# - First plan:  find counties for which there is less than 4 instances (2017, 2012, 2007, 2002, 1997).
# - Second plan: Mike said if a county has a missing value forget about it.

# %%
a_cnty = AgLand.county_fips.unique()[0]

A = AgLand[AgLand.county_fips == a_cnty]
aa = A.year.unique()
missin_yrs = [x for x in USDA_years if x not in aa]

aa

# %%
len(USDA_years) 

# %%
len(aa) < (len(USDA_years))

# %%
#
# if min_yrs_needed is len(USDA_years) then we will have 
# all the data for all years must be fully present
# if min_yrs_needed is (len(USDA_years) - 1) then we allow one year
# of missing data and we need to fill it with interpolation.
#
min_yrs_needed = len(USDA_years)

# %%
AgLand_cnty_toss = {}

for a_cnty in AgLand.county_fips.unique():
    A = AgLand[AgLand.county_fips == a_cnty]
    aa = A.year.unique()
    missin_yrs = [x for x in USDA_years if x not in aa]
    if len(aa) < (min_yrs_needed):
        AgLand_cnty_toss[a_cnty] = missin_yrs

AgLand_cnty_toss

# %%
feed_expense_cnty_toss = {}

for a_cnty in feed_expense.county_fips.unique():
    A = feed_expense[feed_expense.county_fips == a_cnty]
    aa = A.year.unique()
    missin_yrs = [x for x in USDA_years if x not in aa]
    if len(aa) < (min_yrs_needed):
        feed_expense_cnty_toss[a_cnty] = missin_yrs

feed_expense_cnty_toss

# %%
wetLand_area_cnty_toss = {}

for a_cnty in wetLand_area.county_fips.unique():
    A = wetLand_area[wetLand_area.county_fips == a_cnty]
    aa = A.year.unique()
    missin_yrs = [x for x in USDA_years if x not in aa]
    if len(aa) < (min_yrs_needed):
        wetLand_area_cnty_toss[a_cnty] = missin_yrs

print (f"{len(wetLand_area_cnty_toss) = }")
wetLand_area_cnty_toss

# %%
cattle_inventory_cnty_toss = {}

for a_cnty in cattle_inventory.county_fips.unique():
    A = cattle_inventory[cattle_inventory.county_fips == a_cnty]
    aa = A.year.unique()
    missin_yrs = [x for x in USDA_years if x not in aa]
    if len(aa) < (min_yrs_needed):
        cattle_inventory_cnty_toss[a_cnty] = missin_yrs

print (f"{len(cattle_inventory_cnty_toss) = }")
cattle_inventory_cnty_toss

# %%

# %%
# missing_cnty_in_feed_expense = [x for x in season_counties if x not in feed_expense.county_fips.values]
# missing_cnty_in_AgLand = [x for x in season_counties if x not in AgLand.county_fips.values]
# missing_cnty_in_cattle = [x for x in season_counties if x not in cattle_inventory.county_fips.values]
# missing_cnty_in_wetLand = [x for x in season_counties if x not in wetLand_area.county_fips.values]

# missing_cnty_in_feed_expense_set = set(missing_cnty_in_feed_expense)
# missing_cnty_in_AgLand_set = set(missing_cnty_in_AgLand)
# missing_cnty_in_cattle_set = set(missing_cnty_in_cattle)
# missing_cnty_in_wetLand_set = set(missing_cnty_in_wetLand)

# print (f"{len(missing_cnty_in_feed_expense_set) = }")
# print (f"{len(missing_cnty_in_AgLand_set) = }")
# print (f"{len(missing_cnty_in_cattle_set) = }")
# print (f"{len(missing_cnty_in_wetLand_set) = }")

# intersection is correct? or union?
#
# USDA_missing_counties = missing_cnty_in_feed_expense_set.\
#            intersection(missing_cnty_in_AgLand_set).\
#            intersection(missing_cnty_in_cattle).\
#            intersection(missing_cnty_in_wetLand_set)

# print (len(USDA_missing_counties))

# [x for x in missing_cnty_in_feed_expense_set if not (x in USDA_missing_counties)]

# cntyMean_seasonVars_wide = cntyMean_seasonVars_wide[~cntyMean_seasonVars_wide.county_fips.
#                                                     isin(list(missing_cnty_in_wetLand_set))]

# seasonal_5yearLapse = seasonal_5yearLapse[~seasonal_5yearLapse.county_fips.
#                                                     isin(list(missing_cnty_in_wetLand_set))]

# season_counties = cntyMean_seasonVars_wide.county_fips.unique()
# print (f"{len(season_counties) = }")

# [x for x in missing_cnty_in_wetLand_set if x not in missing_cnty_in_cattle_set]

# AgLand = AgLand[~AgLand.county_fips.isin(list(missing_cnty_in_cattle_set))]
# feed_expense = feed_expense[~feed_expense.county_fips.isin(list(missing_cnty_in_cattle_set))]
# wetLand_area = wetLand_area[~wetLand_area.county_fips.isin(list(missing_cnty_in_cattle_set))]
# cattle_inventory = cattle_inventory[~cattle_inventory.county_fips.isin(list(missing_cnty_in_cattle_set))]
# seasonal_5yearLapse = seasonal_5yearLapse[~seasonal_5yearLapse.county_fips.isin(list(
#                                                                            missing_cnty_in_cattle_set))]

# feed_expense_cnty_w_missingYears = {}

# for a_cnty in feed_expense.county_fips.unique():
#     A = feed_expense[feed_expense.county_fips == a_cnty]
#     aa = A.year.values
#     missin_yrs = [x for x in USDA_years if x not in aa]
#     if len(aa)<5:
#         feed_expense_cnty_w_missingYears[a_cnty] = missin_yrs

# feed_expense_cnty_w_missingYears


# feed_expense_cnty_w_missingYears = {}

# for a_cnty in feed_expense.county_fips.unique():
#     A = feed_expense[feed_expense.county_fips == a_cnty]
#     aa = A.year.values
#     missin_yrs = [x for x in USDA_years if x not in aa]
#     if len(aa)<5:
#         feed_expense_cnty_w_missingYears[a_cnty] = missin_yrs

# feed_expense_cnty_w_missingYears

# wetLand_area_cnty_w_missingYears = {}

# for a_cnty in wetLand_area.county_fips.unique():
#     A = wetLand_area[wetLand_area.county_fips == a_cnty]
#     aa = A.year.values
#     missin_yrs = [x for x in USDA_years if x not in aa]
#     if len(aa)<5:
#         wetLand_area_cnty_w_missingYears[a_cnty] = missin_yrs

# wetLand_area_cnty_w_missingYears


# cattle_inventory_cnty_w_missingYears = {}

# for a_cnty in cattle_inventory.county_fips.unique():
#     A = cattle_inventory[cattle_inventory.county_fips == a_cnty]
#     aa = A.year.values
#     missin_yrs = [x for x in USDA_years if x not in aa]
#     if len(aa)<5:
#         cattle_inventory_cnty_w_missingYears[a_cnty] = missin_yrs

# cattle_inventory_cnty_w_missingYears


# AgLand = AgLand[~AgLand.county_fips.isin(list(cattle_inventory_cnty_w_missingYears))]

# feed_expense = feed_expense[~feed_expense.county_fips.isin(list(cattle_inventory_cnty_w_missingYears))]

# wetLand_area = wetLand_area[~wetLand_area.county_fips.isin(list(cattle_inventory_cnty_w_missingYears))]

# cattle_inventory = cattle_inventory[~cattle_inventory.county_fips.isin(
#     list(cattle_inventory_cnty_w_missingYears))]

# seasonal_5yearLapse = seasonal_5yearLapse[~seasonal_5yearLapse.county_fips.isin(
#     list(cattle_inventory_cnty_w_missingYears))]

# feed_expense_cnty_w_missingYears = {}

# for a_cnty in feed_expense.county_fips.unique():
#     A = feed_expense[feed_expense.county_fips == a_cnty]
#     aa = A.year.values
#     missin_yrs = [x for x in USDA_years if x not in aa]
#     if len(aa)<5:
#         feed_expense_cnty_w_missingYears[a_cnty] = missin_yrs

# feed_expense_cnty_w_missingYears

# AgLand_cnty_w_missingYears = {}

# for a_cnty in AgLand.county_fips.unique():
#     A = AgLand[AgLand.county_fips == a_cnty]
#     aa = A.year.values
#     missin_yrs = [x for x in USDA_years if x not in aa]
#     if len(aa)<5:
#         AgLand_cnty_w_missingYears[a_cnty] = missin_yrs

# AgLand_cnty_w_missingYears

# cattle_inventory_cnty_w_missingYears = {}

# for a_cnty in cattle_inventory.county_fips.unique():
#     A = cattle_inventory[cattle_inventory.county_fips == a_cnty]
#     aa = A.year.values
#     missin_yrs = [x for x in USDA_years if x not in aa]
#     if len(aa)<5:
#         cattle_inventory_cnty_w_missingYears[a_cnty] = missin_yrs

# cattle_inventory_cnty_w_missingYears

# bad_cnties = ["34017", "36005"]

# AgLand = AgLand[~AgLand.county_fips.isin(bad_cnties)]

# feed_expense = feed_expense[~feed_expense.county_fips.isin(bad_cnties)]

# wetLand_area = wetLand_area[~wetLand_area.county_fips.isin(bad_cnties)]

# cattle_inventory = cattle_inventory[~cattle_inventory.county_fips.isin(bad_cnties)]

# seasonal_5yearLapse = seasonal_5yearLapse[~seasonal_5yearLapse.county_fips.isin(bad_cnties)]

# seasonal_5yearLapse_w_missingYears = {}

# for a_cnty in seasonal_5yearLapse.county_fips.unique():
#     A = seasonal_5yearLapse[seasonal_5yearLapse.county_fips == a_cnty]
#     aa = A.year.values
#     missin_yrs = [x for x in USDA_years if x not in aa]
#     if len(aa)<5:
#         seasonal_5yearLapse_w_missingYears[a_cnty] = missin_yrs

# seasonal_5yearLapse_w_missingYears

# feed_expense_cnty_w_missingYears = {}

# for a_cnty in feed_expense.county_fips.unique():
#     A = feed_expense[feed_expense.county_fips == a_cnty]
#     aa = A.year.values
#     missin_yrs = [x for x in USDA_years if x not in aa]
#     if len(aa)<5:
#         feed_expense_cnty_w_missingYears[a_cnty] = missin_yrs

# feed_expense_cnty_w_missingYears

# AgLand_cnty_w_missingYears = {}

# for a_cnty in AgLand.county_fips.unique():
#     A = AgLand[AgLand.county_fips == a_cnty]
#     aa = A.year.values
#     missin_yrs = [x for x in USDA_years if x not in aa]
#     if len(aa)<5:
#         AgLand_cnty_w_missingYears[a_cnty] = missin_yrs

# AgLand_cnty_w_missingYears

# cattle_inventory_cnty_w_missingYears = {}

# for a_cnty in cattle_inventory.county_fips.unique():
#     A = cattle_inventory[cattle_inventory.county_fips == a_cnty]
#     aa = A.year.values
#     missin_yrs = [x for x in USDA_years if x not in aa]
#     if len(aa)<5:
#         cattle_inventory_cnty_w_missingYears[a_cnty] = missin_yrs

# cattle_inventory_cnty_w_missingYears

# %%
cntyMean_seasonVars_wide.head(2)

# %%
seasonal_5yearLapse.head(2)

# %% [markdown]
# ### Toss all the counties for which there is not enough data 

# %%
wetLand_area.head(2)

# %%
# check if there are missing values
# for a_year in USDA_years:
#     for season_counties


wetLand_area.data_item.unique()

# %% [markdown]
# # Do the analysis with and without wetland area
# since there are too many missing values.
#
# and ignore all the counties for which we do not have full cattle inventory data.

# %%
feed_expense_noWetLand = feed_expense.copy()
AgLand_noWetLand = AgLand.copy()
cattle_inventory_noWetLand = cattle_inventory.copy()
seasonal_5yearLapse_noWetLand = seasonal_5yearLapse.copy()

# %%
set(wetLand_area_cnty_toss.keys())
set(cattle_inventory_cnty_toss.keys())
set(AgLand_cnty_toss.keys())
set(feed_expense_cnty_toss.keys())

# %%
toss_counties = set(feed_expense_cnty_toss.keys()).\
          union(set(AgLand_cnty_toss.keys())).\
          union(set(cattle_inventory_cnty_toss.keys())).\
          union(set(wetLand_area_cnty_toss.keys()))

# %%
AgLand              = AgLand[~AgLand.county_fips.isin(toss_counties)]
wetLand_area        = wetLand_area[~wetLand_area.county_fips.isin(toss_counties)]
feed_expense        = feed_expense[~feed_expense.county_fips.isin(toss_counties)]
cattle_inventory    = cattle_inventory[~cattle_inventory.county_fips.isin(toss_counties)]
seasonal_5yearLapse = seasonal_5yearLapse[~seasonal_5yearLapse.county_fips.isin(toss_counties)]

# %% [markdown]
# ### now find all counties that they share.

# %%
common_counties = set(AgLand.county_fips.unique()).\
                  intersection(set(wetLand_area.county_fips.unique())).\
                  intersection(set(feed_expense.county_fips.unique())).\
                  intersection(set(cattle_inventory.county_fips.unique())).\
                  intersection(set(seasonal_5yearLapse.county_fips.unique()))

len(common_counties)

# %%
AgLand              = AgLand[AgLand.county_fips.isin(common_counties)]
wetLand_area        = wetLand_area[wetLand_area.county_fips.isin(common_counties)]
feed_expense        = feed_expense[feed_expense.county_fips.isin(common_counties)]
cattle_inventory    = cattle_inventory[cattle_inventory.county_fips.isin(common_counties)]
seasonal_5yearLapse = seasonal_5yearLapse[seasonal_5yearLapse.county_fips.isin(common_counties)]

# %%
print(f"{len(AgLand.county_fips.unique()) = }")
print(f"{len(wetLand_area.county_fips.unique()) = }")
print(f"{len(feed_expense.county_fips.unique()) = }")
print(f"{len(cattle_inventory.county_fips.unique()) = }")
print(f"{len(seasonal_5yearLapse.county_fips.unique()) = }")

# %% [markdown]
# #### Exclude wetland

# %%
toss_counties_excludeWetland = set(feed_expense_cnty_toss.keys()).\
                         union(set(AgLand_cnty_toss.keys())).\
                         union(set(cattle_inventory_cnty_toss.keys()))

# %%
AgLand_noWetLand              = AgLand_noWetLand[~AgLand_noWetLand.county_fips.isin(toss_counties_excludeWetland)]

feed_expense_noWetLand        = feed_expense_noWetLand[~
                                        feed_expense_noWetLand.county_fips.isin(toss_counties_excludeWetland)]

cattle_inventory_noWetLand    = cattle_inventory_noWetLand[~
                                    cattle_inventory_noWetLand.county_fips.isin(toss_counties_excludeWetland)]

seasonal_5yearLapse_noWetLand = seasonal_5yearLapse_noWetLand[~
                                 seasonal_5yearLapse_noWetLand.county_fips.isin(toss_counties_excludeWetland)]

# %%
common_counties_noWetLand = set(AgLand_noWetLand.county_fips.unique()).\
                  intersection(set(feed_expense_noWetLand.county_fips.unique())).\
                  intersection(set(cattle_inventory_noWetLand.county_fips.unique())).\
                  intersection(set(seasonal_5yearLapse_noWetLand.county_fips.unique()))

len(common_counties_noWetLand)

# %%
AgLand_noWetLand              = AgLand_noWetLand[AgLand_noWetLand.county_fips.isin(common_counties_noWetLand)]

feed_expense_noWetLand        = feed_expense_noWetLand[
                                   feed_expense_noWetLand.county_fips.isin(common_counties_noWetLand)]

cattle_inventory_noWetLand    = cattle_inventory_noWetLand[
                                    cattle_inventory_noWetLand.county_fips.isin(common_counties_noWetLand)]


seasonal_5yearLapse_noWetLand = seasonal_5yearLapse_noWetLand[
                                    seasonal_5yearLapse_noWetLand.county_fips.isin(common_counties_noWetLand)]

# %%
print(f"{len(AgLand_noWetLand.county_fips.unique()) = }")
print(f"{len(feed_expense_noWetLand.county_fips.unique()) = }")
print(f"{len(cattle_inventory_noWetLand.county_fips.unique()) = }")
print(f"{len(seasonal_5yearLapse_noWetLand.county_fips.unique()) = }")

# %% [markdown]
# ## Fill the gaps

# %%
min_yrs_needed

# %%
cattle_inventory_noWetLand.head(2)

# %%
A = cattle_inventory_noWetLand[["state", "year", "cattle_cow_inventory"]].groupby(
                                                        ["year", "state"]).sum().reset_index()

A.year = pd.to_datetime(A.year, format='%Y')
A.head(2)

# %%
A.set_index('year', inplace=True)
A.sort_values("cattle_cow_inventory", inplace=True)
A.head(2)

# %%
A[A.state.isin(A.state.unique()[:10])].groupby('state')['cattle_cow_inventory'].plot(legend=True);

# %%
A[A.state.isin(A.state.unique()[10:20])].groupby('state')['cattle_cow_inventory'].plot(legend=True);

# %%
# fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
#                    gridspec_kw={'hspace': 0.35, 'wspace': .05});

A[A.state.isin(A.state.unique()[20:])].groupby('state')['cattle_cow_inventory'].plot(legend=True);

# %%
A[A.state=="Texas"]

# %%
Beef_Cows_fromCATINV = pd.read_csv(reOrganized_dir + "Beef_Cows_fromCATINV.csv")
Beef_Cows_fromCATINV.head(2)

# %%
state_to_abbrev = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ",
                   "Arkansas": "AR", "California": "CA", "Colorado": "CO",
                   "Connecticut": "CT", "Delaware": "DE", "Florida": "FL",
                   "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
                   "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
                   "Kansas": "KS", "Kentucky": "KY",    "Louisiana": "LA",
                   "Maine": "ME",    "Maryland": "MD",    "Massachusetts": "MA",
                   "Michigan": "MI",    "Minnesota": "MN",    "Mississippi": "MS",
                   "Missouri": "MO",    "Montana": "MT",    "Nebraska": "NE",
                   "Nevada": "NV",    "New Hampshire": "NH",    "New Jersey": "NJ",
                   "New Mexico": "NM",    "New York": "NY",    "North Carolina": "NC",
                   "North Dakota": "ND",    "Ohio": "OH",    "Oklahoma": "OK",
                   "Oregon": "OR",    "Pennsylvania": "PA",    "Rhode Island": "RI",
                   "South Carolina": "SC",    "South Dakota": "SD",    "Tennessee": "TN",
                   "Texas": "TX",    "Utah": "UT",    "Vermont": "VT",
                   "Virginia": "VA",    "Washington": "WA",    "West Virginia": "WV",
                   "Wisconsin": "WI",    "Wyoming": "WY",    "District of Columbia": "DC",
                   "American Samoa": "AS",    "Guam": "GU",    "Northern Mariana Islands": "MP",
                   "Puerto Rico": "PR",    "United States Minor Outlying Islands": "UM",    
                   "U.S. Virgin Islands": "VI"}

states_abb_list = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                    'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                    'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                    'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                    'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

abb_2_full_dict = {'AK': 'Alaska',    'AL': 'Alabama',    'AR': 'Arkansas',    
                   'AZ': 'Arizona',    'CA': 'California',    'CO': 'Colorado',
                   'CT': 'Connecticut',    'DC': 'District of Columbia',    'DE': 'Delaware',
                   'FL': 'Florida',    'GA': 'Georgia',    'HI': 'Hawaii',
                   'IA': 'Iowa',    'ID': 'Idaho',    'IL': 'Illinois',
                   'IN': 'Indiana',    'KS': 'Kansas',    'KY': 'Kentucky',
                   'LA': 'Louisiana',    'MA': 'Massachusetts',    'MD': 'Maryland',
                   'ME': 'Maine',    'MI': 'Michigan',    'MN': 'Minnesota',
                   'MO': 'Missouri',    'MS': 'Mississippi',    'MT': 'Montana',
                   'NC': 'North Carolina',    'ND': 'North Dakota',    'NE': 'Nebraska',
                   'NH': 'New Hampshire',    'NJ': 'New Jersey',    'NM': 'New Mexico',
                   'NV': 'Nevada',    'NY': 'New York',    'OH': 'Ohio',
                   'OK': 'Oklahoma',    'OR': 'Oregon',    'PA': 'Pennsylvania',    'RI': 'Rhode Island',
                   'SC': 'South Carolina',    'SD': 'South Dakota',    'TN': 'Tennessee',
                   'TX': 'Texas',    'UT': 'Utah',    'VA': 'Virginia',
                   'VT': 'Vermont',    'WA': 'Washington',    'WI': 'Wisconsin',
                   'WV': 'West Virginia',    'WY': 'Wyoming'}

# %%
state_25_abb = [state_to_abbrev[x] for x in SoI]

# %%
Beef_Cows_fromCATINV = Beef_Cows_fromCATINV[Beef_Cows_fromCATINV.state.isin(state_25_abb)]
Beef_Cows_fromCATINV.reset_index(drop=True, inplace=True)
Beef_Cows_fromCATINV.head(2)

# %%
shannon_years = [str(x) for x in np.arange(1997, 2018)]
cols = ["state"] + shannon_years
Beef_Cows_fromCATINV = Beef_Cows_fromCATINV[cols]
Beef_Cows_fromCATINV.head(2)

# %%
fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=False, # sharey='col', # sharex=True, sharey=True,
                   gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs[0].grid(axis='y', which='both'); axs[1].grid(axis='y', which='both'); axs[2].grid(axis='y', which='both')

##########################################################################################

state_="TX"
axs[0].plot(pd.to_datetime(shannon_years, format='%Y'), 
         Beef_Cows_fromCATINV.loc[Beef_Cows_fromCATINV.state==state_, shannon_years].values[0],
         c="dodgerblue", linewidth=2, label=state_ + " Shannon");

state_ = "Texas"
B = A[A.state == state_].copy()
B.sort_index(inplace=True)
axs[0].plot(B.index, 
         B[B.state == state_].cattle_cow_inventory.values,
         c="red", linewidth=2, label=state_);

axs[0].legend(loc="best");
##########################################################################################
state_="AL"
axs[1].plot(pd.to_datetime(shannon_years, format='%Y'), 
         Beef_Cows_fromCATINV.loc[Beef_Cows_fromCATINV.state==state_, shannon_years].values[0],
         c="dodgerblue", linewidth=2, label=state_ + " Shannon");

state_ = "Alabama"
B = A[A.state == state_].copy()
B.sort_index(inplace=True)
axs[1].plot(B.index, 
         B[B.state == state_].cattle_cow_inventory.values,
         c="red", linewidth=2, label=state_);

axs[1].legend(loc="best");
##########################################################################################
state_="OK"
axs[2].plot(pd.to_datetime(shannon_years, format='%Y'), 
         Beef_Cows_fromCATINV.loc[Beef_Cows_fromCATINV.state==state_, shannon_years].values[0],
         c="dodgerblue", linewidth=2, label=state_ + " Shannon");

state_ = "Oklahoma"
B = A[A.state == state_].copy()
B.sort_index(inplace=True)
axs[2].plot(B.index, 
         B[B.state == state_].cattle_cow_inventory.values,
         c="red", linewidth=2, label=state_);

axs[2].legend(loc="best");

# %% [markdown]
# ## Merge different variables

# %%
seasonal_5yearLapse.head(2)

# %%
feed_expense.head(2)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Merge different variables

# %%
#
# Merge seasonal variables and feed expenses.
#
need_cols = ["year", "county_fips", "feed_expense", "feed_expense_cv_(%)"]
season_Feed = pd.merge(seasonal_5yearLapse, 
                       feed_expense[need_cols].drop_duplicates(), 
                       on=["year", "county_fips"], how='left')

del(need_cols)
print (f"{seasonal_5yearLapse.shape = }")
print (f"{feed_expense.shape = }")
print (f"{season_Feed.shape = }")
season_Feed.head(2)

# %%
feed_expense_FIPS_yr = feed_expense["county_fips"].astype(str)  + "_" + \
                          feed_expense["year"].astype(str)

seasonal_5yearLapse_FIPS_yr = seasonal_5yearLapse["county_fips"].astype(str)  + "_" + \
                                 seasonal_5yearLapse["year"].astype(str)

A = [x for x in list(seasonal_5yearLapse_FIPS_yr) if not (x in list(feed_expense_FIPS_yr))]
print (f"{len(A) = }")
A[:10]

# %%
season_Feed[season_Feed.county_fips=="27107"]

# %%
seasonal_5yearLapse[seasonal_5yearLapse.county_fips=="06075"]

# %%
season_Feed[season_Feed.county_fips=="06075"]

# %% [markdown]
# ### CRP
# ```Wetland``` file:
#
# Acres enrolled in Conservation Reserve, Wetlands Reserve, Farmable Wetlands, or Conservation Reserve Enhancement Programs, 1997-2017:\\
#
# https://quickstats.nass.usda.gov/#3A734A89-9829-3674-AFC6-C4764DF7B728

# %%
if "value" in wetLand_area.columns:
    wetLand_area.rename(columns={"value":"wetLand_area", 
                                 "cv_(%)":"wetLand_area_cv_(%)"}, 
                        inplace=True)

print(f"{wetLand_area.data_item.unique() = }")
wetLand_area.head(2)

# %%
wetLand_area_25state = wetLand_area[wetLand_area.state.isin(SoI)]
print (wetLand_area_25state.shape)
print(len(wetLand_area_25state.state.unique()))
print(len(wetLand_area_25state.county.unique()))
print(len(wetLand_area_25state.year.unique()))

# %%
need_cols = ["year", "county_fips", "wetLand_area", "wetLand_area_cv_(%)"]
season_Feed_CRP = pd.merge(season_Feed, 
                           wetLand_area[need_cols].drop_duplicates(), 
                            on=["year", "county_fips"], how='left')
del(need_cols)

# %%
wetLand_area.year.unique()

# %%
print (f"{season_Feed.shape = }")
print (f"{wetLand_area.shape = }")
print (f"{season_Feed_CRP.shape = }")

# %% [markdown]
# ### Irrigated Acre
#
# ```AgLand```. Irrigated acres and total land in farms by county, 1997-2017.
#
# https://quickstats.nass.usda.gov/#B2688D70-61AC-3E14-AA15-11882355E95E
#
# Ask for **Kirti**'s input.

# %%
print ("AgLand.data_item are {}".format(list(AgLand.data_item.unique())))
print ()
AgLand.head(2)

# %%
# wetLand_area_25state = wetLand_area[wetLand_area.state.isin(SoI)]
print (AgLand.shape)
print(len(AgLand.state.unique()))
print(len(AgLand.county.unique()))
print(len(AgLand.year.unique()))

# %%
AgLand_25state = AgLand[AgLand.state.isin(SoI)]
print (AgLand_25state.shape)
print(len(AgLand_25state.state.unique()))
print(len(AgLand_25state.county.unique()))
print(len(AgLand_25state.year.unique()))

# %%

# %% [markdown]
# # TEST
#
# Test and see what are the ```data_item```s here. Do they reflect percentages in WA counties?
#
# We need to add those in terms of percentage not absolute values.

# %%
# COI = ['Adams', 'Benton', 'Franklin', 'Grant', 'Walla Walla', 'Yakima']

# WSDA_SF_data_dir = "/Users/hn/Documents/01_research_data/00_shapeFiles/01_shapefiles_data_part_not_filtered/"
# WSDA_SF_data_dir_2 = "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/"

# WSDA_2018 = pd.read_csv(WSDA_SF_data_dir + "WSDA_DataTable_2018.csv")
# print (WSDA_2018.county.unique())
# WSDA_2018.head(2)

# %%
# WSDA_2018 = WSDA_2018[WSDA_2018.county.isin(COI)]
# len(WSDA_2018.ID.unique())

# %%
# all_eastern = pd.read_csv(WSDA_SF_data_dir_2 + "all_SF_data_concatenated.csv")

# all_eastern["SF_year"] = all_eastern.ID.str.split("_", expand=True)[3]
# all_eastern["SF_year"] = all_eastern["SF_year"].astype(int)

# all_eastern_2018 = all_eastern[all_eastern.SF_year == 2018].copy()
# print (len(all_eastern_2018.ID.unique()))
# print (all_eastern_2018.county.unique())
# all_eastern.head(2)

# %%
# all_eastern_total_Acr_perCounty = all_eastern[["ExctAcr", "county"]].groupby(["county"]).sum().reset_index()
# all_eastern_2018_total_Acr_perCounty = all_eastern_2018[["ExctAcr", "county"]].groupby(
#                                                            ["county"]).sum().reset_index()
# WSDA_2018_total_Acr_perCounty = WSDA_2018[["ExctAcr", "county"]].groupby(["county"]).sum().reset_index()

# %%
# sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
# import NASA_core as nc
# all_eastern_irr = nc.filter_out_nonIrrigated(all_eastern)
# all_eastern_2018_irr = nc.filter_out_nonIrrigated(all_eastern_2018)

# print (f"{all_eastern.shape = }")
# print (f"{all_eastern_irr.shape = }")
# all_eastern_irr.head(2)

# all_eastern_irr_Acr_perCounty = all_eastern_irr[["ExctAcr", "county"]].groupby(["county"]).sum().reset_index()
# all_eastern_2018_irr_Acr_perCounty = all_eastern_2018_irr[["ExctAcr", "county"]].groupby(["county"]
#                                                                                         ).sum().reset_index()

# all_eastern_irr_Acr_perCounty.rename(columns={"ExctAcr":"irr_area"}, inplace=True)
# all_eastern_2018_irr_Acr_perCounty.rename(columns={"ExctAcr":"irr_area"}, inplace=True)

# %%
# WSDA_areas_df = pd.merge(all_eastern_total_Acr_perCounty, 
#                          all_eastern_irr_Acr_perCounty, 
#                          on=["county"], how='left')
# WSDA_areas_df["irr_as_perc"] = (WSDA_areas_df.irr_area / WSDA_areas_df.ExctAcr)*100
# WSDA_areas_df.iloc[:, 1:] = WSDA_areas_df.iloc[:, 1:].round(2)
# WSDA_areas_df

# WSDA_2018_irr = nc.filter_out_nonIrrigated(WSDA_2018)
# WSDA_2018_irr_Acr_perCounty = WSDA_2018_irr[["ExctAcr", "county"]].groupby(["county"]).sum().reset_index()
# WSDA_2018_irr_Acr_perCounty.rename(columns={"ExctAcr":"irr_area"}, inplace=True)
# WSDA_2018_irr_Acr_perCounty

# WSDA_2018_areas_df = pd.merge(WSDA_2018_total_Acr_perCounty, 
#                               WSDA_2018_irr_Acr_perCounty, 
#                               on=["county"], how='left')
# WSDA_2018_areas_df["irr_as_perc"] = (WSDA_2018_areas_df.irr_area / WSDA_2018_areas_df.ExctAcr)*100
# WSDA_2018_areas_df.iloc[:, 1:] = WSDA_2018_areas_df.iloc[:, 1:].round(2)
# WSDA_2018_areas_df

# %%

# %%
# AgLand_WA = AgLand[AgLand.state == "Washington"].copy()
# AgLand_WA.rename(columns={"value":"area"}, inplace=True)
# AgLand_WA.area = AgLand_WA.area.replace(',','', regex=True)
# AgLand_WA.area = AgLand_WA.area.astype(int)
# AgLand_WA = AgLand_WA[AgLand_WA.county.isin(COI)]
# AgLand_irrigated_WA = AgLand_WA[AgLand_WA.data_item == "AG LAND, IRRIGATED - ACRES"].copy()
# AgLand_irrigated_WA.reset_index(drop=True, inplace=True)

# AgLand_FarmOper_WA = AgLand_WA[AgLand_WA.data_item == "FARM OPERATIONS - ACRES OPERATED"].copy()
# AgLand_FarmOper_WA.reset_index(drop=True, inplace=True)

# print (AgLand_irrigated_WA.shape)
# AgLand_irrigated_WA.head(2)

# %%
# AgLand_FarmOper_WA.head(2)

# %%
# AgLand_WA_FarmOper_acr_perCounty = AgLand_FarmOper_WA[["county", "area", "year"]].groupby(
#                                                     ["county", "year"]).sum().reset_index()
# AgLand_WA_FarmOper_acr_perCounty.head(2)

# %%
# AgLand_WA_irr_acr_perCounty = AgLand_irrigated_WA[["county", "area", "year"]].groupby(
#                                                      ["county", "year"]).sum().reset_index()
# AgLand_WA_irr_acr_perCounty.rename(columns={"area":"irr_area"}, inplace=True)

# %%
# Agland_areas_df = pd.merge(AgLand_WA_total_acr_perCounty, 
#                            AgLand_WA_irr_acr_perCounty, 
#                            on=["county", "year"], how='left')
# Agland_areas_df["irr_as_perc"] = (Agland_areas_df.irr_area / Agland_areas_df.area)*100
# Agland_areas_df.iloc[:, 1:] = Agland_areas_df.iloc[:, 1:].round(2)
# Agland_areas_df.head(10)

# %%
# Agland_areas_df2 = pd.merge(AgLand_WA_FarmOper_acr_perCounty, 
#                             AgLand_WA_irr_acr_perCounty, 
#                             on=["county", "year"], how='left')
# Agland_areas_df2["irr_as_perc"] = (Agland_areas_df2.irr_area / Agland_areas_df2.area)*100
# Agland_areas_df2.iloc[:, 1:] = Agland_areas_df2.iloc[:, 1:].round(2)
# Agland_areas_df2.head(10)

# %%
AgLand.rename(columns={"value":"area"}, inplace=True)

AgLand = AgLand[AgLand.area != ' (D)'].copy()
AgLand.area = AgLand.area.replace(',','', regex=True)
AgLand.area = AgLand.area.astype(int)

AgLand_FarmOper = AgLand[AgLand.data_item == "FARM OPERATIONS - ACRES OPERATED"].copy()
AgLand_FarmOper.reset_index(drop=True, inplace=True)

AgLand_irrigated = AgLand[AgLand.data_item == "AG LAND, IRRIGATED - ACRES"].copy()
AgLand_irrigated.reset_index(drop=True, inplace=True)
AgLand_irrigated.head(2)

# %%

# %%
# if "value" in wetLand_area.columns:
#     wetLand_area.rename(columns={"value":"irrigated_area", 
#                                  "cv_(%)":"irrigated_area_cv_(%)"}, 
#                         inplace=True)

# print(f"{wetLand_area.data_item.unique() = }")
# AgLand_irrigated.head(2)

# %%
# I do not know why I have "irrigated.columns" below.
# GitHub said wetLand. So, I fixed it above.
if "value" in AgLand_irrigated.columns:
    AgLand_irrigated.rename(columns={"value":"irrigated_area", 
                                     "cv_(%)":"irrigated_area_cv_(%)"}, 
                        inplace=True)

print(f"{AgLand_irrigated.data_item.unique() = }")
AgLand_irrigated.head(2)

# %%
AgLand_FarmOper_acr_perCounty = AgLand_FarmOper[["state", "county", "area", "year", "county_fips"]].groupby(
                                                    ["state", "county", "year", "county_fips"]).sum().reset_index()
AgLand_FarmOper_acr_perCounty.head(2)

AgLand_irr_acr_perCounty = AgLand_irrigated[["state", "county", "area", "year", "county_fips"]].groupby(
                                                     ["state", "county", "year", "county_fips"]).sum().reset_index()
AgLand_irr_acr_perCounty.rename(columns={"area":"irr_area"}, inplace=True)


# %%
Agland_areas_df = pd.merge(AgLand_FarmOper_acr_perCounty, 
                            AgLand_irr_acr_perCounty, 
                            on=["state", "county", "year", "county_fips"], how='left')
Agland_areas_df["irr_as_perc"] = (Agland_areas_df.irr_area / Agland_areas_df.area)*100
Agland_areas_df.iloc[:, 1:] = Agland_areas_df.iloc[:, 1:].round(2)
Agland_areas_df.head(5)

# %%
Agland_areas_df[Agland_areas_df.state=="Washington"].head(5)

# %%
season_Feed_CRP.head(2)

# %%
Agland_areas_df.head(2)

# %%
season_Feed_CRP.head(2)

# %%
need_cols = ["state", "county", "year", "county_fips", "irr_as_perc"]
season_Feed_CRP_irr = pd.merge(season_Feed_CRP, 
                               Agland_areas_df[need_cols].drop_duplicates(), 
                               on=["year", "county_fips", "state", "county"], how='left')
del(need_cols)

season_Feed_CRP_irr.head(2)

# %%

# %% [markdown]
# ### County Population
#
# This is population of people. I do not know how this is relevant. Each farm would send beef outside of the county, no?

# %%
population_file_names = [x for x in os.listdir(census_population_dir) if x.startswith("z")]
population_file_names

# %%
pop_2000_2010 = pd.read_csv(census_population_dir + "z_2000_2010_co-est00int-tot.csv", encoding= 'unicode_escape')
pop_2010_2020 = pd.read_csv(census_population_dir + "z_2010-2020-co-est2020.csv", encoding= 'unicode_escape')


pop_2000_2010.drop(['SUMLEV', 'REGION', "DIVISION", ], axis=1, inplace=True)
pop_2000_2010.rename(columns={"STATE": "state_fip", 
                              "COUNTY": "cnty_fip",
                              "STNAME":"state", 
                              "CTYNAME":"county"}, inplace=True)

pop_2010_2020.drop(['SUMLEV', 'REGION', "DIVISION", ], axis=1, inplace=True)
pop_2010_2020.rename(columns={"STATE": "state_fip", 
                              "COUNTY": "cnty_fip",
                              "STNAME":"state", 
                              "CTYNAME":"county"}, inplace=True)

# %%
pop_2000_2010.head(2)

# %%
pop_2010_2020.head(2)

# %%
pop_2000_2010.state_fip = pop_2000_2010.state_fip.astype(str)
pop_2000_2010.cnty_fip = pop_2000_2010.cnty_fip.astype(str)

for idx in pop_2000_2010.index:
    if len(pop_2000_2010.loc[idx, "state_fip"]) == 1:
        pop_2000_2010.loc[idx, "state_fip"] = "0" + pop_2000_2010.loc[idx, "state_fip"]
    
    col = "cnty_fip"
    if len(pop_2000_2010.loc[idx, col]) == 1:
        pop_2000_2010.loc[idx, col] = "00" + pop_2000_2010.loc[idx, col]
    elif len(pop_2000_2010.loc[idx, col]) == 2:
        pop_2000_2010.loc[idx, col] = "0" + pop_2000_2010.loc[idx, col]
        
pop_2000_2010.head(2)

# %%
pop_2010_2020.state_fip = pop_2010_2020.state_fip.astype(str)
pop_2010_2020.cnty_fip = pop_2010_2020.cnty_fip.astype(str)

for idx in pop_2010_2020.index:
    if len(pop_2010_2020.loc[idx, "state_fip"]) == 1:
        pop_2010_2020.loc[idx, "state_fip"] = "0" + pop_2010_2020.loc[idx, "state_fip"]
    
    col = "cnty_fip"
    if len(pop_2010_2020.loc[idx, col]) == 1:
        pop_2010_2020.loc[idx, col] = "00" + pop_2010_2020.loc[idx, col]
    elif len(pop_2010_2020.loc[idx, col]) == 2:
        pop_2010_2020.loc[idx, col] = "0" + pop_2010_2020.loc[idx, col]

pop_2010_2020.head(2)

# %% [markdown]
# ### ```cnty_fip == 000```
#
# presentes all state. i.e. sum of all counties. Drop them.

# %%
pop_2000_2010.drop(pop_2000_2010[pop_2000_2010.cnty_fip == "000"].index, inplace=True)
pop_2010_2020.drop(pop_2010_2020[pop_2010_2020.cnty_fip == "000"].index, inplace=True)

# %%
pop_2000_2010["county_fips"] = pop_2000_2010["state_fip"] + pop_2000_2010["cnty_fip"]
pop_2010_2020["county_fips"] = pop_2010_2020["state_fip"] + pop_2010_2020["cnty_fip"]

pop_2000_2010.drop(['state_fip', 'cnty_fip'], axis=1, inplace=True)
pop_2010_2020.drop(['state_fip', 'cnty_fip'], axis=1, inplace=True)

pop_2000_2010.head(2)

# %%
pop_2010_2020.head(2)

# %%
season_Feed_CRP_irr.year.unique()

# %%
pop_2000_2010 = pop_2000_2010[pop_2000_2010.state.isin(SoI)]
pop_2010_2020 = pop_2010_2020[pop_2010_2020.state.isin(SoI)]

pop_2000_2010.reset_index(drop=True, inplace=True)
pop_2010_2020.reset_index(drop=True, inplace=True)

# %%

# %%
need_cols_2000_2010 = ["county_fips", "POPESTIMATE2002", "POPESTIMATE2007"]
need_cols_2010_2020 = ["county_fips", "POPESTIMATE2012", "POPESTIMATE2017"]

# %%
pop_2000_2010 = pop_2000_2010[need_cols_2000_2010]
pop_2010_2020 = pop_2010_2020[need_cols_2010_2020]

# %%
pop_2000_2010.head(2)

# %%
pop_2010_2020.head(2)

# %%
print (f"{len(pop_2000_2010) = }")
print (f"{len(pop_2010_2020) = }")

# %%
A = [x for x in pop_2010_2020.county_fips if not(x in list(pop_2000_2010.county_fips))]
A

# %%
A = [x for x in pop_2000_2010.county_fips if not(x in list(pop_2010_2020.county_fips))]
A

# %%
print ("46113" in list(season_Feed_CRP_irr.county_fips))
print ("51515" in list(season_Feed_CRP_irr.county_fips))
print ("46102" in list(season_Feed_CRP_irr.county_fips))

# %%
A = [x for x in pop_2000_2010.county_fips if not(x in list(season_Feed_CRP_irr.county_fips))]
A

# %%
A = [x for x in pop_2010_2020.county_fips if not(x in list(season_Feed_CRP_irr.county_fips))]
A

# %%
L = len(pop_2000_2010) + len(pop_2010_2020)
pop_wide = pd.DataFrame(columns=["county_fips", "year", "population"], index=range(L*2))

pop_wide.county_fips = "666"
pop_wide.year = 666
pop_wide.population = 666

years_2000_2010 = [2002, 2007]*len(pop_2000_2010)
years_2010_2020 = [2012, 2017]*len(pop_2010_2020)
pop_wide.year = years_2000_2010 + years_2010_2020

pop_wide.head(5)

# %%
wide_pointer = 0 

for idx in pop_2000_2010.index:
    pop_wide.loc[wide_pointer, "county_fips"] = pop_2000_2010.loc[idx, "county_fips"]
    pop_wide.loc[wide_pointer, "population"] = pop_2000_2010.loc[idx, "POPESTIMATE2002"]
    
    pop_wide.loc[wide_pointer+1, "county_fips"] = pop_2000_2010.loc[idx, "county_fips"]
    pop_wide.loc[wide_pointer+1, "population"] = pop_2000_2010.loc[idx, "POPESTIMATE2007"]
    wide_pointer += 2
    
for idx in pop_2010_2020.index:
    pop_wide.loc[wide_pointer, "county_fips"] = pop_2010_2020.loc[idx, "county_fips"]
    pop_wide.loc[wide_pointer, "population"] = pop_2010_2020.loc[idx, "POPESTIMATE2012"]
    
    pop_wide.loc[wide_pointer+1, "county_fips"] = pop_2010_2020.loc[idx, "county_fips"]
    pop_wide.loc[wide_pointer+1, "population"] = pop_2010_2020.loc[idx, "POPESTIMATE2017"]
    wide_pointer += 2

# %%
season_Feed_CRP_irr.head(2)

# %%
pop_wide.head(2)

# %%
# AgLand_25state = AgLand[AgLand.state.isin(SoI)]
print (pop_wide.shape)
print(len(pop_wide.county_fips.unique()))
# print(len(pop_wide.county.unique()))
# print(len(pop_wide.year.unique()))

# %%
print (season_Feed_CRP_irr.shape)
print(len(season_Feed_CRP_irr.county_fips.unique()))

# %%
season_Feed_CRP_irr_pop = pd.merge(season_Feed_CRP_irr, pop_wide, 
                                   on=["year", "county_fips"], how='left')

season_Feed_CRP_irr_pop.head(2)

# %%
print (f"{season_Feed_CRP_irr.shape = }")
print (f"{season_Feed_CRP_irr_pop.shape = }")
season_Feed_CRP_irr_pop.head(5)

# %% [markdown]
# ### Beef Cow Inventory (heads)
#
#  - Total Beef Cow inventory: https://quickstats.nass.usda.gov/#ADD6AB04-62EF-3DBF-9F83-5C23977C6DC7
#  - Inventory of Beef Cows: https://quickstats.nass.usda.gov/#B2688D70-61AC-3E14-AA15-11882355E95E

# %%
# totalBeefCowInv = pd.read_csv(USDA_data_dir + "totalBeefCowInv.csv")
# totalBeefCowInv.head(2)

# %%
print (len(totalBeefCowInv.state.unique()))
print (len(totalBeefCowInv.county.unique()))
print (len(totalBeefCowInv.year.unique()))

totalBeefCowInv.head(2)

# %%
totalBeefCowInv.rename(columns={"value":"total_beefCowInv", 
                             "cv_(%)":"total_beefCowInv_(%)"}, inplace=True)
totalBeefCowInv.head(2)

# %%
totalBeefCowInv = totalBeefCowInv[totalBeefCowInv.state.isin(SoI)].copy()

print (totalBeefCowInv.shape)
print(len(totalBeefCowInv.state.unique()))
print(len(totalBeefCowInv.county.unique()))
print(len(totalBeefCowInv.year.unique()))

totalBeefCowInv.head(2)

# %%
need_cols = ["year", "county_fips", "total_beefCowInv", "total_beefCowInv_(%)"]
season_Feed_CRP_irr_pop_beef = pd.merge(season_Feed_CRP_irr_pop, 
                                        totalBeefCowInv[need_cols].drop_duplicates(), 
                                        on=["year", "county_fips"], how='left')

# %%
season_Feed_CRP_irr_pop_beef.head(2)

# %%

# %%
